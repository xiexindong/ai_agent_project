"""
LLM 结构化意图识别
==================
流水线第 4 步：只做「任务大类分类」，不猜具体工具。

【为什么只分类不猜工具】
    工具名会随业务不断增加，让 LLM 从几十上百个工具名里挑，既不稳定也难校验。
    这里只让它回答 6 选 1 的大类 + 几个辅助字段，具体候选交给 Tool-RAG 用向量召回 ——
    分类越粗，模型越稳；细活交给检索。

【输出固定 6 个字段】
    intent_type        6 选 1，见 IntentType
    intent_confidence  0~1，低于 0.5 时下游禁止调用任何工具
    need_memory        本轮是否需要历史记忆
    domain_tag         业务域，Tool-RAG 按它过滤候选
    keyword_list       关键词，用于 embedding 不可用时的关键词降级
    exclude_tool_list  黑名单，本轮不许被召回的工具

【异常降级】
    LLM 没配 key / 调用失败 / 返回非法 JSON → 走规则兜底，
    兜底结果 confidence 一律 ≤ 0.4（必然低于 0.5 阈值 = 不调工具），
    宁可让模型空手回答，也不让一个不确定的分类触发工具调用。
"""
import re
from typing import Any, Dict, List, Optional

from modules.common.logger import get_logger
from modules.pipeline.models import IntentResult, IntentType

_logger = get_logger("pipeline.intent")

# ----------------------------------------------------------------------
# 【分类提示词】三段式：先给类别定义，再给输出格式，最后给判断要点
#   明确写"不要回答用户的问题" —— 否则模型常常一边分类一边把答案也写了，
#   既浪费 token，又让输出混进非 JSON 文本，加大解析失败概率。
# ----------------------------------------------------------------------
_INTENT_SYSTEM = """你是一个请求分类器，负责判断用户输入的"任务大类"。
你只做分类，不要回答用户的问题，不要调用任何工具，不要输出解释。

【6 个类别定义】
- invalid         ：无效输入（纯乱码、空内容、无法理解的字符）
- chat            ：普通闲聊、常识问答，不需要外部数据和工具
- local_skill     ：需要调用本地函数工具才能完成（计算、查时间、查天气、查业务数据）
- rag             ：需要检索知识库/文档/制度才能回答
- mcp_remote      ：需要调用远程 MCP 技能才能完成（企业系统对接类操作）
- continue_memory ：承接上一轮对话继续（"继续"、"刚才那个再说细一点"）

【输出格式】严格输出如下 JSON，不要有多余字段、不要有解释文字：
{
  "intent_type": "从上面 6 个里选 1 个",
  "intent_confidence": 0.0 到 1.0 之间的小数,
  "need_memory": true 或 false,
  "domain_tag": "业务域英文标签，如 finance / hr / it / general，判断不出就填 general",
  "keyword_list": ["关键词1", "关键词2"],
  "exclude_tool_list": []
}

【判断要点】
1. 拿不准就给较低的 intent_confidence（低于 0.5 时系统不会调用任何工具）
2. keyword_list 最多 8 个名词，用于后续工具召回
3. exclude_tool_list 只在明确要排除某个工具时才填，否则给空数组
4. 能直接回答的常识问题归 chat，不要因为提到了某个名词就归 rag"""

# 【什么】用户提示词模板：只占位 {text} 一处
# 【作用】把待分类文本单独放用户侧，分类定义留在 system 侧，减少每轮重复 token
_INTENT_PROMPT_TMPL = "请判断下面这条用户输入的任务大类：\n\n{text}"

# LLM 返回的字段长度上限：防止模型把整段原文塞进 keyword_list
_MAX_KEYWORDS = 8        # 关键词最多保留 8 个
_MAX_WORD_LEN = 24       # 单个关键词最多 24 字符，超了直接截断
_MAX_DOMAIN_LEN = 32     # domain_tag 最多 32 字符

# ----------------------------------------------------------------------
# 【规则兜底表】LLM 不可用时的降级判定（关键词 → 意图 + 业务域）
#   confidence 上限 0.4：规则判定一定低于 0.5 阈值，
#   保证降级时不会误触发工具调用（安全默认值）。
# ----------------------------------------------------------------------
_RULE_FALLBACK: List[tuple] = [
    ("continue_memory", "general", r"^(继续|接着|然后呢|再说|详细说|上面|刚才|之前那个|上一个)", 0.40),
    ("rag", "general", r"(制度|规定|流程|文档|知识库|政策|标准|手册|指南|办法)", 0.40),
    ("local_skill", "general", r"(几点|时间|现在.*时候|天气|计算|算一下|等于|汇率)", 0.35),
    ("mcp_remote", "finance", r"(报销|发票|预算|成本|审计|付款)", 0.35),
    ("mcp_remote", "hr", r"(考勤|请假|年假|招聘|绩效|组织|花名册)", 0.35),
    ("mcp_remote", "it", r"(工单|运维|服务器|权限|VPN|资产|账号)", 0.35),
    ("local_skill", "general", r"(订单|库存|数据|统计|报表|查询)", 0.35),
]
# 【什么】预编译后的兜底规则表
# 【作用】导入时编译一次正则，降级路径每条请求都要过一遍，避免重复解析
_COMPILED_FALLBACK = [(t, d, re.compile(p), c) for t, d, p, c in _RULE_FALLBACK]

# 兜底用的关键词抽取：中文词 2-6 字 + 英文单词 2 字以上
_WORD_RE = re.compile(r"[\u4e00-\u9fa5]{2,6}|[A-Za-z]{2,}")


# 【什么】按正则从文本里抽关键词（去重、限数量）
# 【作用】给降级路径和 keyword_list 兜底，保证召回总能拿到可用的关键词
def _extract_keywords(text: str, limit: int = _MAX_KEYWORDS) -> List[str]:
    """从文本里抽关键词（兜底路径用，也用于给 LLM 的 keyword_list 兜底）。

    【输入输出对照】
        输入 "帮我查一下上海明天的天气怎么样"
            → 输出 ["帮我", "查一下", "上海", "明天", "天气", "怎么样"]
    """
    words: List[str] = []
    for w in _WORD_RE.findall(text or ""):
        if w not in words:
            words.append(w)
        if len(words) >= limit:
            break
    return words


# 【什么】把 LLM 各种形态的"是/否"统一成 bool
# 【作用】模型有时给 true、有时给"是"、有时给 1，不统一会在下游判断上出错
def _to_bool(value: Any) -> bool:
    """把 LLM 可能给出的各种形态统一成 bool。

    【输入输出对照】
        True → True；"true" → True；"是" → True；1 → True
        "false"/"" /None/0 → False
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in ("true", "1", "yes", "是", "需要")


# 【什么】规整字符串列表：只留字符串、去空白、去重、限长、限数量
# 【作用】挡住模型把整段原文或"无"塞进数组这类输出，保证下游遍历有意义
def _clean_list(value: Any, limit: int = _MAX_KEYWORDS) -> List[str]:
    """规整字符串列表：只留字符串、去空白、去重、限长、限数量。

    【为什么必须清洗】
        LLM 偶尔返回 "无"（字符串而非数组）或把整段原文塞进数组。
        不清洗的话，下游 for 循环会把字符串当字符序列逐字遍历，
        得到 ['无'] 这种看似正常实则无意义的结果。

    【输入输出对照】
        输入 ["报销", "报销", " 发票 "]  → 输出 ["报销", "发票"]
        输入 "无"                        → 输出 []
        输入 ["A"*50]                    → 输出 [前 24 个字符]
    """
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: List[str] = []
    for item in value:
        s = str(item).strip()
        if not s or s in ("无", "none", "null", "[]"):
            continue
        if len(s) > _MAX_WORD_LEN:
            s = s[:_MAX_WORD_LEN]
        if s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return out


# 【什么】规整业务域标签：只留字母数字下划线、小写、限长
# 【作用】domain_tag 会拼进召回过滤条件，必须收敛成可控字符集
def _normalize_domain(value: Any) -> str:
    """规整业务域标签：小写、只留字母数字下划线、超长截断、空值兜底 general。"""
    s = re.sub(r"[^A-Za-z0-9_]", "", str(value or "")).lower()
    if not s:
        return "general"
    return s[:_MAX_DOMAIN_LEN]


# 【什么】把 LLM 原始 JSON 归一化成 IntentResult
# 【作用】字段逐个校验收敛，出参的六个字段对下游而言全部可信
def _normalize(raw: Dict[str, Any]) -> IntentResult:
    """把 LLM 原始 JSON 归一化成 IntentResult（字段全部可信）。"""
    itype = str(raw.get("intent_type", "")).strip().lower()
    if itype not in IntentType.ALL:
        # --------------------------------------------------------------
        # 【非法类别怎么处理】不抛异常，直接归为 invalid。
        #   模型偶尔会自创类别（如 "tool_call"、"search"），
        #   归 invalid 的效果是：不召回工具、走拒答/澄清分支 —— 比猜一个类别安全。
        # --------------------------------------------------------------
        if itype:
            _logger.warning(f"LLM 返回未知意图类别: {itype} → 归为 invalid")
        itype = IntentType.INVALID

    try:
        conf = float(raw.get("intent_confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))   # clamp 到 [0,1]

    return IntentResult(
        intent_type=itype,
        intent_confidence=conf,
        need_memory=_to_bool(raw.get("need_memory", False)),
        domain_tag=_normalize_domain(raw.get("domain_tag", "general")),
        keyword_list=_clean_list(raw.get("keyword_list")),
        exclude_tool_list=_clean_list(raw.get("exclude_tool_list")),
        raw=raw if isinstance(raw, dict) else {},
        degraded=False,
    )


# 【什么】规则兜底：按关键词表猜一个意图
# 【作用】LLM 不可用时也能给出可用结果，且置信度封顶 0.4 保证不误触发工具
def _rule_fallback(text: str, reason: str) -> IntentResult:
    """规则兜底：LLM 不可用时的降级判定。

    【confidence 为什么封顶 0.4】
        规则匹配是"看着像"，不是"判得准"。封在 0.4 以下，
        下游 0.5 阈值会一律禁止调工具 —— 降级时只损失能力，不产生误调用。
    """
    if not text:
        return IntentResult(intent_type=IntentType.INVALID, intent_confidence=0.9,
                            degraded=True, degrade_reason=reason or "输入为空")

    for itype, domain, pattern, conf in _COMPILED_FALLBACK:
        if pattern.search(text):
            return IntentResult(
                intent_type=itype, intent_confidence=conf, need_memory=False,
                domain_tag=domain, keyword_list=_extract_keywords(text),
                degraded=True, degrade_reason=reason,
            )

    return IntentResult(
        intent_type=IntentType.CHAT, intent_confidence=0.30, need_memory=False,
        domain_tag="general", keyword_list=_extract_keywords(text),
        degraded=True, degrade_reason=reason,
    )


# 【什么】意图识别总入口：LLM 结构化输出 + 规则降级
# 【作用】给出 6 选 1 大类与辅助字段；任何异常都不上抛，一律转规则兜底
def recognize_intent(text: str, llm=None, use_llm: bool = True) -> IntentResult:
    """意图识别（LLM 结构化输出 + 规则降级）。

    Args:
        text: Token 控制后的文本
        llm: LLMClient 实例；为 None 时直接走规则兜底
        use_llm: False 时强制走规则兜底（测试/离线场景用）

    Returns:
        IntentResult：degraded=True 表示这次结果来自规则而非 LLM

    【输入输出对照】
        输入 "我的报销单到哪一步了" + LLM 可用
            → IntentResult(intent_type="mcp_remote", confidence=0.86,
                           domain_tag="finance", keyword_list=["报销单","审批"],
                           degraded=False)
        输入 同上 + LLM 不可用
            → IntentResult(intent_type="mcp_remote", confidence=0.35,
                           domain_tag="finance", degraded=True,
                           degrade_reason="LLM 未配置或调用失败")
    """
    if not use_llm:
        return _rule_fallback(text, "已关闭 LLM 意图识别（use_llm=False）")

    try:
        if llm is None:
            from modules.agent.llm import LLMClient
            from config import get_settings
            if not get_settings().deepseek_api_key:
                return _rule_fallback(text, "未配置 DEEPSEEK_API_KEY，走规则兜底")
            llm = LLMClient()

        # text[:2000]：分类不需要全文，超长部分砍掉省 token，也避免被细节带偏
        prompt = _INTENT_PROMPT_TMPL.format(text=text[:2000])
        raw = llm.invoke_structured(prompt, system_prompt=_INTENT_SYSTEM, temperature=0.1)

        # ------------------------------------------------------------------
        # 【invoke_structured 的失败形态】解析不出来时它返回 {"error":..., "raw":...}
        #   而不是抛异常 —— 所以必须显式检查 error 字段，不能只看"返回了 dict"。
        # ------------------------------------------------------------------
        if not isinstance(raw, dict) or "error" in raw:
            _logger.warning(f"意图识别解析失败，走规则兜底: {str(raw)[:200]}")
            return _rule_fallback(text, "LLM 输出无法解析为 JSON")

        result = _normalize(raw)
        _logger.info(
            f"意图识别: {result.intent_type} conf={result.intent_confidence:.2f} "
            f"domain={result.domain_tag} keywords={result.keyword_list}"
        )
        return result
    except Exception as e:
        _logger.warning(f"意图识别异常，走规则兜底: {e}")
        return _rule_fallback(text, f"LLM 调用异常: {e}")
