"""
Token 长度控制与超长压缩
========================
流水线第 3 步：把输入压到「意图识别吃得下」的长度。

【为什么卡在意图识别之前】
    意图识别只需要判断大类，不需要全文细节。
    把 3 万字原文塞给分类模型，既费钱又容易被细节带偏（长文本里混着别的诉求，
    模型容易选错类）。先压到预算内，分类反而更稳。

【三级压缩，从便宜到贵】
    ① 规则压缩（免费、零延迟）：去重复行、去寒暄套话、去空行
    ② LLM 压缩（要花一次调用）：保留实体与诉求的摘要式压缩
    ③ 硬截断（最后兜底）：按比例砍字符，一定会丢信息但绝不会失败

【降级链】
    规则压缩够用就不调 LLM；LLM 失败/没配 key 就直接硬截断 ——
    压缩环节失败不该让整条流水线挂掉。
"""
from typing import Optional

from modules.common.logger import get_logger
from modules.common.token_counter import count_tokens
from modules.pipeline.models import TokenResult

_logger = get_logger("pipeline.token")

# ----------------------------------------------------------------------
# 【寒暄套话】压缩时优先去掉的部分：它们对判断意图没有贡献
#   只去开头的（用户铺垫用），不去中间的（可能是真实诉求的一部分）
# ----------------------------------------------------------------------
_SMALL_TALK_PREFIX = ("你好", "您好", "hi", "hello", "请问", "麻烦问一下", "打扰一下", "谢谢", "感谢")  # 9 个开头寒暄词

# LLM 压缩提示词：只要求"保留什么"，不要求"怎么说"，避免模型自由发挥
_COMPRESS_SYSTEM = (
    "你是文本压缩助手。把用户原文压缩到指定长度以内，"
    "必须保留：时间、金额、人名/部门名、核心诉求、关键约束条件。"
    "丢弃：寒暄、重复描述、与诉求无关的背景故事。"
    "只输出压缩后的文本，不要解释、不要加前缀。"
)


# 【什么】第 ① 级压缩：去重复行 + 去开头寒暄 + 压空白
# 【作用】零成本零延迟地砍掉冗余，多数超长输入在这一级就能回到预算内
def _rule_compress(text: str) -> str:
    """规则压缩：去重行 + 去寒暄 + 压空白。

    【输入输出对照】
        输入 "你好\n请问我的报销单\n我的报销单到哪一步了\n\n\n"
            → 输出 "我的报销单\n我的报销单到哪一步了"
            （开头"你好/请问"已去，连续空行已压）

    【为什么保留重复行里的第一次出现】
        用户常先说结论再复述一遍细节，去重时保留首次出现 = 保留第一次的描述，
        语义顺序不变，不会把"原因"挪到"结论"前面。
    """
    lines = []
    seen = set()
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line in seen:          # 完全重复的行只留第一次
            continue
        seen.add(line)
        lines.append(line)

    out = "\n".join(lines)
    low = out.lower()
    for word in _SMALL_TALK_PREFIX:
        if low.startswith(word) and len(out) > len(word) + 4:
            out = out[len(word):].lstrip("，,、：: ")
            low = out.lower()
            break
    return out


# 【什么】第 ② 级压缩：调一次模型做摘要式压缩
# 【作用】规则压不动时靠模型保留实体与诉求；失败返回 None 交给第 ③ 级
def _llm_compress(text: str, max_tokens: int, llm) -> Optional[str]:
    """LLM 压缩：调用一次模型做摘要式压缩。

    Returns:
        压缩后的文本；失败返回 None（调用方接着走硬截断）
    """
    if llm is None:
        return None
    try:
        prompt = f"请把下面文本压缩到 {max_tokens} token 以内：\n\n{text}"
        # temperature=0.1 压低发散；输出上限给预算的 2 倍，避免模型自己把结果写一半截断
        out = llm.invoke(prompt, system_prompt=_COMPRESS_SYSTEM, temperature=0.1, max_tokens=max_tokens * 2)
        out = (out or "").strip()
        return out or None
    except Exception as e:
        _logger.warning(f"LLM压缩失败，转硬截断: {e}")
        return None


# 【什么】第 ③ 级压缩：按比例硬截断
# 【作用】最后兜底，一定成功；会丢尾部信息但保证 token 一定达标
def _truncate(text: str, max_tokens: int, tokens_now: int) -> str:
    """硬截断：按 token 超出的比例反推要保留多少字符。

    【为什么要乘 0.95】
        留 5% 余量。字符与 token 不是线性关系（中文 1 字≈2 token，英文 1 词≈1.3 token），
        按比例算出来的字符数切完可能仍略超预算；留余量能保证截断后一定达标。

    【输入输出对照】
        text="A"*10000, tokens_now=5000, max_tokens=2000
            → 保留 10000 * 2000/5000 * 0.95 = 3800 个字符
    """
    if tokens_now <= 0:
        return text[:max_tokens]
    keep = int(len(text) * max_tokens / tokens_now * 0.95)
    return text[:max(1, keep)]


# 【什么】Token 长度控制总入口，按 ① 规则 → ② LLM → ③ 硬截断逐级降级
# 【作用】把输入压到意图识别吃得下的长度，且压缩环节失败也不让流水线挂掉
def control_tokens(
    text: str,                       # 清洗 + 风控后的文本
    max_tokens: int = 2000,          # token 预算，默认取 settings.pipeline_max_input_tokens
    llm=None,                        # LLMClient；为 None 时跳过第 ② 级
    compress_enabled: bool = True,   # False 时压缩只走硬截断
) -> TokenResult:
    """Token 长度控制与超长压缩。

    Args:
        text: 清洗 + 风控后的文本
        max_tokens: token 预算（默认取 settings.pipeline_max_input_tokens）
        llm: LLMClient 实例，规则压缩不够时用它做摘要式压缩；为 None 则跳过第 ② 级
        compress_enabled: False 时压缩只走硬截断（配置开关）

    Returns:
        TokenResult.text 是最终文本，后续环节一律用它

    【输入输出对照】
        text="3*7 等于多少", max_tokens=2000
            → TokenResult(text="3*7 等于多少", tokens_before=6, tokens_after=6,
                          compressed=False, method="none")
        text=5000 token 的重复长文, max_tokens=2000
            → 先规则压缩，若降到 1800 token → method="rule", compressed=True
            → 若规则后仍 3000 token → 调 LLM → method="llm"
            → LLM 失败 → method="truncate", 截断到 <=2000 token
    """
    tokens_before = count_tokens(text)
    if tokens_before <= max_tokens:
        return TokenResult(text=text, tokens_before=tokens_before,
                           tokens_after=tokens_before, compressed=False, method="none")

    if not compress_enabled:
        cut = _truncate(text, max_tokens, tokens_before)
        return TokenResult(text=cut, tokens_before=tokens_before,
                           tokens_after=count_tokens(cut), compressed=True, method="truncate")

    # ① 规则压缩
    ruled = _rule_compress(text)
    tokens_ruled = count_tokens(ruled)
    if tokens_ruled <= max_tokens:
        _logger.info(f"规则压缩: {tokens_before} -> {tokens_ruled} token")
        return TokenResult(text=ruled, tokens_before=tokens_before,
                           tokens_after=tokens_ruled, compressed=True, method="rule")

    # ② LLM 压缩
    llm_out = _llm_compress(ruled, max_tokens, llm)
    if llm_out:
        tokens_llm = count_tokens(llm_out)
        if tokens_llm <= max_tokens:
            _logger.info(f"LLM压缩: {tokens_before} -> {tokens_llm} token")
            return TokenResult(text=llm_out, tokens_before=tokens_before,
                               tokens_after=tokens_llm, compressed=True, method="llm")

    # ③ 硬截断（兜底：一定成功）
    cut = _truncate(ruled, max_tokens, tokens_ruled)
    tokens_cut = count_tokens(cut)
    _logger.warning(f"硬截断: {tokens_before} -> {tokens_cut} token（规则压缩后仍 {tokens_ruled}）")
    return TokenResult(text=cut, tokens_before=tokens_before,
                       tokens_after=tokens_cut, compressed=True, method="truncate")
