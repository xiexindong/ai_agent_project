"""
提示词组装器模块
================
把「角色 + 策略 + 上下文 + 输出格式」四层拼成最终 messages，并在拼装时做 token 预算裁剪。

【四层结构，为什么分四层】
    角色层     role          这轮对话模型"是谁"（企业助手 / 财务专员 / 审核员）
    策略层     strategy      这轮"怎么问"（zero_shot / few_shot / cot / react）
    上下文层   **context     这轮"拿什么答"（知识库、记忆、权限、工具清单）
    输出层     output_format 这轮"答成什么样"（纯文本 / JSON / Markdown）

    四层互不耦合：换策略不用动模板，换模板不用动裁剪逻辑。
    而现在 react_agent.py:305 是把模板、策略、上下文全写在一个 format() 里，
    想单独调"策略"这一层，只能在那一大段字符串里改。

【token 预算裁剪的顺序】
    上下文不是平均重要。预算不够时按优先级从低往高丢：
        权限边界 > 工具清单 > 观察结果 > 技能SOP > 知识库 > 历史记忆 > 工具历史
    权限边界排最前是因为：它是硬性约束，丢了会越权；
    历史记忆排后面是因为：它只是"可能有帮助"，丢了最多答得没那么贴人。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from modules.common.logger import get_logger
from modules.common.token_counter import count_tokens
from modules.common.validator import repair_json
from modules.prompt.builtin import get_registry
from modules.prompt.strategies import FewShotStore, build_strategy_block
from modules.prompt.templates import PromptRegistry

_logger = get_logger("prompt.builder")

# 上下文块优先级，从高到低。不在表里的键排在最后（按名字排序，保证顺序稳定）。
_CONTEXT_PRIORITY = (
    "permission_context",  # 权限边界：硬性约束，丢了会越权，优先级最高
    "tools_desc",          # 工具清单：没有它模型无从调用工具
    "observation",         # 工具观察结果：本轮刚拿到的新事实
    "skill_context",       # 技能 SOP：已激活技能时的作业流程
    "rag_context",         # 知识库参考：回答的主要依据
    "memory_context",      # 历史记忆：让回答更贴人，丢了不致命
    "tool_history",        # 历史工具调用：只在排查重复调用时有用，优先级最低
)

# 上下文类变量：这类变量"有就给、没有就没有"，调用方不传时填占位，不报错。
# 与之相对的是业务变量（query / answer / summaries / labels / max_iter），
# 那些缺了整个提示词就没意义，必须由模板层的 render() 抛错拦住。
# 注意这里不含 strategy_block：策略是"指令"不是"上下文"，由 build_record 直接注入，
# 不参与上下文预算竞争，否则预算一满就会先把"怎么问"丢掉，只剩一堆资料。
_CONTEXT_KEYS = set(_CONTEXT_PRIORITY)   # 上下文类变量的集合，用于判断某个键是否参与预算裁剪
_EMPTY_PLACEHOLDER = "（未提供）"            # 上下文没传时的占位文本，保证模板始终能渲染出来


# 【什么】把四层内容拼成最终 messages 的组装器
# 【作用】统一负责渲染、策略注入、token 预算裁剪，调用方只管传上下文
@dataclass
class PromptBuilder:
    """提示词组装器。

    Attributes:
        template_name:    用的哪个模板（默认 qa.grounded）
        template_version: 指定版本，留空取最新
        strategy:         见 strategies.STRATEGIES
        role:             角色指令，拼在 system 开头
        output_format:    text / json / markdown
        output_schema:    json 模式下补充的字段说明
        max_context_tokens: 上下文 token 预算（不含 query 与指令本身）
        few_shot_store:   few_shot 策略的示例库
        few_shot_k:       few_shot 取几条示例
        max_iter:         react 策略的循环上限
        registry:         模板注册表，留空用全局单例
    """

    template_name: str = "qa.grounded"             # 用的模板名，默认知识库问答模板
    template_version: str = ""                     # 指定版本；留空取最后注册的那版
    strategy: str = "zero_shot"                    # 提问策略，见 strategies.STRATEGIES
    role: str = ""                                 # 角色指令，拼在 system 开头
    output_format: str = "text"                    # 输出格式：text / json / markdown
    output_schema: str = ""                        # json 模式补充的字段说明
    max_context_tokens: int = 3000                 # 上下文 token 预算，不含 query 与指令本身
    few_shot_store: Optional[FewShotStore] = None  # few_shot 示例库；为空则退化成 zero_shot
    few_shot_k: int = 2                            # few_shot 取几条示例
    max_iter: int = 10                             # react 策略的循环上限，写进策略块
    registry: Optional[PromptRegistry] = None      # 模板注册表；留空用全局单例

    # 残羹冷炙阈值：裁剪后剩下的 token 少于这个值就不塞了，
    # 半截上下文比没有更容易误导模型（看着像有依据，实际是被砍断的）。
    _MIN_KEEP_TOKENS: int = field(default=50, repr=False)

    # ------------------------------------------------------------------
    # 【build 的返回结构】
    #   返回 OpenAI 格式的 messages：system 一条 + user 一条。
    #   system 放「角色 + 策略 + 输出格式」（这三层与具体提问无关，可复用），
    #   user 放「渲染后的模板」（随每次提问变化）。
    #   分开的好处：多轮对话时 system 不变，服务端可命中前缀缓存（省 token 也省钱）。
    # ------------------------------------------------------------------
    # 【什么】对外主入口：返回可直接喂给模型的 messages
    # 【作用】只要 messages 时用这个；要看 token 与裁剪记录用 build_record
    def build(self, query: str, **context) -> List[Dict[str, str]]:
        rec = self.build_record(query, **context)
        return rec["messages"]

    # 【什么】完整构建：messages + token 数 + 裁剪记录
    # 【作用】排查慢查询、评测效果时用它，能看到哪块上下文被截断或丢弃
    def build_record(self, query: str, **context) -> dict:
        """完整构建，除 messages 外还返回 token 数与裁剪记录（排查与评测用）。

        Returns:
            {
              "messages": [{"role": "system", ...}, {"role": "user", ...}],
              "system": str, "user": str,
              "template": "qa.grounded(v1)", "strategy": "few_shot",
              "prompt_tokens": int,       # system + user 合计
              "context_tokens": int,      # 裁剪后上下文实际占用
              "trimmed": [str, ...],      # 裁剪动作，空列表表示没超预算
            }
        """
        registry = self.registry or get_registry()
        tpl = registry.get(self.template_name, self.template_version)
        needed = tpl.variables

        # ------------------------------------------------------------------
        # 【策略段到底进 user 还是进 system】取决于模板有没有 {strategy_block} 插位：
        #   有插位（qa.grounded）→ 策略段作为变量渲染进 user，system 里不再拼；
        #   没插位（react.reason）→ 策略段整段进 system，user 只放上下文。
        #   两条路只走一条，否则同一段指令会出现两次，白烧 token。
        #
        # 【工具清单同理】strategy=react 的策略块自带「## 可用工具」，
        #   而 react.reason 模板也有 {tools_desc} 插位，两处都填就是同一份清单两遍。
        #   判断依据同样是 needed：模板自己有插位，策略块就不带清单。
        # ------------------------------------------------------------------
        tools_desc = str(context.get("tools_desc", "") or "")
        ctx, trimmed = self._fit_context(needed, context)
        if "strategy_block" in needed:
            ctx["strategy_block"] = build_strategy_block(
                self.strategy,
                query=query,
                store=self.few_shot_store,
                k=self.few_shot_k,
                tools_desc="",
                max_iter=self.max_iter,
            )
        if "query" in needed:
            ctx["query"] = query
        user = tpl.render(**ctx)

        system = self.system_prompt(
            query=query,
            tools_desc="" if "tools_desc" in needed else tools_desc,
            include_strategy="strategy_block" not in needed,
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        prompt_tokens = count_tokens(system) + count_tokens(user)
        _logger.debug(
            f"提示词组装: {tpl.name}({tpl.version}) strategy={self.strategy} "
            f"tokens={prompt_tokens}"
        )
        return {
            "messages": messages,
            "system": system,
            "user": user,
            "template": f"{tpl.name}({tpl.version})",
            "strategy": self.strategy,
            "prompt_tokens": prompt_tokens,
            "context_tokens": sum(count_tokens(v) for v in ctx.values() if isinstance(v, str)),
            "trimmed": trimmed,
        }

    # ------------------------------------------------------------------
    # 【只生成 system 段】给「system 固定、user 每轮变」的场景用。
    #   react_agent 的 reason_step 正是这种：角色与流程每轮都一样，变的只是
    #   query 与上下文，所以 system 在构造时算一次缓存起来，每轮只重拼 user
    #   —— 省一次渲染，也让服务端能命中前缀缓存（prefix caching）。
    #
    # 【include_strategy=False 的时机】
    #   模板自带 {strategy_block} 插位时（qa.grounded），策略段留在 user，
    #   system 就不重复拼，见 build_record 里的分支。
    #
    # 【输入输出对照（strategy=react, max_iter=10）】
    #   输入  role="你是企业级智能AI助手..."  tools_desc=""
    #   输出  "你是企业级智能AI助手...\n\n## 工作流程（ReAct）\n1. Reason...\n\n## 可用工具\n（无可用工具）"
    # ------------------------------------------------------------------
    # 【什么】只生成 system 段（角色 + 策略 + 输出格式）
    # 【作用】system 固定、user 每轮变的场景可缓存它，还能命中前缀缓存
    def system_prompt(
        self,
        role: str = "",
        query: str = "",
        tools_desc: str = "",
        include_strategy: bool = True,
    ) -> str:
        parts = [role or self.role]
        if include_strategy:
            parts.append(
                build_strategy_block(
                    self.strategy,
                    query=query,
                    store=self.few_shot_store,
                    k=self.few_shot_k,
                    tools_desc=tools_desc,
                    max_iter=self.max_iter,
                )
            )
        fmt = self._format_instruction()
        if fmt:
            parts.append(fmt)
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # 【贪心装包：按优先级顺序一块一块往预算里塞】
    #   塞得下 → 整块保留；塞不下 → 截断到剩余预算；剩余预算太少 → 整块丢弃。
    #
    # 【输入输出对照（max_context_tokens=500）】
    #   输入  rag_context=1200 token 的长文档，memory_context=80 token
    #   输出  rag_context 截断到 500 token，memory_context 丢弃
    #         trimmed = ["rag_context: 截断 1200 → 486 token",
    #                    "memory_context: 丢弃 80 token（预算已满）"]
    #
    # 【为什么不截断 query】问题是本次请求的全部意义，砍它等于换了个问题。
    # ------------------------------------------------------------------
    # 【什么】按预算装上下文：能塞就塞，塞不下就截断或丢弃
    # 【作用】保证总 token 不超 max_context_tokens（默认 3000），并记录裁剪动作
    def _fit_context(self, needed, context: dict) -> tuple[dict, List[str]]:
        ordered = [k for k in _CONTEXT_PRIORITY if k in needed and k in _CONTEXT_KEYS]
        ordered += sorted(
            k for k in needed if k in _CONTEXT_KEYS and k not in _CONTEXT_PRIORITY
        )
        # 上下文变量没传 → 填占位，保证模板能渲染出来；
        # 业务变量（query/answer/...）不在这里，缺了由 tpl.render() 抛错。
        filled = {k: str(context.get(k) or _EMPTY_PLACEHOLDER) for k in ordered}

        budget = self.max_context_tokens
        used = 0
        kept: dict = {}
        trimmed: List[str] = []
        for key in ordered:
            text = filled[key]
            if not text:
                continue
            cost = count_tokens(text)
            if used + cost <= budget:
                kept[key] = text
                used += cost
                continue
            remain = budget - used
            if remain >= self._MIN_KEEP_TOKENS:
                clipped = self._clip(text, remain)
                kept[key] = clipped
                trimmed.append(f"{key}: 截断 {cost} → {count_tokens(clipped)} token")
                used = budget
            else:
                # 【丢弃也要留占位符】整块不塞进 prompt 时，模板里的 {xxx} 仍然要填。
                #   不填 → render() 以"缺少变量"抛错，一次裁剪把整轮对话搞挂；
                #   填"（未提供）" → 模型明确知道这一块没有，比留空更不容易瞎编。
                kept[key] = _EMPTY_PLACEHOLDER
                trimmed.append(f"{key}: 丢弃 {cost} token（预算已满）")
        return kept, trimmed

    # ------------------------------------------------------------------
    # 【按 token 反推该留多少字符】
    #   tiktoken 不能"只编前 N 个 token"再倒推文本（编码不可逆到字符边界），
    #   所以按「token : 字符」的比例估算，再留 12 个字符给省略标记。
    #   截断只是保底手段，真正该做的是上游就别塞这么长的上下文进来。
    # ------------------------------------------------------------------
    # 【什么】把文本按 token 上限截短
    # 【作用】超出预算时保留前段并追加「…（已截断）」标记，给模型明确提示
    @staticmethod
    def _clip(text: str, limit_tokens: int) -> str:
        total = count_tokens(text)
        if total <= limit_tokens:
            return text
        keep = max(0, int(len(text) * (limit_tokens / total)) - 12)
        return text[:keep] + "\n…（已截断）"

    # 【什么】生成输出格式约束段
    # 【作用】json / markdown 才产出指令，text 返回空串即不附加约束
    def _format_instruction(self) -> str:
        """输出格式约束。放 system 而不是 user：
        模型对 system 里的格式要求服从度更高，且多轮时不会随历史被淹没。"""
        if self.output_format == "json":
            schema = f"\n字段: {self.output_schema}" if self.output_schema else ""
            return (
                "## 输出格式\n只输出一个 JSON 对象，不要用 ``` 包裹，"
                "不要写任何解释文字。" + schema
            )
        if self.output_format == "markdown":
            return "## 输出格式\n用 Markdown：一级标题写结论，下面用要点列出依据。"
        return ""


# ----------------------------------------------------------------------
# 【结构化输出的解析：与 builder 配套的收尾】
#   builder 负责"让模型输出 JSON"，这里负责"把输出变成 dict"。
#   复用 validator.repair_json：它能修尾逗号、缺右括号等常见毛病
#   （repair 失败返回 None，调用方按 None 处理，不要假装解析成功）。
#
# 【输入输出对照】
#   输入 '{"intent": "报销咨询", "confidence": 0.9,}'   ← 尾逗号，json.loads 会失败
#   输出 {"intent": "报销咨询", "confidence": 0.9}
#   输入 '我觉得这是报销咨询'                            ← 完全不是 JSON
#   输出 None
# ----------------------------------------------------------------------
# 【什么】把模型输出解析成 dict
# 【作用】配合 output_format=json 收尾；解析不了返回 None 由调用方处理
def parse_json_output(text: str) -> Optional[dict]:
    if not text:
        return None
    return repair_json(text.strip())
