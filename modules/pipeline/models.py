"""
流水线数据结构
==============
请求处理流水线各环节的输入输出对象，全部带 to_dict()，方便直接当 API 响应返回。

【为什么单独放一个文件】
    流水线有 7 个环节，每个环节都要把结果交给下一环。
    如果各环节自己返回 dict，字段名写错时不会报错、只会静默拿到 None；
    统一成 dataclass 后，字段少传/拼错会在构造时立刻暴露。

【数据流向】
    CleanResult → RiskResult → TokenResult → IntentResult → RecallResult → RouteDecision
    全部装进 PipelineResult 返回给调用方。
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


class IntentType:
    """意图大类（LLM 只能从这 6 个里选一个）。

    【为什么只有 6 个】
        这一层只做「任务大类分类」，不猜具体用哪个工具 —— 具体工具交给 Tool-RAG 召回。
        分类数越少，LLM 越不容易摇摆，输出也越好校验。
    """

    INVALID = "invalid"            # 无效/乱码/纯噪音输入
    CHAT = "chat"                  # 普通闲聊、常识问答，不需要任何外部能力
    LOCAL_SKILL = "local_skill"    # 需要调用本地函数工具（计算器、数据查询…）
    RAG = "rag"                    # 需要检索知识库（制度、文档、标准答案）
    MCP_REMOTE = "mcp_remote"      # 需要调用 MCP 远程技能
    CONTINUE_MEMORY = "continue_memory"  # 承接上一轮对话（"继续"、"刚才那个"）

    ALL = (INVALID, CHAT, LOCAL_SKILL, RAG, MCP_REMOTE, CONTINUE_MEMORY)

    # ------------------------------------------------------------------
    # 【只有这两类才值得跑 Tool-RAG】rag 走向量库、continue_memory 走记忆库、
    #   chat 直接答 —— 它们都不需要工具候选，跑召回纯属浪费一次 embedding。
    # ------------------------------------------------------------------
    TOOL_INTENTS = (LOCAL_SKILL, MCP_REMOTE)


class Branch:
    """分支路由的最终落点（与 IntentType 一一对应，invalid 落到 reject）。"""

    REJECT = "reject"
    CHAT = "chat"
    LOCAL_SKILL = "local_skill"
    RAG = "rag"
    MCP_REMOTE = "mcp_remote"
    CONTINUE_MEMORY = "continue_memory"


@dataclass
class CleanResult:
    """文本清洗结果。"""

    text: str = ""                       # 清洗后的文本（后续环节都用这份）
    original_length: int = 0
    cleaned_length: int = 0
    changed: bool = False                # 是否被改动过（用于判断是否记日志）
    removed: List[str] = field(default_factory=list)  # 被清掉的东西，如 ["控制字符x3","零宽字符x2"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskResult:
    """前置高危规则风控结果。

    【三个字段怎么配合】
        level   = 严重程度（none / low / medium / high）
        action  = 处置动作（pass 放行 / review 挂人工复核 / block 直接拦截）
        passed  = action != "block"，调用方只看这一个字段决定要不要继续
    """

    passed: bool = True
    level: str = "none"
    action: str = "pass"
    hits: List[Dict[str, str]] = field(default_factory=list)  # [{"code","level","desc"}]
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenResult:
    """Token 长度控制与压缩结果。"""

    text: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    compressed: bool = False
    method: str = "none"   # none=未超阈值 / rule=规则压缩 / llm=LLM压缩 / truncate=硬截断

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntentResult:
    """意图识别结果（LLM 结构化输出的落地形态）。

    【字段与 LLM 输出严格一一对应】
        intent_type        6 选 1
        intent_confidence  0~1，低于 pipeline_min_confidence（默认 0.5）禁止调工具
        need_memory        本轮是否需要取历史记忆
        domain_tag         业务域标签，Tool-RAG 按它过滤（finance / hr / it / general）
        keyword_list       关键词，用于向量召回失败时的关键词降级
        exclude_tool_list  黑名单：这些工具本轮不许被召回
    """

    intent_type: str = IntentType.CHAT
    intent_confidence: float = 0.0
    need_memory: bool = False
    domain_tag: str = "general"
    keyword_list: List[str] = field(default_factory=list)
    exclude_tool_list: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)   # LLM 原始输出，排障用
    degraded: bool = False        # True = LLM 没跑成，走了规则兜底
    degrade_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolHit:
    """Tool-RAG 召回到的一个候选技能。"""

    name: str
    description: str = ""
    score: float = 0.0            # 相似度（余弦），0~1
    domain_tag: str = "general"
    source_type: str = "local"    # local=本地工具 / mcp_remote=远程技能
    server_name: str = ""         # MCP 场景下属于哪个 server（按需加载用）
    source: str = "pgvector"      # pgvector / keyword

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecallResult:
    """Tool-RAG 召回结果。"""

    executed: bool = False                 # 是否真的跑了召回（非工具意图时为 False）
    tools: List[ToolHit] = field(default_factory=list)
    source: str = "none"                   # pgvector / keyword / none
    reason: str = ""
    degraded: bool = False
    degrade_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executed": self.executed,
            "source": self.source,
            "reason": self.reason,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "tools": [t.to_dict() for t in self.tools],
        }


@dataclass
class RouteDecision:
    """分支路由决策。"""

    branch: str = Branch.CHAT
    tool_calls_allowed: bool = False   # False = 本轮禁止调用任何工具
    tools: List[ToolHit] = field(default_factory=list)
    reason: str = ""
    min_confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch": self.branch,
            "tool_calls_allowed": self.tool_calls_allowed,
            "reason": self.reason,
            "min_confidence": self.min_confidence,
            "tools": [t.to_dict() for t in self.tools],
        }


@dataclass
class PipelineResult:
    """流水线最终输出：每一环的结果都在里面，排障时一眼看到是哪一步做的决定。"""

    query: str = ""
    clean: CleanResult = field(default_factory=CleanResult)
    risk: RiskResult = field(default_factory=RiskResult)
    tokens: TokenResult = field(default_factory=TokenResult)
    intent: IntentResult = field(default_factory=IntentResult)
    recall: RecallResult = field(default_factory=RecallResult)
    route: RouteDecision = field(default_factory=RouteDecision)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    stopped_at: str = ""       # 非空 = 流水线在这一步提前结束（如风控拦截）
    degraded: bool = False     # 链路中任意环节降级过

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "clean": self.clean.to_dict(),
            "risk": self.risk.to_dict(),
            "tokens": self.tokens.to_dict(),
            "intent": self.intent.to_dict(),
            "recall": self.recall.to_dict(),
            "route": self.route.to_dict(),
            "trace": self.trace,
            "stopped_at": self.stopped_at,
            "degraded": self.degraded,
        }
