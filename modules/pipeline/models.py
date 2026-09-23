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


# 【什么】意图大类常量集：6 个类别值 + 合法全集 + 需召回子集
# 【作用】给 LLM 分类圈定取值范围，下游校验与"要不要召回"都看它
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

    ALL = (INVALID, CHAT, LOCAL_SKILL, RAG, MCP_REMOTE, CONTINUE_MEMORY)  # 合法类别全集，归一化时按它校验

    # ------------------------------------------------------------------
    # 【只有这两类才值得跑 Tool-RAG】rag 走向量库、continue_memory 走记忆库、
    #   chat 直接答 —— 它们都不需要工具候选，跑召回纯属浪费一次 embedding。
    # ------------------------------------------------------------------
    TOOL_INTENTS = (LOCAL_SKILL, MCP_REMOTE)


# 【什么】分支落点常量集：6 条支路，与 IntentType 一一对应
# 【作用】把意图翻译成链路名，上层按 branch 分派到对应处理逻辑
class Branch:
    """分支路由的最终落点（与 IntentType 一一对应，invalid 落到 reject）。"""

    REJECT = "reject"                    # 拒答/澄清：输入无效或被风控拦截
    CHAT = "chat"                        # 普通对话：不带工具，直接答
    LOCAL_SKILL = "local_skill"          # 本地函数工具链路：带召回到的工具
    RAG = "rag"                          # 知识库检索链路：走 Milvus，不带工具
    MCP_REMOTE = "mcp_remote"            # MCP 远程技能链路：带召回到的工具
    CONTINUE_MEMORY = "continue_memory"  # 记忆续聊链路：先取历史记忆再回答


# 【什么】第 ① 环文本清洗的结果对象
# 【作用】把清洗后文本与长度变化交给下一环，removed 供日志溯源
@dataclass
class CleanResult:
    """文本清洗结果。"""

    text: str = ""                       # 清洗后的文本（后续环节都用这份）
    original_length: int = 0             # 清洗前字符数
    cleaned_length: int = 0              # 清洗后字符数
    changed: bool = False                # 是否被改动过（用于判断是否记日志）
    removed: List[str] = field(default_factory=list)  # 被清掉的东西，如 ["控制字符x3","零宽字符x2"]

    # 【什么】把 dataclass 原样转成普通 dict
    # 【作用】API 响应与日志直接吃这份结构，上层不必再逐字段拼装
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 【什么】第 ② 环前置风控的结果对象
# 【作用】给出放行与否与处置动作，pipeline 只看 passed 决定要不要继续
@dataclass
class RiskResult:
    """前置高危规则风控结果。

    【三个字段怎么配合】
        level   = 严重程度（none / low / medium / high）
        action  = 处置动作（pass 放行 / review 挂人工复核 / block 直接拦截）
        passed  = action != "block"，调用方只看这一个字段决定要不要继续
    """

    passed: bool = True       # 是否放行；调用方只看这一个字段
    level: str = "none"       # 严重程度：none/low/medium/high，取所有命中的最高级
    action: str = "pass"      # 处置动作：pass 放行 / review 挂复核 / block 拦截
    hits: List[Dict[str, str]] = field(default_factory=list)  # [{"code","level","desc"}]
    reason: str = ""          # 给人看的说明，正常放行时为空串

    # 【什么】把 dataclass 原样转成普通 dict
    # 【作用】API 响应与日志直接吃这份结构，上层不必再逐字段拼装
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 【什么】第 ③ 环 Token 控制与压缩的结果对象
# 【作用】给出最终文本，method 说明这次走了哪一级压缩，便于排障
@dataclass
class TokenResult:
    """Token 长度控制与压缩结果。"""

    text: str = ""            # 最终文本，后续环节一律用它
    tokens_before: int = 0    # 压缩前 token 数
    tokens_after: int = 0     # 压缩后 token 数
    compressed: bool = False  # 是否发生过压缩或截断
    method: str = "none"      # none=未超阈值 / rule=规则压缩 / llm=LLM压缩 / truncate=硬截断

    # 【什么】把 dataclass 原样转成普通 dict
    # 【作用】API 响应与日志直接吃这份结构，上层不必再逐字段拼装
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 【什么】第 ④ 环意图识别的结果对象
# 【作用】承载 LLM 的 6 字段结构化输出，供召回与路由两个环节取用
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

    intent_type: str = IntentType.CHAT  # 意图类别：6 选 1，见 IntentType
    intent_confidence: float = 0.0      # 置信度 0~1；低于 0.5 时下游禁止调工具
    need_memory: bool = False           # 本轮是否需要取历史记忆
    domain_tag: str = "general"         # 业务域标签，Tool-RAG 按它过滤候选
    keyword_list: List[str] = field(default_factory=list)  # 关键词，向量召回失败时降级用
    exclude_tool_list: List[str] = field(default_factory=list)  # 黑名单：这些工具本轮不许被召回
    raw: Dict[str, Any] = field(default_factory=dict)   # LLM 原始输出，排障用
    degraded: bool = False              # True = LLM 没跑成，走了规则兜底
    degrade_reason: str = ""            # 降级原因，degraded=True 时才有内容

    # 【什么】把 dataclass 原样转成普通 dict
    # 【作用】API 响应与日志直接吃这份结构，上层不必再逐字段拼装
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 【什么】Tool-RAG 召回到的单个候选技能
# 【作用】把"名字 + 描述 + 相似度 + 来源"打包，供路由挑选与按需加载
@dataclass
class ToolHit:
    """Tool-RAG 召回到的一个候选技能。"""

    name: str                    # 工具/技能名，表里的唯一键
    description: str = ""         # 工具描述，关键词降级时参与匹配
    score: float = 0.0            # 相似度（余弦），0~1
    domain_tag: str = "general"   # 业务域标签
    source_type: str = "local"    # local=本地工具 / mcp_remote=远程技能
    server_name: str = ""         # MCP 场景下属于哪个 server（按需加载用）
    source: str = "pgvector"      # pgvector / keyword

    # 【什么】把 dataclass 原样转成普通 dict
    # 【作用】API 响应与日志直接吃这份结构，上层不必再逐字段拼装
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 【什么】第 ⑤ 环 Tool-RAG 召回的结果对象
# 【作用】带回候选工具清单，并说明这次走的向量还是关键词降级
@dataclass
class RecallResult:
    """Tool-RAG 召回结果。"""

    executed: bool = False                 # 是否真的跑了召回（非工具意图时为 False）
    tools: List[ToolHit] = field(default_factory=list)  # 候选工具，按 score 降序
    source: str = "none"                   # pgvector / keyword / none
    reason: str = ""                       # 结果说明，如"向量召回为空，改用关键词召回"
    degraded: bool = False                 # True = 没走成向量召回，用了降级路径
    degrade_reason: str = ""               # 降级原因，如"pgvector 存储不可用"

    # 【什么】转成 dict，并把 tools 逐个转 dict
    # 【作用】嵌套的 ToolHit 不是 JSON 原生类型，asdict 之外还要手动展开一层
    def to_dict(self) -> Dict[str, Any]:
        return {
            "executed": self.executed,
            "source": self.source,
            "reason": self.reason,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "tools": [t.to_dict() for t in self.tools],
        }


# 【什么】第 ⑥ 环分支路由的决策对象
# 【作用】定支路 + 定工具白名单，是流水线交给上层的最终结论
@dataclass
class RouteDecision:
    """分支路由决策。"""

    branch: str = Branch.CHAT              # 落点支路，见 Branch
    tool_calls_allowed: bool = False       # False = 本轮禁止调用任何工具
    tools: List[ToolHit] = field(default_factory=list)  # 本轮可用工具白名单，禁工具时为空
    reason: str = ""                       # 决策说明，如"置信度 0.35 < 阈值 0.50"
    min_confidence: float = 0.5            # 本次生效的置信度下限（默认 0.5）

    # 【什么】转成 dict，并把 tools 逐个转 dict
    # 【作用】同上：嵌套 ToolHit 需手动展开，另外带上生效的 min_confidence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch": self.branch,
            "tool_calls_allowed": self.tool_calls_allowed,
            "reason": self.reason,
            "min_confidence": self.min_confidence,
            "tools": [t.to_dict() for t in self.tools],
        }


# 【什么】流水线最终输出：六环结果 + trace 全在里面
# 【作用】一次返回全链路证据，排障时能直接看出是哪一环做的决定
@dataclass
class PipelineResult:
    """流水线最终输出：每一环的结果都在里面，排障时一眼看到是哪一步做的决定。"""

    query: str = ""            # 用户原始输入（未清洗）
    clean: CleanResult = field(default_factory=CleanResult)  # ① 文本清洗结果
    risk: RiskResult = field(default_factory=RiskResult)  # ② 前置风控结果
    tokens: TokenResult = field(default_factory=TokenResult)  # ③ Token 控制结果
    intent: IntentResult = field(default_factory=IntentResult)  # ④ 意图识别结果
    recall: RecallResult = field(default_factory=RecallResult)  # ⑤ Tool-RAG 召回结果
    route: RouteDecision = field(default_factory=RouteDecision)  # ⑥ 分支路由决策
    trace: List[Dict[str, Any]] = field(default_factory=list)  # 每环执行记录：step/ok/ms/detail
    stopped_at: str = ""       # 非空 = 流水线在这一步提前结束（如风控拦截）
    degraded: bool = False     # 链路中任意环节降级过

    # 【什么】把全链路结果转成一个可序列化的 dict
    # 【作用】API 直接返回它即可，前端/日志能一次看到六环结果与降级标记
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
