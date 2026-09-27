"""
多 Agent 数据契约模块
=====================
用 pydantic v2 定义多智能体协作的全部数据形状：
角色定义、任务计划、消息、单次输出、追踪事件、运行结果、评测用例。

【为什么多 Agent 必须先定死数据契约】
    多个 Agent 之间靠自然语言传话是最常见的翻车点：
    supervisor 说"去查一下"，worker 回一大段散文，supervisor 再解析这段散文做汇总，
    任一环节格式漂移就全链路静默降质。这里把每一步都定成 BaseModel，
    跨 Agent 传递的是可校验的结构，不是自由文本。

【pydantic 在这里承担的四件事】
    1. 校验   —— 模型返回的 JSON 缺字段/类型错，在 model_validate 当场暴露
    2. 文档   —— Field(description) 直接生成给 LLM 看的结构说明，不再手写 prompt
    3. 序列化 —— RunResult.model_dump() 直接能进 API 响应和 trace 落库
    4. 默认与边界 —— 步数、超时、并行度的下限由 Field(ge=1) 这类约束守住
"""
import time
import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# 【什么】生成带前缀的短 ID
# 【作用】给 run/msg 等实体做可复现的标识；取 uuid4 十六进制前 8 位（如 msg_a1b2c3d4）
def new_id(prefix: str) -> str:
    """生成短 ID：run_a1b2c3d4。用 uuid4 前 8 位，够短又不会撞车。"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# 【什么】当前时间的毫秒时间戳
# 【作用】给 TraceEvent.ts 等字段兜底，统一成毫秒避免与秒混用出错
def now_ms() -> int:
    return int(time.time() * 1000)


# 【什么】Agent 角色枚举
# 【作用】决定默认提示词与可用工具；SUPERVISOR 是唯一的调度者
class AgentRole(str, Enum):
    """Agent 角色。supervisor 是调度者，其余是执行者。"""

    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    GENERAL = "general"


# 【什么】一个 Agent 的静态定义
# 【作用】把「名字、角色、工具白名单、步数预算」固化成一行配置，团队组成由此决定
class AgentSpec(BaseModel):
    """一个 Agent 的静态定义（配置即代码：团队组成写在这里，也能来自 .env/YAML）。"""

    name: str = Field(description="Agent 唯一名称，如 researcher_1")
    role: AgentRole = Field(description="角色，决定默认提示词与可用工具")
    description: str = Field(default="", description="一句话职责说明，给 supervisor 选人时看")
    system_prompt: str = Field(default="", description="角色提示词；留空则用角色默认提示词")
    tools: list[str] = Field(default_factory=list, description="允许调用的工具名白名单")
    multimodal: bool = Field(default=False, description="是否能接收图片（看图走 VisionClient）")
    model: str = Field(default="", description="指定模型；留空用全局默认模型")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)   # 采样温度，0~2；越低越稳定
    max_steps: int = Field(default=3, ge=1, le=20, description="单个 Agent 最多推理步数")
    enabled: bool = Field(default=True)                       # False 的 Agent 不注册、不参与派活

    # 【什么】校验 name 非空
    # 【作用】空名字会让 automatic 注册进字典却无法派活，在入口就拦住
    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Agent 名称不能为空")
        return v.strip()


# 【什么】主管拆出来的一个子任务
# 【作用】用 depends_on 声明上下游，多个 SubTask 构成可拓扑排序的 DAG
class SubTask(BaseModel):
    """一个子任务。depends_on 填上游 SubTask 的 id，构成 DAG。"""

    id: str = Field(description="子任务唯一 id，如 t1")
    assignee: str = Field(description="执行它的 Agent 名称（对应 AgentSpec.name）")
    instruction: str = Field(description="给该 Agent 的具体指令")
    depends_on: list[str] = Field(default_factory=list, description="依赖的上游任务 id 列表")
    priority: int = Field(default=1, ge=1, le=5, description="数字越大越优先")   # 同一批内排序用，范围 1~5


# 【什么】主管的规划结果
# 【作用】要么给出任务清单，要么给出直接答案，二者必有其一
class PlanResult(BaseModel):
    """Supervisor 的规划结果。

    【两种形态二选一，由 model_validator 强制】
        A 拆任务: subtasks=[SubTask(id="t1",assignee="researcher_1",instruction="查 2024 营收")]
        B 直接答: direct_answer="你好，有什么可以帮你？"（寒暄/简单问答，不值得开团队）

    【不强制二选一会怎样】
        模型可能同时返回空 subtasks 和空 direct_answer，
        harness 就会"派了 0 个任务却继续等汇总"，最后返回一句空答案，
        且不报错——典型的静默失败。
    """

    thought: str = Field(default="", description="supervisor 的规划思路，用于 trace 排查")
    subtasks: list[SubTask] = Field(default_factory=list)
    direct_answer: str = Field(default="", description="不拆任务时的直接回答")
    needs_review: bool = Field(default=True, description="是否需要 reviewer 终审")

    # 【什么】强制「拆任务」与「直接答」二选一
    # 【作用】防止模型两边都返回空，导致 harness 派 0 个任务还得等汇总
    @model_validator(mode="after")
    def _must_have_plan_or_answer(self) -> "PlanResult":
        if not self.subtasks and not self.direct_answer.strip():
            raise ValueError("PlanResult 必须给出 subtasks 或 direct_answer 之一")
        return self

    # 【什么】校验子任务 id 不重复
    # 【作用】id 是 ctx.upstream 的字典键，重复会互相覆盖导致上游产出丢失
    @field_validator("subtasks")
    @classmethod
    def _unique_ids(cls, v: list[SubTask]) -> list[SubTask]:
        ids = [t.id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"子任务 id 重复: {ids}")
        return v

    # 【什么】是否走「直接答」分支
    # 【作用】harness 据此跳过派活，直接把 direct_answer 原样返回
    @property
    def is_direct(self) -> bool:
        return not self.subtasks


# 【什么】单个 Agent 一次执行的产出
# 【作用】成功与失败共用同一结构，上游只需看 success 与 content/error
class AgentOutput(BaseModel):
    """单个 Agent 一次执行的产出。失败也用这个结构表达，success=False + error。"""

    agent: str = Field(description="执行的 Agent 名称")
    role: AgentRole = Field(default=AgentRole.GENERAL)
    task_id: str = Field(default="", description="对应 SubTask.id；supervisor 步骤为空")
    content: str = Field(default="", description="产出正文")
    success: bool = Field(default=True)
    error: str = Field(default="")
    steps: int = Field(default=1, description="实际推理步数")
    tool_calls: list[dict] = Field(default_factory=list)
    duration_ms: int = Field(default=0)
    multimodal: bool = Field(default=False, description="本次是否处理了多模态输入")

    # 【什么】统一构造失败产出
    # 【作用】收敛各处手写 success=False 的写法，保证 error 一定被填上
    @classmethod
    def failed(cls, agent: str, task_id: str, error: str, **kw) -> "AgentOutput":
        """统一构造失败产出（避免各处手写 success=False 忘了填 error）。"""
        return cls(agent=agent, task_id=task_id, success=False, error=error, **kw)


# 【什么】trace 事件类型字面量集合
# 【作用】TraceEvent.kind 与 harness/crewai 的 _emit(kind=...) 共用同一套取值，
#        写错字符串在类型检查阶段就暴露，不用等运行时才发现 trace 里混进未知类型
TraceKind = Literal[
    "plan", "dispatch", "agent_start", "agent_end",
    "synthesize", "review", "done", "error",
    # 下面四个是协调与人工介入环节的事件，和上面同一套取值空间，
    # 写错字符串同样在类型检查阶段暴露
    "consensus", "arbitration", "hitl", "replan",
]


# 【什么】一条追踪事件
# 【作用】把运行过程按时间顺序记进 RunResult.trace，事后排查全靠它还原现场
class TraceEvent(BaseModel):
    """一条追踪事件。harness 全程往 RunResult.trace 里追加，排查全靠它。"""

    ts: int = Field(default_factory=now_ms)   # 毫秒时间戳，默认取写入时刻
    kind: TraceKind = Field(description="事件类型")
    agent: str = Field(default="")            # 事件关联的 Agent 名，无归属时留空
    message: str = Field(default="")
    payload: dict = Field(default_factory=dict)


# 【什么】一次团队运行的状态枚举
# 【作用】区分「成功 / 部分成功 / 失败 / 超时」四种结局，给监控与评测用
class RunStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"      # 有 worker 失败，但仍有汇总答案
    FAILED = "failed"        # 没有任何可用产出
    TIMEOUT = "timeout"      # 撞上总 deadline，任务未跑完


# 【什么】一次团队运行的完整结果
# 【作用】作为 API 的直接响应体，同时承载 trace、评测与协调环节的数据
class RunResult(BaseModel):
    """一次团队运行的完整结果（可直接作为 API 响应体）。"""

    run_id: str = Field(default_factory=lambda: new_id("run"))
    query: str = Field(default="")
    status: RunStatus = Field(default=RunStatus.SUCCESS)
    plan: Optional[PlanResult] = None
    outputs: list[AgentOutput] = Field(default_factory=list)
    final_answer: str = Field(default="")
    review: Optional[AgentOutput] = None
    # 【什么】协调环节的三个结果
    # 【作用】与 outputs 分开存：outputs 是"执行者干了什么"，
    #        这三项是"团队如何收敛"——排查"为什么选了这个答案"时看它们
    consensus: Optional["ConsensusResult"] = None          # 产出一致性判定
    arbitration: Optional[AgentOutput] = None              # 冲突仲裁结论
    replan_count: int = Field(default=0, description="实际发生的重规划次数")
    interrupt_count: int = Field(default=0, description="本次运行触发的人工介入次数")
    # ------------------------------------------------------------------
    # 【观测链路的两个索引】
    #   trace_id 是统一观测链路的 ID，凭它能拉出完整的 span 树；
    #   spans 是本次运行的 span 快照（已序列化），
    #   让 API 响应自带链路，不用再去查一遍追踪服务。
    # ------------------------------------------------------------------
    trace_id: str = Field(default="", description="统一观测链路 ID")
    spans: list[dict] = Field(default_factory=list, description="span 树快照（序列化后）")
    trace: list[TraceEvent] = Field(default_factory=list)
    duration_ms: int = Field(default=0)
    multimodal: bool = Field(default=False)
    error: str = Field(default="")

    # 【什么】本次运行是否算「有可用答案」
    # 【作用】SUCCESS 与 PARTIAL 都算通过；FAILED/TIMEOUT 不算
    @property
    def ok(self) -> bool:
        return self.status in (RunStatus.SUCCESS, RunStatus.PARTIAL)

    # 【什么】列出执行失败的 Agent 名
    # 【作用】用于最终状态的降级判断与评测里的「完成率」计算
    @property
    def failed_agents(self) -> list[str]:
        return [o.agent for o in self.outputs if not o.success]


# 【什么】团队运行参数配置
# 【作用】把并发度、步数预算、总超时、协调与人工介入开关集中在一处，便于按环境切换
class TeamConfig(BaseModel):
    """团队配置：一组 AgentSpec。默认团队由 harness 构造。"""

    name: str = Field(default="default_team")
    agents: list[AgentSpec] = Field(default_factory=list)
    max_workers: int = Field(default=4, ge=1, le=16, description="并行执行子任务的线程数")
    max_total_steps: int = Field(default=30, ge=1, le=200, description="整次运行的总步数上限")
    timeout_seconds: float = Field(default=120.0, gt=0, description="整次运行总超时（秒）")
    review_enabled: bool = Field(default=True, description="是否让 reviewer 终审")

    # ------------------------------------------------------------------
    # 【协调相关配置】
    #   默认口径的取舍：一致性检测默认开（它只多记一条 trace，不改答案路径），
    #   重规划默认关（要多调一次 LLM，是真金白银的成本）。
    #   这样装上去默认行为基本不变，想用再打开。
    # ------------------------------------------------------------------
    consensus_enabled: bool = Field(default=True, description="是否检测多个产出之间的一致性")
    consensus_threshold: float = Field(default=0.35, ge=0.0, le=1.0,
                                       description="相似度低于此值判定为冲突")
    arbitration_enabled: bool = Field(default=True, description="冲突时是否让审核者仲裁")
    share_findings: bool = Field(default=True, description="每批结束后是否把产出共享到消息总线")
    replan_enabled: bool = Field(default=False, description="有 Agent 失败时是否让主管重规划")
    max_replans: int = Field(default=1, ge=0, le=3, description="最多重规划几次")

    # ------------------------------------------------------------------
    # 【编排治理配置】限流 / 配额 / 防环 / 校验 / 观测
    #   默认值按"装上去不改变现有行为"来定：
    #   - 限流给得很宽（20 QPS、突发 40），正常协作根本撞不到
    #   - 跳数上限 2：允许"发问 → 回答 → 追问"三跳，再往下就是空转
    #   - 敏感词默认空表：没配规则就不拦，避免误伤
    #   - 观测默认开：它只多写内存里的 span，不额外调 LLM
    # ------------------------------------------------------------------
    orchestration_enabled: bool = Field(default=True, description="是否启用中心化编排（关则消息直连总线）")
    orchestration_max_hops: int = Field(default=2, ge=0, le=10, description="一条对话链最多几跳")
    orchestration_max_per_run: int = Field(default=100, ge=1, description="单次运行消息总量上限")
    orchestration_max_per_round: int = Field(default=30, ge=1, description="单轮消息上限")
    orchestration_max_per_sender_round: int = Field(default=3, ge=1, description="同一发送者单轮上限")
    orchestration_rate_per_sec: float = Field(default=20.0, gt=0, description="全局消息速率（条/秒）")
    orchestration_burst: int = Field(default=40, ge=1, description="令牌桶容量（允许的瞬时突发）")
    orchestration_repeat_limit: int = Field(default=2, ge=1, description="相同内容最多出现几次")
    orchestration_pingpong_limit: int = Field(default=2, ge=1, description="双向各几次判定为踢皮球")
    orchestration_sensitive_words: str = Field(default="", description="敏感词，逗号分隔")
    orchestration_on_sensitive: str = Field(default="redact", description="命中敏感词后：redact|reject")
    orchestration_max_content_chars: int = Field(default=4000, ge=1, description="单条消息正文上限")
    orchestration_trace_enabled: bool = Field(default=True, description="是否记录消息级观测 span")

    # ------------------------------------------------------------------
    # 【人工介入配置】
    #   默认关：开启后链路会在敏感动作处停下来等人，
    #   没有人值守的场合会一直等到超时，所以必须显式打开。
    # ------------------------------------------------------------------
    hitl_enabled: bool = Field(default=False, description="是否启用 Human-in-the-Loop 人工介入")

    # 【什么】团队合法性校验器
    # 【作用】保证 Agent 名不重名、且至少有一个启用状态的 supervisor
    @model_validator(mode="after")
    def _check_team(self) -> "TeamConfig":
        names = [a.name for a in self.agents if a.enabled]
        if len(names) != len(set(names)):
            raise ValueError(f"Agent 名称重复: {names}")
        if self.agents and not any(a.role == AgentRole.SUPERVISOR for a in self.agents if a.enabled):
            raise ValueError("团队必须包含且仅包含一个 enabled 的 supervisor")
        known = {a.name for a in self.agents}
        return self


# 【什么】一个评测用例
# 【作用】用关键词命中 + 耗时边界做回归测试，结果确定、可进 CI
class EvalCase(BaseModel):
    """一个评测用例。关键词命中 = 最朴素也最实用的回归测试。"""

    name: str
    query: str
    expected_keywords: list[str] = Field(default_factory=list, description="答案里应出现的关键词")
    max_duration_ms: int = Field(default=20000, description="超时即判不通过")   # 默认 20 秒
    require_all_agents: bool = Field(default=False, description="是否要求所有 Agent 都成功")


# 【什么】单条用例的评测结果
# 【作用】记录得分、通过与失败原因，构成报告的最小单元
class EvalResult(BaseModel):
    """单条用例的评测结果。"""

    case: str
    passed: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)   # 0~1 综合分，0.6 及以上才判通过
    reason: str = Field(default="")
    duration_ms: int = Field(default=0)
    status: str = Field(default="")


# 【什么】整份评测报告
# 【作用】汇总所有用例结果并给出平均分，供一次回归的整体判断
class EvalReport(BaseModel):
    """整份评测报告。"""

    total: int = 0
    passed: int = 0
    score: float = 0.0
    results: list[EvalResult] = Field(default_factory=list)

    # 【什么】通过率的展示形式
    # 【作用】返回「通过数/总数」字符串，便于直接打印和写入日志
    @property
    def pass_rate(self) -> str:
        return f"{self.passed}/{self.total}"


# ---------------------------------------------------------------------------
# 【什么】Agent 之间直接对话的消息契约
# 【作用】让执行者之间能"求助/质疑/共享发现"，而不必事事绕回主管
#
#   【为什么 Supervisor-Worker 之外还需要横向对话】
#       纯中心化有个死角：researcher 查到的数据 analyst 觉得口径不对，
#       按中心化只能各自把结论交给主管，由主管发现"两个人对不上"——
#       这时两个人都已经做完了，返工成本已经付出。
#       横向消息让 analyst 当场发一条 objection 给 researcher，
#       在同一个批次内就把口径对齐。
#
#   【为什么必须有上限与收敛】
#       放开自由对话会退化成"群聊式多 Agent"：消息数随人数平方增长，
#       且没人负责收口。这里的约束是：消息只在同一批次内有效、
#       总线有容量上限、objection 只作为仲裁的输入，不驱动新的执行轮次。
# ---------------------------------------------------------------------------
# 【什么】Agent 间消息的类型枚举
# 【作用】区分共享发现、请求澄清、转派、异议四种横向沟通意图
class MessageKind(str, Enum):
    """Agent 间消息类型。"""

    FINDING = "finding"        # 共享发现：这个事实供你参考
    CLARIFY = "clarify"        # 请求澄清：你的产出里这点我没看懂
    DELEGATE = "delegate"      # 转派：这部分更适合你来做
    OBJECTION = "objection"    # 异议：我认为你的结论有问题


# 【什么】一条 Agent 之间的点对点或广播消息
# 【作用】承载横向沟通内容；receiver 为 "*" 时广播给所有其他成员
class AgentMessage(BaseModel):
    """一条 Agent 之间的消息。

    receiver 语义：
        具体 Agent 名  → 点对点（如 analyst_1 → researcher_1）
        "*"            → 广播给所有其他成员
    """

    id: str = Field(default_factory=lambda: new_id("msg"))
    sender: str = Field(description="发送者 Agent 名")
    receiver: str = Field(default="*", description="接收者 Agent 名，* 表示广播")
    kind: MessageKind = Field(default=MessageKind.FINDING)
    content: str = Field(default="", description="消息正文")
    ref_task_id: str = Field(default="", description="关联的 SubTask.id")
    ts: int = Field(default_factory=now_ms)

    # 【什么】判断某 Agent 是否能看到这条消息
    # 【作用】广播（receiver="*"）对所有人可见，点对点消息只给指定接收者
    def visible_to(self, name: str) -> bool:
        return self.receiver == "*" or self.receiver == name


# 【什么】多个产出之间的一致性判定结果
# 【作用】用确定性的文本重叠度判断成员是否互相矛盾，决定要不要触发仲裁
class ConsensusResult(BaseModel):
    """多个产出的一一致性判定结果。

    【为什么用文本重叠度而不是让 LLM 判"谁对"】
        LLM 判定更准，但每多一次判定就多一次调用，
        而且它自己也会漂移 —— 不适合作为每批都跑的确定性开关。
        文本重叠度（字符 2-gram Jaccard）是确定性的：
        今天判冲突，明天同样输入还判冲突，适合做 CI 门禁和 trace 追溯。
        LLM 只在确认有冲突后、且开启了仲裁时才介入。

    【输入 → 输出对照】
        输入: outputs = [A("营收 1200 万"), B("营收 1200 万，同比 +8%"), C("营收 800 万")]
              threshold = 0.35
        输出: ConsensusResult(has_conflict=True, similarity=0.21,
                             clusters=[["a","b"], ["c"]], majority=["a","b"],
                             needs_arbitration=True)
    """

    has_conflict: bool = Field(default=False, description="产出之间是否存在实质分歧")
    similarity: float = Field(default=1.0, ge=0.0, le=1.0, description="产出两两相似度的最小值")
    clusters: list[list[str]] = Field(default_factory=list, description="按相似度聚类后的 Agent 分组")
    majority: list[str] = Field(default_factory=list, description="最大簇（多数派）")
    needs_arbitration: bool = Field(default=False, description="是否需要仲裁（无多数派或存在冲突）")
    note: str = Field(default="", description="判定说明，进 trace")


# 【什么】把 pydantic 模型压缩成给 LLM 看的结构说明
# 【作用】让模型按我们定死的字段名与类型填空，同时比原始 schema 省一半以上 token
def json_schema_hint(model: type[BaseModel]) -> str:
    """
    把 pydantic 模型的 JSON Schema 精简成"给 LLM 看的结构说明"。

    【为什么不让 LLM 自由发挥，而要喂 schema】
        直接说"请返回 JSON" → 模型可能返回 {"tasks": [...]} 这种自创键名，
        解析失败还得猜。喂 model_json_schema() 后，字段名和类型都由我们定死，
        模型只是在格子里填空，稳定性质变。

    【输入 → 输出对照】
        输入: json_schema_hint(PlanResult)
        输出: '{"thought": "string", "subtasks": [{"id": "string", "assignee": "string",
               "instruction": "string", "depends_on": ["string"], "priority": "integer"}],
               "direct_answer": "string", "needs_review": "boolean"}'

    【为什么要精简】
        原始 model_json_schema() 带 title/$defs/嵌套定义，一个 PlanResult 能占
        七八百 token。这里只保留"字段名→类型"这一层，token 省一半以上。
    """
    # 【为什么必须 json.dumps 而不是直接 f-string 这个 dict】
    #   Python dict 的字符串表示是单引号（{'a': 'b'}），不是合法 JSON。
    #   直接塞进 prompt 会让模型学到一个错误示范：它可能照抄单引号，
    #   而 json.loads 只认双引号 —— 于是解析失败，还得再花一轮重试。
    import json
    return json.dumps(_simplify(model.model_json_schema(), model.model_json_schema()),
                      ensure_ascii=False)


# 【什么】递归压缩 JSON Schema 的内部实现
# 【作用】逐层把 object/array/enum 变成「字段名→类型」，并处理 $ref 解引用
def _simplify(node: dict, root: dict) -> Any:
    """递归把 JSON Schema 压成"字段名→类型"的紧凑结构（带 $ref 解引用）。"""
    if "$ref" in node:
        ref = node["$ref"].split("/")[-1]   # "#/$defs/SubTask" → "SubTask"
        return _simplify(root.get("$defs", {}).get(ref, {}), root)
    t = node.get("type")
    if t == "object" or "properties" in node:
        props = node.get("properties", {})
        return {k: _simplify(v, root) for k, v in props.items()}
    if t == "array":
        items = node.get("items", {})
        return [_simplify(items, root)] if items else ["any"]
    if isinstance(t, list):
        return t[0]
    if t:
        return t
    if "enum" in node:
        return "enum"
    return "any"


# ---------------------------------------------------------------------------
# 【为什么要在文件末尾显式 rebuild】
#   RunResult 用字符串前向引用了 ConsensusResult（"ConsensusResult"），
#   而 ConsensusResult 的定义位置在 RunResult 之后。
#   RunResult 定义那一刻 pydantic 还找不到这个名字，
#   必须等本模块整体加载完再重建一次模型，
#   否则第一次实例化 RunResult 就会抛"名字未定义"。
# ---------------------------------------------------------------------------
RunResult.model_rebuild()
