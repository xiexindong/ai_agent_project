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


def new_id(prefix: str) -> str:
    """生成短 ID：run_a1b2c3d4。用 uuid4 前 8 位，够短又不会撞车。"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def now_ms() -> int:
    return int(time.time() * 1000)


class AgentRole(str, Enum):
    """Agent 角色。supervisor 是调度者，其余是执行者。"""

    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    GENERAL = "general"


class AgentSpec(BaseModel):
    """一个 Agent 的静态定义（配置即代码：团队组成写在这里，也能来自 .env/YAML）。"""

    name: str = Field(description="Agent 唯一名称，如 researcher_1")
    role: AgentRole = Field(description="角色，决定默认提示词与可用工具")
    description: str = Field(default="", description="一句话职责说明，给 supervisor 选人时看")
    system_prompt: str = Field(default="", description="角色提示词；留空则用角色默认提示词")
    tools: list[str] = Field(default_factory=list, description="允许调用的工具名白名单")
    multimodal: bool = Field(default=False, description="是否能接收图片（看图走 VisionClient）")
    model: str = Field(default="", description="指定模型；留空用全局默认模型")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_steps: int = Field(default=3, ge=1, le=20, description="单个 Agent 最多推理步数")
    enabled: bool = Field(default=True)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Agent 名称不能为空")
        return v.strip()


class SubTask(BaseModel):
    """一个子任务。depends_on 填上游 SubTask 的 id，构成 DAG。"""

    id: str = Field(description="子任务唯一 id，如 t1")
    assignee: str = Field(description="执行它的 Agent 名称（对应 AgentSpec.name）")
    instruction: str = Field(description="给该 Agent 的具体指令")
    depends_on: list[str] = Field(default_factory=list, description="依赖的上游任务 id 列表")
    priority: int = Field(default=1, ge=1, le=5, description="数字越大越优先")


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

    @model_validator(mode="after")
    def _must_have_plan_or_answer(self) -> "PlanResult":
        if not self.subtasks and not self.direct_answer.strip():
            raise ValueError("PlanResult 必须给出 subtasks 或 direct_answer 之一")
        return self

    @field_validator("subtasks")
    @classmethod
    def _unique_ids(cls, v: list[SubTask]) -> list[SubTask]:
        ids = [t.id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"子任务 id 重复: {ids}")
        return v

    @property
    def is_direct(self) -> bool:
        return not self.subtasks


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

    @classmethod
    def failed(cls, agent: str, task_id: str, error: str, **kw) -> "AgentOutput":
        """统一构造失败产出（避免各处手写 success=False 忘了填 error）。"""
        return cls(agent=agent, task_id=task_id, success=False, error=error, **kw)


class TraceEvent(BaseModel):
    """一条追踪事件。harness 全程往 RunResult.trace 里追加，排查全靠它。"""

    ts: int = Field(default_factory=now_ms)
    kind: Literal["plan", "dispatch", "agent_start", "agent_end", "synthesize", "review", "done", "error"] = Field(
        description="事件类型"
    )
    agent: str = Field(default="")
    message: str = Field(default="")
    payload: dict = Field(default_factory=dict)


class RunStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"      # 有 worker 失败，但仍有汇总答案
    FAILED = "failed"        # 没有任何可用产出
    TIMEOUT = "timeout"


class RunResult(BaseModel):
    """一次团队运行的完整结果（可直接作为 API 响应体）。"""

    run_id: str = Field(default_factory=lambda: new_id("run"))
    query: str = Field(default="")
    status: RunStatus = Field(default=RunStatus.SUCCESS)
    plan: Optional[PlanResult] = None
    outputs: list[AgentOutput] = Field(default_factory=list)
    final_answer: str = Field(default="")
    review: Optional[AgentOutput] = None
    trace: list[TraceEvent] = Field(default_factory=list)
    duration_ms: int = Field(default=0)
    multimodal: bool = Field(default=False)
    error: str = Field(default="")

    @property
    def ok(self) -> bool:
        return self.status in (RunStatus.SUCCESS, RunStatus.PARTIAL)

    @property
    def failed_agents(self) -> list[str]:
        return [o.agent for o in self.outputs if not o.success]


class TeamConfig(BaseModel):
    """团队配置：一组 AgentSpec。默认团队由 harness 构造。"""

    name: str = Field(default="default_team")
    agents: list[AgentSpec] = Field(default_factory=list)
    max_workers: int = Field(default=4, ge=1, le=16, description="并行执行子任务的线程数")
    max_total_steps: int = Field(default=30, ge=1, le=200, description="整次运行的总步数上限")
    timeout_seconds: float = Field(default=120.0, gt=0, description="整次运行总超时（秒）")
    review_enabled: bool = Field(default=True, description="是否让 reviewer 终审")

    @model_validator(mode="after")
    def _check_team(self) -> "TeamConfig":
        names = [a.name for a in self.agents if a.enabled]
        if len(names) != len(set(names)):
            raise ValueError(f"Agent 名称重复: {names}")
        if self.agents and not any(a.role == AgentRole.SUPERVISOR for a in self.agents if a.enabled):
            raise ValueError("团队必须包含且仅包含一个 enabled 的 supervisor")
        known = {a.name for a in self.agents}
        return self


class EvalCase(BaseModel):
    """一个评测用例。关键词命中 = 最朴素也最实用的回归测试。"""

    name: str
    query: str
    expected_keywords: list[str] = Field(default_factory=list, description="答案里应出现的关键词")
    max_duration_ms: int = Field(default=20000, description="超时即判不通过")
    require_all_agents: bool = Field(default=False, description="是否要求所有 Agent 都成功")


class EvalResult(BaseModel):
    """单条用例的评测结果。"""

    case: str
    passed: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="")
    duration_ms: int = Field(default=0)
    status: str = Field(default="")


class EvalReport(BaseModel):
    """整份评测报告。"""

    total: int = 0
    passed: int = 0
    score: float = 0.0
    results: list[EvalResult] = Field(default_factory=list)

    @property
    def pass_rate(self) -> str:
        return f"{self.passed}/{self.total}"


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


def _simplify(node: dict, root: dict) -> Any:
    """递归把 JSON Schema 压成"字段名→类型"的紧凑结构（带 $ref 解引用）。"""
    if "$ref" in node:
        ref = node["$ref"].split("/")[-1]
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
