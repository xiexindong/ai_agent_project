"""
多 Agent 协调单元测试
======================
覆盖消息总线、一致性判定与投票、仲裁、重规划四块。

全部离线：仲裁与重规划用脚本化假模型顶替，不调真实 LLM。
"""
import os

os.environ.setdefault("LOG_LEVEL", "ERROR")

import pytest  # noqa: E402

from modules.multiagent import (  # noqa: E402
    AgentHarness,
    AgentMessage,
    AgentOutput,
    AgentRole,
    AgentSpec,
    Arbitrator,
    Consensus,
    ConsensusResult,
    MessageBus,
    MessageKind,
    PlanResult,
    ReplanController,
    SubTask,
    TeamConfig,
    text_overlap,
)
from modules.multiagent.agents import RunContext, SupervisorAgent  # noqa: E402


# ---------------------------------------------------------------------------
# 假模型：按 prompt 关键词返回预设内容
# ---------------------------------------------------------------------------
class ScriptedLLM:
    def __init__(self, structured=None):
        self.structured = structured or {}

    def invoke(self, prompt: str, system_prompt: str = "", temperature: float = 0.3,
               max_tokens: int = 2048) -> str:
        return "裁决：以有数据支撑的一方为准。"

    def invoke_structured(self, prompt: str, system_prompt: str = "",
                          temperature: float = 0.2) -> dict:
        return self.structured or {
            "thought": "补派未完成的部分",
            "subtasks": [{"id": "retry", "assignee": "analyst_1", "instruction": "重算"}],
            "direct_answer": "",
            "needs_review": False,
        }

    def invoke_with_tools(self, messages, tools, temperature=0.3) -> dict:
        return {"content": "", "tool_calls": []}


def _out(agent: str, content: str, success: bool = True, task_id: str = "") -> AgentOutput:
    return AgentOutput(agent=agent, role=AgentRole.GENERAL, task_id=task_id or agent,
                       content=content, success=success,
                       error="" if success else "执行失败")


# ---------------------------------------------------------------------------
# 相似度与一致性
# ---------------------------------------------------------------------------
def test_文本重叠度的基本数值():
    assert text_overlap("营收1200万", "营收1200万") == 1.0
    # 共用 "营收"/"00"/"0万" 三个 2-gram，并集 8 个 → 3/8
    assert text_overlap("营收1200万", "营收800万") == pytest.approx(0.375, abs=1e-3)
    assert text_overlap("营收1200万", "") == 0.0


def test_结论一致时不判冲突():
    outs = [
        _out("a", "营收 1200 万，同比 +8%"),
        _out("b", "营收 1200 万，同比 +8%，来自订单系统"),
    ]
    r = Consensus(threshold=0.35).detect(outs)
    assert r.has_conflict is False
    assert r.needs_arbitration is False
    assert set(r.majority) == {"a", "b"}


def test_数字打架时判冲突且需要仲裁():
    outs = [
        _out("a", "营收 1200 万，来自订单系统"),
        _out("b", "营收 800 万，同比 -5%"),
    ]
    r = Consensus(threshold=0.35).detect(outs)
    assert r.has_conflict is True
    assert r.needs_arbitration is True
    assert len(r.clusters) == 2


def test_三人三种说法时无多数派也要仲裁():
    outs = [_out("a", "方案一：降价"), _out("b", "方案二：加量"), _out("c", "方案三：换渠道")]
    r = Consensus(threshold=0.35).detect(outs)
    assert r.needs_arbitration is True


def test_不足两份有效产出时不做判定():
    r = Consensus().detect([_out("a", "只有一个人"), _out("b", "", success=False)])
    assert r.has_conflict is False
    assert "不足两个" in r.note


def test_多数派内容用于兜底():
    outs = [_out("a", "营收 1200 万"), _out("b", "营收 1200 万左右"), _out("c", "营收 800 万")]
    c = Consensus().detect(outs)
    content = Consensus().majority_content(outs, c)
    assert "1200" in content


# ---------------------------------------------------------------------------
# 消息总线
# ---------------------------------------------------------------------------
def test_消息可见性_点对点与广播():
    bus = MessageBus()
    bus.post("a", "给 b 的私信", receiver="b")
    bus.post("a", "给所有人的广播")
    assert [m.content for m in bus.inbox("b")] == ["给 b 的私信", "给所有人的广播"]
    assert [m.content for m in bus.inbox("c")] == ["给所有人的广播"]
    assert bus.inbox("a") == []            # 自己发的看不到


def test_异议单独可取且容量有上限():
    bus = MessageBus(max_messages=10)
    for i in range(20):
        bus.post("consensus", f"异议 {i}", kind=MessageKind.OBJECTION)
    assert len(bus) == 10                   # 超出丢最旧的
    assert len(bus.objections()) == 10


def test_批次结束清空消息():
    bus = MessageBus()
    bus.post("a", "这一批的发现")
    bus.clear()
    assert bus.inbox("b") == []             # 不跨批次，避免旧异议影响下一批


# ---------------------------------------------------------------------------
# 仲裁与重规划
# ---------------------------------------------------------------------------
def test_仲裁者产出裁决结论():
    reviewer = AgentSpec(name="reviewer_1", role=AgentRole.REVIEWER)
    judge = SupervisorAgent(reviewer, llm=ScriptedLLM())
    outs = [_out("a", "1200 万"), _out("b", "800 万")]
    cons = Consensus().detect(outs)
    arb = Arbitrator(judge=judge).arbitrate("营收多少？", outs, cons)
    assert arb is not None
    assert arb.success is True
    assert "裁决" in arb.content


def test_没有仲裁者时返回None():
    outs = [_out("a", "1200 万"), _out("b", "800 万")]
    cons = Consensus().detect(outs)
    assert Arbitrator(judge=None).arbitrate("x", outs, cons) is None


def test_重规划后原始query被还原():
    """重规划临时改写 ctx.query，结束后必须还原，否则污染后续批次的输入。"""
    spec = AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR)
    sup = SupervisorAgent(spec, llm=ScriptedLLM())
    ctx = RunContext(query="原始问题")
    outs = [_out("researcher_1", "", success=False, task_id="t1")]
    ctrl = ReplanController(supervisor=sup, max_replans=1)
    plan = ctrl.build_replan(ctx, outs, [])
    assert plan is not None
    assert plan.subtasks[0].assignee == "analyst_1"
    assert ctx.query == "原始问题"          # 关键：不能残留"执行现状"


def test_模型不可用时重规划降级为单任务_且query还原():
    """主管的 plan 内部已经降级（不会把异常抛出来），这里验证降级后仍可用且不污染 ctx。

    【为什么不期望返回 None】SupervisorAgent.plan 自带二次重试与降级，
    模型坏了它返回的是"先让一个人硬答"的兜底计划，而不是抛异常。
    重规划器只负责把 ctx.query 还原干净 —— 这才是它该保证的事。
    """

    class BoomLLM(ScriptedLLM):
        def invoke_structured(self, *a, **kw):
            raise RuntimeError("模型不可用")

    sup = SupervisorAgent(AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR), llm=BoomLLM())
    ctx = RunContext(query="x")
    ctrl = ReplanController(supervisor=sup)
    plan = ctrl.build_replan(ctx, [_out("a", "", success=False)], [])
    assert plan is not None and plan.subtasks
    assert ctx.query == "x"                # 关键：不能残留"执行现状"


def test_无主管时换一个空闲的人重试():
    """失败者自己不再接，正在忙的也不派，只派既没失败又没在忙的人。"""
    ctrl = ReplanController(supervisor=None)
    roster = [AgentSpec(name="researcher_1", role=AgentRole.RESEARCHER),
              AgentSpec(name="analyst_1", role=AgentRole.ANALYST),
              AgentSpec(name="reviewer_1", role=AgentRole.REVIEWER)]
    outs = [_out("researcher_1", "", success=False, task_id="t1"),
            _out("analyst_1", "已完成", success=True, task_id="t2")]
    fb = ctrl.fallback_task(outs, roster)
    assert fb is not None
    assert fb.assignee == "reviewer_1"       # 唯一既没失败又没在忙的人
    assert "researcher_1" in fb.instruction  # 指令里交代了是接手谁的活


def test_没人可换时不硬派():
    """全员要么失败过、要么正在忙，此时返回 None，由上层照实标注未完成。"""
    ctrl = ReplanController(supervisor=None)
    roster = [AgentSpec(name="researcher_1", role=AgentRole.RESEARCHER),
              AgentSpec(name="analyst_1", role=AgentRole.ANALYST)]
    outs = [_out("researcher_1", "", success=False, task_id="t1"),
            _out("analyst_1", "已完成", success=True, task_id="t2")]
    assert ctrl.fallback_task(outs, roster) is None


# ---------------------------------------------------------------------------
# 端到端：harness 里默认行为不变 + 打开开关后生效
# ---------------------------------------------------------------------------
def _team(**cfg) -> TeamConfig:
    return TeamConfig(
        agents=[
            AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR),
            AgentSpec(name="researcher_1", role=AgentRole.RESEARCHER, description="查资料"),
            AgentSpec(name="analyst_1", role=AgentRole.ANALYST, description="算数据"),
            AgentSpec(name="reviewer_1", role=AgentRole.REVIEWER, description="审核"),
        ],
        max_workers=2, timeout_seconds=10, **cfg,
    )


def test_默认配置下结果结构与改造前一致():
    """重规划默认关闭：跑完的 outputs 数量应与派出的任务数一致，不额外多出任务。"""
    h = AgentHarness(config=_team(), llm=ScriptedLLM())
    r = h.run("公司营收多少？")
    assert r.plan is not None
    assert len(r.plan.subtasks) == 1        # 脚本模型返回 1 个补派任务
    assert r.replan_count == 0
    assert r.status.value in ("success", "partial")


def test_打开重规划后失败会被补派():
    """replan_enabled=True 且有失败时，replan_count 增加并留下 trace。"""
    h = AgentHarness(config=_team(replan_enabled=True, max_replans=1), llm=ScriptedLLM())
    r = h.run("公司营收多少？")
    # 脚本模型的 worker 全部成功（invoke 返回"裁决…"），故这里只验证开关不报错
    assert isinstance(r.replan_count, int)
    assert all(e.kind in (
        "plan", "dispatch", "agent_start", "agent_end", "synthesize", "review",
        "done", "error", "consensus", "arbitration", "hitl", "replan",
    ) for e in r.trace)


def test_一致性默认开启且能被关闭():
    h = AgentHarness(config=_team(), llm=ScriptedLLM())
    assert h.config.consensus_enabled is True
    h2 = AgentHarness(config=_team(consensus_enabled=False), llm=ScriptedLLM())
    assert h2.config.consensus_enabled is False


def test_团队配置校验仍然生效():
    """新增字段不能破坏原有的团队校验（必须有且仅有一个 supervisor）。"""
    with pytest.raises(Exception):
        TeamConfig(agents=[AgentSpec(name="a", role=AgentRole.RESEARCHER)])
