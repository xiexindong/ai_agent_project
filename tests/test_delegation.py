"""
委派（Delegation）测试
======================
覆盖：治理五道闸、执行与回传、并入委派方产出、关闭时行为不变。

【怎么跑】
    python -m pytest tests/test_delegation.py -q

【为什么用假 LLM】
    委派要调 LLM 做"要不要转派"的决策，测试不该依赖真模型。
    这里用 FakeLLM 按 prompt 关键字返回脚本化结果，
    既验证解析逻辑，也验证"模型填一半时会不会崩"。
"""
from __future__ import annotations

import pytest

from modules.multiagent.agents import RunContext, WorkerAgent
from modules.multiagent.delegation import (
    DelegationExecutor, DelegationLedger, DelegationPolicy, decide_delegation,
)
from modules.multiagent.harness import AgentHarness
from modules.multiagent.models import (
    AgentOutput, AgentRole, AgentSpec, DelegationDecision, DelegationRecord,
    SubTask, TeamConfig,
)


# 【什么】按脚本返回结果的假 LLM
# 【作用】三种调用各按关键字应答：规划 / 委派决策 / 普通产出
class FakeLLM:
    def __init__(self, delegate: bool = True, to_agent: str = "b"):
        self.delegate = delegate
        self.to_agent = to_agent
        self.prompts: list = []

    def invoke_structured(self, prompt: str, system_prompt: str = "", temperature: float = 0.2) -> dict:
        self.prompts.append(prompt)
        if "should_delegate" in prompt:                      # 委派决策
            return {"should_delegate": self.delegate, "to_agent": self.to_agent,
                    "instruction": "查 Q3/Q4 营收", "reason": "缺原始数据"}
        return {                                             # 主管规划
            "thought": "先查后算", "direct_answer": "", "needs_review": False,
            "subtasks": [{"id": "t1", "assignee": "a", "instruction": "算环比",
                          "depends_on": [], "priority": 1}],
        }

    def invoke(self, prompt: str, system_prompt: str = "", temperature: float = 0.3,
               max_tokens: int = 2048) -> str:
        self.prompts.append(prompt)
        if "你委派出去的工作产出" in prompt:                  # 委派方拿到了受托产出
            return "拿到委派数据后算得环比 +9%"
        return "Q3 1100 万，Q4 1200 万"                      # 受托方的产出

    def invoke_with_tools(self, messages, tools, temperature: float = 0.3) -> dict:
        return {"content": "ok", "tool_calls": None}


def _roster(**kw) -> dict:
    """造一份花名册：a 有委派资格，b/c 是候选受托方，s 是主管（不可被委派）。"""
    return {
        "a": AgentSpec(name="a", role=AgentRole.ANALYST, can_delegate=True, **kw),
        "b": AgentSpec(name="b", role=AgentRole.RESEARCHER, can_delegate=True),
        "c": AgentSpec(name="c", role=AgentRole.RESEARCHER),
        "s": AgentSpec(name="s", role=AgentRole.SUPERVISOR),
        "off": AgentSpec(name="off", role=AgentRole.RESEARCHER, enabled=False),
    }


def _ledger(**policy_kw) -> DelegationLedger:
    return DelegationLedger(DelegationPolicy(enabled=True, **policy_kw), roster=_roster())


# ---------------------------------------------------------------------------
# ① 治理五道闸
# ---------------------------------------------------------------------------
def test_disabled_policy_rejects_everything():
    lg = DelegationLedger(DelegationPolicy(enabled=False), roster=_roster())
    ok, why = lg.can("a", "b")
    assert ok is False and "未启用" in why


def test_delegate_to_self_is_rejected():
    ok, why = _ledger().can("a", "a")
    assert ok is False and "自己" in why


def test_unknown_and_disabled_target_rejected():
    lg = _ledger()
    assert lg.can("a", "ghost")[0] is False and "花名册" in lg.can("a", "ghost")[1]
    assert lg.can("a", "off")[0] is False and "停用" in lg.can("a", "off")[1]


def test_agent_without_permission_rejected():
    lg = DelegationLedger(DelegationPolicy(enabled=True), roster={
        "a": AgentSpec(name="a", role=AgentRole.ANALYST),      # can_delegate 默认 False
        "b": AgentSpec(name="b", role=AgentRole.RESEARCHER),
    })
    ok, why = lg.can("a", "b")
    assert ok is False and "被授权" in why


def test_whitelist_rejects_outsider():
    lg = DelegationLedger(DelegationPolicy(enabled=True),
                          roster=_roster(delegate_to=["b"]))
    assert lg.can("a", "c")[0] is False and "白名单" in lg.can("a", "c")[1]
    assert lg.can("a", "b")[0] is True


def test_cycle_is_rejected():
    # a→b 之后 b 想把活转回 a：链上已经有 a，直接拦掉（否则两人互相甩锅到预算耗尽）
    ok, why = _ledger().can("b", "a", depth=1, chain=["a"])
    assert ok is False and "成环" in why


def test_depth_limit_rejected():
    ok, why = _ledger(max_depth=1).can("b", "c", depth=1)
    assert ok is False and "深度" in why


def test_per_run_limit_rejected():
    lg = _ledger(max_per_run=1)
    lg.open(DelegationRecord(from_agent="a", to_agent="b"))
    ok, why = lg.can("a", "c")
    assert ok is False and "运行委派上限" in why


def test_per_agent_limit_rejected():
    lg = _ledger(max_per_agent=1)
    lg.open(DelegationRecord(from_agent="a", to_agent="b"))
    ok, why = lg.can("a", "c")
    assert ok is False and "单人委派上限" in why
    assert lg.can("b", "c")[0] is True          # 换个人，额度还在


def test_denials_are_recorded():
    lg = _ledger()
    lg.can("a", "ghost")
    assert lg.denials and lg.denials[0]["reason"]


# ---------------------------------------------------------------------------
# ② 决策解析：填一半要降级，不能派出空任务
# ---------------------------------------------------------------------------
def test_decision_downgrades_when_incomplete():
    d = DelegationDecision(should_delegate=True, to_agent="b", instruction="")
    assert d.should_delegate is False           # 缺 instruction → 不委派


def test_decide_delegation_returns_none_on_parse_failure():
    class BrokenLLM:
        def invoke_structured(self, *a, **k):
            raise RuntimeError("模型不可用")
    assert decide_delegation(BrokenLLM(), "算环比", []) is None


def test_decide_delegation_parses_decision():
    got = decide_delegation(FakeLLM(), "算环比", list(_roster().values()))
    assert got is not None and got.should_delegate is True and got.to_agent == "b"


# ---------------------------------------------------------------------------
# ③ 执行：真跑一次，产出回传
# ---------------------------------------------------------------------------
def test_executor_runs_delegate_and_returns_record():
    called = {}

    def run_fn(agent_name, task, extra):
        called["name"] = agent_name
        called["task"] = task
        called["chain"] = extra["extra"]["delegation_chain"]
        return AgentOutput(agent=agent_name, task_id=task.id, content="Q3 1100 万", success=True)

    lg = _ledger()
    ex = DelegationExecutor(run_fn=run_fn, ledger=lg, roster=_roster())
    rec = ex.delegate("a", "b", "查 Q3/Q4 营收", parent_task_id="t1")

    assert rec is not None and rec.success is True
    assert rec.content == "Q3 1100 万"
    assert called["name"] == "b"
    assert called["task"].assignee == "b"                       # 受托方真被派了任务
    assert called["chain"] == ["a"]                             # 委派链带下去，供下游防环
    assert called["task"].id.startswith("t1")                   # task_id 带委派痕迹
    assert lg.count("a") == 1


def test_executor_returns_none_when_rejected():
    ex = DelegationExecutor(run_fn=lambda *a, **k: AgentOutput(agent="b"),
                            ledger=DelegationLedger(DelegationPolicy(enabled=False), roster=_roster()),
                            roster=_roster())
    assert ex.delegate("a", "b", "查数据") is None              # 被拒：不抛异常，委派方自己干


def test_executor_survives_delegate_failure():
    def run_fn(agent_name, task, extra):
        return AgentOutput.failed(agent_name, task.id, "工具超时")
    ex = DelegationExecutor(run_fn=run_fn, ledger=_ledger(), roster=_roster())
    rec = ex.delegate("a", "b", "查数据")
    assert rec is not None and rec.success is False and "超时" in rec.error


def test_executor_captures_exception():
    def run_fn(agent_name, task, extra):
        raise RuntimeError("boom")
    ex = DelegationExecutor(run_fn=run_fn, ledger=_ledger(), roster=_roster())
    rec = ex.delegate("a", "b", "查数据")
    assert rec is not None and rec.success is False and "boom" in rec.error


# ---------------------------------------------------------------------------
# ④ Agent 内委派：产出并入自己的结论
# ---------------------------------------------------------------------------
def test_agent_delegates_and_uses_returned_output():
    def run_fn(agent_name, task, extra):
        return AgentOutput(agent=agent_name, task_id=task.id, content="Q3 1100 万，Q4 1200 万",
                           success=True)

    spec = AgentSpec(name="a", role=AgentRole.ANALYST, can_delegate=True)
    agent = WorkerAgent(spec, llm=FakeLLM())
    ctx = RunContext(query="算环比", delegation=DelegationExecutor(
        run_fn=run_fn, ledger=_ledger(), roster=_roster()))

    out = agent.run(SubTask(id="t1", assignee="a", instruction="算环比"), ctx)

    assert len(out.delegations) == 1
    assert out.delegations[0].to_agent == "b"
    assert out.delegations[0].success is True
    # 委派产出拼进了自己的 prompt：这是"拿到数据后才算出结论"的证据
    assert any("你委派出去的工作产出" in p for p in agent.llm.prompts)
    assert out.content == "拿到委派数据后算得环比 +9%"


def test_agent_without_delegation_context_works_as_before():
    agent = WorkerAgent(AgentSpec(name="a", role=AgentRole.ANALYST, can_delegate=True), llm=FakeLLM())
    out = agent.run(SubTask(id="t1", assignee="a", instruction="算环比"), RunContext(query="算环比"))
    assert out.delegations == [] and out.success is True        # 没装委派器：与改造前一致


# ---------------------------------------------------------------------------
# ⑤ Harness 端到端
# ---------------------------------------------------------------------------
def _team_cfg(**kw) -> TeamConfig:
    return TeamConfig(
        agents=[
            AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR),
            AgentSpec(name="a", role=AgentRole.ANALYST, can_delegate=True),
            AgentSpec(name="b", role=AgentRole.RESEARCHER),
        ],
        review_enabled=False, consensus_enabled=False, share_findings=False,
        **kw,
    )


def test_harness_run_emits_delegation():
    h = AgentHarness(config=_team_cfg(delegation_enabled=True), llm=FakeLLM())
    r = h.run("Q4 环比增长多少？")

    assert len(r.delegations) == 1
    assert r.delegations[0].from_agent == "a" and r.delegations[0].to_agent == "b"
    assert r.delegations[0].success is True
    assert "1100 万" in r.delegations[0].content
    assert any(e.kind == "delegate" for e in r.trace)          # trace 里有委派事件
    # 委派方 a 的产出带委派痕迹：能看出这个结论不是它一个人做的
    a_out = [o for o in r.outputs if o.agent == "a"][0]
    assert len(a_out.delegations) == 1


def test_delegation_appears_in_span_tree():
    h = AgentHarness(config=_team_cfg(delegation_enabled=True), llm=FakeLLM())
    r = h.run("Q4 环比增长多少？")
    # 委派要进链路：只看 trace 事件知道"发生过"，看 span 才知道"卡在哪一环"
    assert "delegate" in [s["kind"] for s in r.spans]
    dsp = [s for s in r.spans if s["kind"] == "delegate"][0]
    assert "analyst" in dsp["name"] or "a→b" in dsp["name"] or "→" in dsp["name"]


def test_harness_delegation_disabled_by_default():
    h = AgentHarness(config=_team_cfg(), llm=FakeLLM())
    r = h.run("Q4 环比增长多少？")
    assert r.delegations == []                                  # 默认关闭：零委派、行为不变
    assert not any(e.kind == "delegate" for e in r.trace)


def test_harness_delegate_api_respects_governance():
    h = AgentHarness(config=_team_cfg(delegation_enabled=True), llm=FakeLLM())
    rec = h.delegate("a", "b", "查 Q3/Q4 营收", query="环比多少")
    assert rec is not None and rec.success is True
    assert h.delegate("a", "b", "再来一次") is not None
    # 单人上限 2（默认）用尽后，第三次被拒 → 返回 None
    assert h.delegate("a", "b", "第三次") is None


def test_harness_delegate_rejects_unknown_agent():
    h = AgentHarness(config=_team_cfg(delegation_enabled=True), llm=FakeLLM())
    assert h.delegate("a", "ghost", "查数据", query="x") is None


def test_delegation_budget_resets_between_runs():
    cfg = _team_cfg(delegation_enabled=True, delegation_max_per_run=1)
    h = AgentHarness(config=cfg, llm=FakeLLM())
    h.run("Q4 环比？")
    # 第二次运行是新的预算：上一轮用掉的配额不应该吃掉这一轮
    r2 = h.run("Q4 同比？")
    assert len(r2.delegations) == 1
