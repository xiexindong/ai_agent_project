"""
任务规划（Plan-and-Execute）测试
=================================
覆盖：复杂度判定、计划生成与解析、步骤推进、失败后重规划、关闭时行为不变。

【怎么跑】
    python -m pytest tests/test_agent_planning.py -q

【为什么用假 LLM】
    规划要调 LLM，但测试不该依赖真模型（慢、要钱、结果不稳定）。
    这里用 _FakeLLM 按脚本返回文本，既验证解析逻辑，也验证"模型胡说时会不会崩"。
"""
from __future__ import annotations

import pytest

from modules.agent.planner import (
    STATUS_DONE, STATUS_FAILED, STATUS_PENDING, STATUS_SKIPPED,
    ExecutionPlan, PlanStep, TaskPlanner,
)
from modules.agent.react_agent import ReActAgent
from modules.agent.state import AgentState
from modules.tools.base_tool import ToolResult


# 【什么】按脚本返回文本的假 LLM
# 【作用】让规划器不依赖真模型；顺便记下收到的 prompt 供断言
class _FakeLLM:
    def __init__(self, replies: list):
        self.replies = list(replies)
        self.prompts: list = []

    def invoke(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "{}"

    def invoke_with_tools(self, messages, tools=None, **kwargs) -> dict:
        self.prompts.append(str(messages))
        return {"content": "好的", "tool_calls": None}


# 【什么】假工具注册表
# 【作用】让 act_step 能"执行"工具并返回预设结果，不碰真实工具
class _FakeRegistry:
    def __init__(self, result: ToolResult):
        self._result = result
        self.calls: list = []

    def list_tools(self):
        return []

    def get_openai_tools_schema(self, user_level: str = "normal", allowed=None) -> list:
        return []                      # 空清单：本用例不测 Function Calling，只测计划链路

    def execute_tool(self, name, params, **kwargs) -> ToolResult:
        self.calls.append({"name": name, "params": params})
        return self._result


def _planner(replies: list, **kw) -> tuple[TaskPlanner, _FakeLLM]:
    llm = _FakeLLM(replies)
    return TaskPlanner(llm=llm, enabled=True, **kw), llm


# ----------------------------------------------------------------------
# ① 复杂度判定：复杂流程才规划
# ----------------------------------------------------------------------
def test_complex_query_needs_planning():
    p, _ = _planner([])
    assert p.needs_planning("查一下北京今天的天气，然后写一首关于它的短诗") is True


def test_simple_query_does_not_need_planning():
    p, _ = _planner([])
    assert p.needs_planning("3*7 等于多少") is False
    assert p.needs_planning("你好") is False


def test_disabled_planner_never_plans():
    p = TaskPlanner(llm=_FakeLLM([]), enabled=False)
    assert p.needs_planning("先查天气然后写诗再总结成报告") is False


def test_step_marks_in_query_trigger_planning():
    p, _ = _planner([])
    assert p.needs_planning("第一步查订单，第二步退款，第三步通知用户") is True


# ----------------------------------------------------------------------
# ② 计划生成与解析
# ----------------------------------------------------------------------
def test_create_plan_parses_and_sorts_by_dependency():
    reply = ('{"rationale": "先取数再汇总", "steps": ['
             '{"id": "s2", "instruction": "汇总两年营收", "tool_hint": "", "depends_on": ["s1"]},'
             '{"id": "s1", "instruction": "查 2024 营收", "tool_hint": "data_query", "depends_on": []}]}')
    p, _ = _planner([reply])
    plan = p.create_plan("对比 2024 与 2025 营收")
    # 拓扑排序后：被依赖的 s1 必须排在 s2 前面
    assert [s.id for s in plan.steps] == ["s1", "s2"]
    assert plan.rationale == "先取数再汇总"
    assert plan.steps[0].tool_hint == "data_query"
    assert plan.current().id == "s1"


def test_create_plan_falls_back_on_garbage():
    p, _ = _planner(["这不是 JSON，是模型的胡言乱语"])
    plan = p.create_plan("查天气然后写诗")
    # 降级成单步计划：步骤还在、不抛异常，ReAct 照常能跑
    assert len(plan.steps) == 1
    assert plan.steps[0].instruction == "查天气然后写诗"


def test_create_plan_falls_back_when_llm_down():
    class _BrokenLLM:
        def invoke(self, *a, **k):
            raise RuntimeError("连接超时")
    p = TaskPlanner(llm=_BrokenLLM(), enabled=True)
    plan = p.create_plan("查天气然后写诗")
    assert len(plan.steps) == 1


def test_create_plan_caps_step_count():
    steps = ",".join(f'{{"id":"s{i}","instruction":"做第{i}件事","depends_on":[]}}' for i in range(1, 10))
    p, _ = _planner([f'{{"rationale":"很多步","steps":[{steps}]}}'], max_steps=3)
    plan = p.create_plan("做很多事")
    assert len(plan.steps) == 3            # 超出 max_steps 的部分被截掉


# ----------------------------------------------------------------------
# ③ 进度推进
# ----------------------------------------------------------------------
def test_plan_progress_and_completion():
    plan = ExecutionPlan(query="查天气再写诗", rationale="先事实后创作",
                         steps=[PlanStep(id="s1", instruction="查天气"),
                                PlanStep(id="s2", instruction="写诗", depends_on=["s1"])])
    assert plan.current().id == "s1"
    plan.mark("s1", STATUS_DONE, "晴 26℃")
    assert plan.current().id == "s2"
    assert plan.is_complete() is False
    text = plan.progress_text()
    assert "[x] s1" in text and "晴 26℃" in text
    assert "当前步骤：s2" in text
    plan.mark("s2", STATUS_DONE, "诗成")
    assert plan.is_complete() is True
    assert "所有步骤已完成" in plan.progress_text()   # 做完要提示汇总，别再调工具


def test_failed_step_closes_the_plan():
    plan = ExecutionPlan(query="查天气", steps=[PlanStep(id="s1", instruction="查天气")])
    plan.mark("s1", STATUS_FAILED, "接口超时")
    # failed 也算"不再做"：否则计划永远完不成，模型会卡在"还有步骤没做"
    assert plan.is_complete() is True


# ----------------------------------------------------------------------
# ④ 重规划：只重排剩下的
# ----------------------------------------------------------------------
def test_replan_keeps_done_steps_and_replaces_remaining():
    plan = ExecutionPlan(query="查天气再写诗", steps=[
        PlanStep(id="s1", instruction="查天气", status=STATUS_DONE, result="晴"),
        PlanStep(id="s2", instruction="写诗"),
        PlanStep(id="s3", instruction="配图"),
    ])
    reply = '{"rationale":"换个写法","steps":[{"id":"r1","instruction":"改写诗句","depends_on":[]}]}'
    p, _ = _planner([reply], replan_enabled=True, max_replans=1)
    new = p.replan(plan, observation="写诗工具失败")
    assert [s.id for s in new.done()] == ["s1"]            # 已完成的成果保留
    assert [s.id for s in new.pending()] == ["r1"]         # 剩下的被替换
    assert new.replan_count == 1
    assert all(s.status == STATUS_SKIPPED for s in new.steps if s.id in ("s2", "s3"))


def test_replan_respects_max_replans():
    plan = ExecutionPlan(query="q", steps=[PlanStep(id="s1", instruction="做事")], replan_count=1)
    p, _ = _planner(['{"steps":[{"id":"r1","instruction":"重做"}]}'], max_replans=1)
    assert p.replan(plan) is plan          # 次数用尽：原样返回


def test_replan_disabled_keeps_plan():
    plan = ExecutionPlan(query="q", steps=[PlanStep(id="s1", instruction="做事")])
    p, _ = _planner(['{"steps":[{"id":"r1","instruction":"重做"}]}'], replan_enabled=False)
    assert p.replan(plan) is plan


# ----------------------------------------------------------------------
# ⑤ 接进 ReAct：计划写入、注入 prompt、执行后推进
# ----------------------------------------------------------------------
def test_agent_writes_plan_into_state():
    reply = ('{"rationale":"先算再答","steps":[{"id":"s1","instruction":"算 3*7",'
             '"tool_hint":"calculator","depends_on":[]}]}')
    planner, _ = _planner([reply])
    agent = ReActAgent(llm_client=_FakeLLM([]), planner=planner)
    state = AgentState(user_query="先算一下 3*7，然后把结果写进一句话总结")
    assert agent.should_plan(state) is True
    agent.plan_step(state)
    assert len(state.sub_tasks) == 1
    assert state.sub_tasks[0]["tool_hint"] == "calculator"
    assert state.plan_rationale == "先算再答"
    assert state.current_task_index == 0


def test_agent_skips_planning_when_disabled():
    agent = ReActAgent(llm_client=_FakeLLM([]), planner=None)
    state = AgentState(user_query="先查天气然后写诗再总结成报告")
    assert agent.planner is None
    assert agent.should_plan(state) is False
    assert agent._plan_context(state) == ""      # 没有计划 → prompt 一字不改


def test_plan_context_injected_into_reason_prompt():
    planner, _ = _planner(['{"rationale":"先算","steps":[{"id":"s1","instruction":"算 3*7"}]}'])
    agent = ReActAgent(llm_client=_FakeLLM([]), planner=planner)
    state = AgentState(user_query="先算 3*7 再写总结报告")
    agent.plan_step(state)
    ctx = agent._plan_context(state)
    assert "执行计划" in ctx and "当前步骤：s1" in ctx


def test_act_step_advances_plan():
    planner, _ = _planner(['{"rationale":"一步","steps":[{"id":"s1","instruction":"算 3*7"}]}'])
    registry = _FakeRegistry(ToolResult(success=True, result="21", error="", duration_ms=1))
    agent = ReActAgent(llm_client=_FakeLLM([]), tool_registry=registry, planner=planner)
    state = AgentState(user_query="先算 3*7 再总结成一句话")
    agent.plan_step(state)
    state.action = {"name": "calculator", "arguments": {"expression": "3*7"}}
    agent.act_step(state)
    assert state.sub_tasks[0]["status"] == STATUS_DONE
    assert state.sub_tasks[0]["result"] == "21"
    assert state.current_task_index == 1
    assert state.task_results[0]["status"] == STATUS_DONE


def test_plan_node_runs_inside_langgraph():
    """端到端：复杂 query 进图后，plan 节点真的跑过并把计划写进终态。"""
    from modules.agent.graph import AgentGraph

    planner, _ = _planner(['{"rationale":"先算后总结","steps":[{"id":"s1","instruction":"算 3*7"}]}'])
    registry = _FakeRegistry(ToolResult(success=True, result="21", error="", duration_ms=1))
    agent = ReActAgent(llm_client=_FakeLLM([]), tool_registry=registry, planner=planner)
    result = AgentGraph(agent).run("先算一下 3*7，然后把结果写成一句话总结", "sess-plan", "u1")
    # 计划已写进图状态：说明 prepare → plan → reason 这条边真的接通了
    assert result.get("sub_tasks")
    assert result["sub_tasks"][0]["instruction"] == "算 3*7"
    assert result.get("plan_rationale") == "先算后总结"


def test_act_step_failure_marks_failed_and_replans():
    planner, _ = _planner([
        '{"rationale":"两步","steps":[{"id":"s1","instruction":"查天气"},{"id":"s2","instruction":"写诗"}]}',
        '{"rationale":"改一步","steps":[{"id":"r1","instruction":"换个方式写诗"}]}',   # 重规划回答
    ], replan_enabled=True, max_replans=1)
    registry = _FakeRegistry(ToolResult(success=False, result=None, error="接口超时", duration_ms=1))
    agent = ReActAgent(llm_client=_FakeLLM([]), tool_registry=registry, planner=planner)
    state = AgentState(user_query="先查天气然后写诗总结成报告")
    agent.plan_step(state)
    state.action = {"name": "weather", "arguments": {"city": "北京"}}
    agent.act_step(state)
    assert state.sub_tasks[0]["status"] == STATUS_FAILED
    assert state.plan_replan_count == 1
    assert [s["id"] for s in state.sub_tasks if s["status"] == STATUS_PENDING] == ["r1"]
