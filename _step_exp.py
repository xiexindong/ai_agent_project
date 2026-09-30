"""临时脚本：真实驱动 plan_step → act_step 循环，观察计划一步一步被推进"""
import json
import types

from modules.agent.react_agent import ReActAgent
from modules.agent.state import AgentState
from modules.agent.planner import TaskPlanner
from modules.tools.base_tool import ToolResult

QUERY = "帮我统计 2024 年 Q4 各区域销售额，然后和 Q3 对比，最后整理成一页分析报告"

PLAN_V1 = json.dumps({
    "rationale": "先取两个季度的事实数据，再对比，最后成文",
    "steps": [
        {"id": "s1", "instruction": "查询 2024 Q4 各区域销售额", "tool_hint": "data_query", "depends_on": []},
        {"id": "s2", "instruction": "查询 2024 Q3 各区域销售额", "tool_hint": "data_query", "depends_on": []},
        {"id": "s3", "instruction": "对比两季度数据并整理成一页报告", "tool_hint": "report_write", "depends_on": ["s1", "s2"]},
    ],
}, ensure_ascii=False)

PLAN_V2 = json.dumps({
    "rationale": "Q3 明细无权限，改从月度汇总表取 Q3",
    "steps": [
        {"id": "r1", "instruction": "从月度汇总表获取 Q3 各区域销售额", "tool_hint": "data_query", "depends_on": []},
        {"id": "r2", "instruction": "对比两季度数据并整理成一页报告", "tool_hint": "report_write", "depends_on": ["r1"]},
    ],
}, ensure_ascii=False)


class QueueLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, prompt, system_prompt=""):
        return self.replies.pop(0) if self.replies else "{}"


class FakeRegistry:
    """按工具名 + 参数返回固定结果；Q3 明细故意失败"""

    def list_tools(self):
        return [
            types.SimpleNamespace(name="data_query", description="查询业务数据"),
            types.SimpleNamespace(name="report_write", description="生成报告"),
        ]

    def execute_tool(self, name, params=None, **kw):
        params = params or {}
        if name == "data_query" and params.get("quarter") == "Q3":
            return ToolResult(success=False, result="", error="无权限访问 Q3 销售明细")
        return ToolResult(success=True, result=f"{name} 结果: {params}", error="", duration_ms=12)


agent = ReActAgent(
    llm_client=QueueLLM([]),
    tool_registry=FakeRegistry(),
    planner=TaskPlanner(llm=QueueLLM([PLAN_V1, PLAN_V2]), enabled=True, replan_enabled=True, max_replans=1),
)

state = AgentState(session_id="s-demo", user_id="U007", user_query=QUERY, max_iterations=10)

print("\n" + "=" * 20 + " 1. 要不要规划 " + "=" * 20)
print("should_plan:", agent.should_plan(state))

print("\n" + "=" * 20 + " 2. plan_step：只生成，不执行 " + "=" * 20)
agent.plan_step(state)
print("sub_tasks:", [(s["id"], s["status"]) for s in state.sub_tasks])
print("current_task_index:", state.current_task_index)

print("\n" + "=" * 20 + " 3. 注入 prompt 的计划块 " + "=" * 20)
print(agent._plan_context(state))

steps = [
    ("data_query", {"quarter": "Q4"}),
    ("data_query", {"quarter": "Q3"}),   # 失败 → 触发重排
    ("data_query", {"quarter": "Q3汇总表"}),
    ("report_write", {"title": "Q4 vs Q3 对比"}),
]

for i, (name, args) in enumerate(steps, 1):
    print("\n" + "=" * 18 + f" 4.{i} act_step: {name}({args}) " + "=" * 18)
    state.action = {"name": name, "arguments": args}
    agent.act_step(state)
    print("--- 推进后 ---")
    print(agent._plan_context(state))
    print("task_results:", [(t["id"], t["status"]) for t in state.task_results])

print("\n" + "=" * 20 + " 5. 关键结论 " + "=" * 20)
print("指针 current_task_index:", state.current_task_index)
print("重排次数:", state.plan_replan_count)
