"""多 Agent 子包：数据契约（models）+ Agent 实现（agents）+ 运行时（harness）。"""

# 【什么】从 models 导入全部数据契约（角色/任务/产出/结果/评测）
# 【作用】外部只 import 本包即可拿到类型，不必记住具体模块路径
from modules.multiagent.models import (
    AgentRole,
    AgentSpec,
    SubTask,
    PlanResult,
    AgentOutput,
    TraceEvent,
    RunResult,
    RunStatus,
    TeamConfig,
    EvalCase,
    EvalResult,
    EvalReport,
    json_schema_hint,
)
# 【什么】从 agents 导入 Agent 基类、三类角色实现与构造工厂
# 【作用】建团队只需 build_default_team / build_agent 两个入口
from modules.multiagent.agents import (
    BaseAgent,
    SupervisorAgent,
    WorkerAgent,
    ReviewerAgent,
    RunContext,
    build_agent,
    build_default_team,
)
# 【什么】从 harness 导入运行时与一行构造工厂
# 【作用】API / demo 直接 build_harness() 就能跑一次多 Agent 协作
from modules.multiagent.harness import AgentHarness, build_harness
# 【什么】导入 crewai 编排后端（可选依赖，默认不启用）
# 【作用】与自研 harness 二选一，两者对外都返回同一个 RunResult
from modules.multiagent.crewai_backend import CrewAIBackend

# 【什么】对外可见的名字清单
# 【作用】from modules.multiagent import * 时只导出这些，不泄露内部符号
__all__ = [
    "AgentRole", "AgentSpec", "SubTask", "PlanResult", "AgentOutput", "TraceEvent",
    "RunResult", "RunStatus", "TeamConfig", "EvalCase", "EvalResult", "EvalReport",
    "json_schema_hint",
    "BaseAgent", "SupervisorAgent", "WorkerAgent", "ReviewerAgent", "RunContext",
    "build_agent", "build_default_team",
    "AgentHarness", "build_harness", "CrewAIBackend",
]
