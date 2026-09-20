"""多 Agent 子包：数据契约（models）+ Agent 实现（agents）+ 运行时（harness）。"""

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
from modules.multiagent.agents import (
    BaseAgent,
    SupervisorAgent,
    WorkerAgent,
    ReviewerAgent,
    RunContext,
    build_agent,
    build_default_team,
)
from modules.multiagent.harness import AgentHarness, build_harness
from modules.multiagent.crewai_backend import CrewAIBackend

__all__ = [
    "AgentRole", "AgentSpec", "SubTask", "PlanResult", "AgentOutput", "TraceEvent",
    "RunResult", "RunStatus", "TeamConfig", "EvalCase", "EvalResult", "EvalReport",
    "json_schema_hint",
    "BaseAgent", "SupervisorAgent", "WorkerAgent", "ReviewerAgent", "RunContext",
    "build_agent", "build_default_team",
    "AgentHarness", "build_harness", "CrewAIBackend",
]
