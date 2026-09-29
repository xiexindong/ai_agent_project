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
    AgentMessage,
    ConsensusResult,
    MessageKind,
    DelegationRecord,
    DelegationDecision,
    json_schema_hint,
)
# 【什么】从 coordination 导入三个协调组件与相似度工具
# 【作用】消息总线（横向对话）、一致性判定（谁和谁打架）、
#        仲裁者与重规划器（打架怎么办、没干成怎么办）
from modules.multiagent.coordination import (
    Arbitrator,
    Consensus,
    MessageBus,
    ReplanController,
    text_overlap,
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
# 【什么】从 delegation 导入委派的三个组件
# 【作用】策略（四条边界）、台账（记账 + 防环）、执行器（真跑一次并回传）
from modules.multiagent.delegation import (
    DelegationExecutor,
    DelegationLedger,
    DelegationPolicy,
    decide_delegation,
)

# ---------------------------------------------------------------------------
# 【为什么 harness 与 crewai 后端改成按需加载】
#   harness 依赖 modules.orchestration（中心化编排层），
#   而 orchestration.protocol 反过来又依赖本包的 models（Envelope 继承 AgentMessage）。
#   如果在包初始化时就 import harness，会形成一条环：
#       protocol → multiagent/__init__ → harness → orchestration.bus → protocol（还没定义完）
#   结果是"先 import 谁"决定成败——先 import multiagent 能跑，
#   先 import orchestration 就报 ImportError，属于最难查的那类问题。
#   改成按需加载后：包初始化只依赖 models / coordination / agents，
#   AgentHarness 在真正被用到时才加载，那时两边都已就绪，两个方向都安全。
# ---------------------------------------------------------------------------
_LAZY_MODULES = {
    "AgentHarness": "modules.multiagent.harness",
    "build_harness": "modules.multiagent.harness",
    "CrewAIBackend": "modules.multiagent.crewai_backend",
}


def __getattr__(name: str):
    """PEP 562 模块级懒加载：from modules.multiagent import AgentHarness 依然可用。"""
    mod_path = _LAZY_MODULES.get(name)
    if mod_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(mod_path), name)


# 【什么】对外可见的名字清单
# 【作用】from modules.multiagent import * 时只导出这些，不泄露内部符号
__all__ = [
    "AgentRole", "AgentSpec", "SubTask", "PlanResult", "AgentOutput", "TraceEvent",
    "RunResult", "RunStatus", "TeamConfig", "EvalCase", "EvalResult", "EvalReport",
    "AgentMessage", "ConsensusResult", "MessageKind",
    "DelegationRecord", "DelegationDecision",
    "json_schema_hint",
    "DelegationPolicy", "DelegationLedger", "DelegationExecutor", "decide_delegation",
    "BaseAgent", "SupervisorAgent", "WorkerAgent", "ReviewerAgent", "RunContext",
    "build_agent", "build_default_team",
    "AgentHarness", "build_harness", "CrewAIBackend",
    "MessageBus", "Consensus", "Arbitrator", "ReplanController", "text_overlap",
]
