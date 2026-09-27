"""
工具 Harness 子包
================
把「工具能干什么、谁批准、怎么隔离、花多少预算、结果怎么回灌」管起来的一层。

【一句话】
    Agent 只负责想"调哪个工具"，Harness 负责"让不让调、在哪调、调完怎么处理"。

【模块清单】
    models.py        能力声明（风险等级/副作用/沙箱/权限）与策略裁决
    budget.py        执行预算：调用次数 / token / 墙钟超时
    middleware.py    调用前中间件：参数 / 权限 / 业务范围 / 限流 / 预算
    sandbox.py       沙箱执行：none / subprocess / docker
    sandbox_worker.py  子进程（容器）里真正跑工具的那一段
    result_filter.py 结果过滤：裁剪 + 脱敏
    harness.py       总装 ToolHarness + build_tool_harness()

【接入方式】
    from modules.harness import build_tool_harness
    registry = ToolRegistry(harness=build_tool_harness())
    registry.harness.bind_registry(registry)     # 让 schema 校验能取到 JSON Schema
"""
# 【什么】子包出口：常用符号一次导出
# 【作用】上层写一行 `from modules.harness import ToolHarness, build_tool_harness` 即可
from modules.harness.models import (
    ACTION_ALLOW, ACTION_APPROVE, ACTION_DENY, RiskLevel, ToolCallRequest, ToolCapability,
    ToolDecision, ToolPolicy, ToolVerdict,
)
from modules.harness.budget import BudgetManager, BudgetSpec, RunBudget
from modules.harness.middleware import (
    BudgetMiddleware, MiddlewareChain, PermissionMiddleware, RateLimitMiddleware,
    SchemaMiddleware, ScopeMiddleware, ToolMiddleware,
)
from modules.harness.sandbox import (
    MODE_DOCKER, MODE_NONE, MODE_REMOTE, MODE_SUBPROCESS, DockerRunner, InProcessRunner,
    RemoteSandboxRunner, SandboxRunner, SubprocessRunner, build_sandbox,
)
from modules.harness.result_filter import ResultFilter, ResultFilterConfig
from modules.harness.harness import ToolHarness, build_tool_harness

__all__ = [
    "ACTION_ALLOW", "ACTION_APPROVE", "ACTION_DENY", "RiskLevel",
    "ToolCallRequest", "ToolCapability", "ToolDecision", "ToolPolicy", "ToolVerdict",
    "BudgetManager", "BudgetSpec", "RunBudget",
    "ToolMiddleware", "MiddlewareChain", "SchemaMiddleware", "PermissionMiddleware",
    "ScopeMiddleware", "RateLimitMiddleware", "BudgetMiddleware",
    "MODE_NONE", "MODE_SUBPROCESS", "MODE_DOCKER", "MODE_REMOTE", "SandboxRunner", "InProcessRunner",
    "SubprocessRunner", "DockerRunner", "RemoteSandboxRunner", "build_sandbox",
    "ResultFilter", "ResultFilterConfig",
    "ToolHarness", "build_tool_harness",
]
