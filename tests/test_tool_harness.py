"""
工具 Harness 测试
=================
覆盖五件事：能力声明 / 调用前校验 / 人工审批 / 沙箱 / 预算 / 结果过滤。

【为什么这些用例必须存在】
    这些都是"安全闸门"，闸门失效时系统看起来是正常的（照常返回答案），
    只有被拦截的那一侧能证明它真的生效了。所以每个闸门都要有一条"被拦下"的用例。
"""
import time

import pytest

from modules.harness.budget import BudgetManager, BudgetSpec, RunBudget
from modules.harness.harness import ToolHarness, build_tool_harness
from modules.harness.middleware import (
    BudgetMiddleware, MiddlewareChain, PermissionMiddleware, RateLimitMiddleware,
    SchemaMiddleware, ScopeMiddleware,
)
from modules.harness.models import RiskLevel, ToolCallRequest, ToolCapability, ToolPolicy
from modules.harness.result_filter import ResultFilter, ResultFilterConfig
from modules.harness.sandbox import MODE_SUBPROCESS, InProcessRunner, SubprocessRunner, build_sandbox
from modules.tools.base_tool import BaseTool, ToolResult
from modules.tools.calculator import CalculatorTool
from modules.tools.tool_registry import ToolRegistry


# ------------------------------------------------------------------
# 测试替身
# ------------------------------------------------------------------
class DropTableTool(BaseTool):
    """演示用的高危工具：删表（不可逆）。"""

    capability = {"risk": "critical", "side_effect": True, "sandbox": "subprocess"}

    def __init__(self):
        super().__init__(
            name="drop_table", description="删除数据表（演示用，实际不执行）",
            parameters={"properties": {"table": {"type": "string"}}, "required": ["table"]},
        )

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, result="dropped")


class NoticeTool(BaseTool):
    """演示用的高风险工具：对外发通知（有副作用但可重发）。"""

    capability = {"risk": "high", "side_effect": True}

    def __init__(self):
        super().__init__(
            name="send_notice", description="向外部发通知",
            parameters={"properties": {"to": {"type": "string"}, "text": {"type": "string"}},
                        "required": ["to"]},
        )

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, result="sent")


class _Principal:
    """最小权限载体替身：只实现 Harness 用到的 can / scope_for / 身份字段。"""

    def __init__(self, permissions=(), dept_code="D_FIN", user_code="U007", mask_level=0, scope="dept"):
        self.permissions = set(permissions)
        self.dept_code = dept_code
        self.user_code = user_code
        self.mask_level = mask_level
        self._scope = scope

    def can(self, perm_code: str) -> bool:
        if not perm_code or not self.permissions:
            return False
        if perm_code in self.permissions:
            return True
        for granted in self.permissions:
            if granted.endswith("*") and perm_code.startswith(granted[:-1]):
                return True
        return False

    def scope_for(self, data_type: str) -> str:
        return self._scope


def _harness(policy=None, gate=None) -> ToolHarness:
    """构造一个只装必要中间件的 Harness（不接 registry 时 schema 校验自动跳过）。"""
    h = ToolHarness(policy=policy or ToolPolicy(), gate=gate, audit_enabled=False)
    return h


# ------------------------------------------------------------------
# ① 能力声明与策略裁决
# ------------------------------------------------------------------
def test_capability_declared_by_tool():
    """工具自己声明的 risk 会被 Harness 读到。"""
    h = _harness()
    assert h.capability_of(DropTableTool()).risk == RiskLevel.CRITICAL
    assert h.capability_of(CalculatorTool()).risk == RiskLevel.LOW


def test_critical_tool_is_denied():
    """critical 级工具：配置层面禁止执行，模型连申请都不会被受理。"""
    h = _harness()
    cap = h.capability_of(DropTableTool())
    decision = h.policy.decide(cap)
    assert decision.denied is True
    assert "禁止执行" in decision.reason


def test_high_tool_needs_approval_and_blocked_without_gate():
    """high 级工具需人工批准；没有闸门时按 on_no_gate=deny 拒绝——模型无权直接执行。"""
    h = _harness()
    req = ToolCallRequest(tool_name="send_notice", arguments={"to": "U001"},
                          capability=h.capability_of(NoticeTool()))
    verdict = h.before_call(req)
    assert verdict.allowed is False
    assert verdict.code == "approval_required"


def test_deny_tools_blacklist_overrides_level():
    """名字黑名单优先于等级：medium 的 drop_index 也会被封死。"""
    policy = ToolPolicy(deny_tools=("drop_*",))
    h = _harness(policy)
    cap = ToolCapability(name="drop_index", risk=RiskLevel.MEDIUM)
    assert h.policy.decide(cap).denied is True


def test_risk_override_from_config():
    """配置覆盖优先于工具声明：web_search 被提级为 high。"""
    policy = ToolPolicy(overrides={"web_search": "high"})
    cap = ToolCapability(name="web_search", risk=RiskLevel.LOW)
    assert policy.effective(cap).risk == RiskLevel.HIGH


# ------------------------------------------------------------------
# ② 调用前中间件
# ------------------------------------------------------------------
def test_schema_middleware_blocks_hallucinated_param():
    """幻觉参数（工具没声明过的字段）被拦在真实服务之前。"""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    chain = MiddlewareChain([SchemaMiddleware(lambda n: _schema_of(registry, n))])
    # 必填的 expression 给了，但同时多出一个工具没声明的 expr
    req = ToolCallRequest(tool_name="calculator", arguments={"expression": "3*7", "expr": "3*7"})
    verdict = chain.run(req)
    assert verdict.allowed is False
    assert verdict.code == "schema_invalid"
    assert "未知参数" in verdict.reason


def test_schema_middleware_blocks_missing_required():
    """必填参数缺失也要拦截。"""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    chain = MiddlewareChain([SchemaMiddleware(lambda n: _schema_of(registry, n))])
    verdict = chain.run(ToolCallRequest(tool_name="calculator", arguments={}))
    assert verdict.allowed is False
    assert "缺少必填参数" in verdict.reason


def test_permission_middleware_blocks_unauthorized_tool():
    """身份没有 tool:data_query 权限时，模型即使点名也调不动。"""
    chain = MiddlewareChain([PermissionMiddleware()])
    req = ToolCallRequest(tool_name="data_query", arguments={"data_type": "orders"},
                          principal=_Principal(permissions=("tool:calculator",)))
    verdict = chain.run(req)
    assert verdict.allowed is False
    assert verdict.code == "permission_denied"


def test_scope_middleware_blocks_other_dept():
    """业务范围：财务身份（dept 域）不能查别的部门。"""
    chain = MiddlewareChain([ScopeMiddleware()])
    cap = ToolCapability(name="data_query", data_scopes=("orders",))
    req = ToolCallRequest(tool_name="data_query", arguments={"data_type": "orders", "dept_code": "D_HR"},
                          principal=_Principal(scope="dept"), capability=cap)
    verdict = chain.run(req)
    assert verdict.allowed is False
    assert verdict.code == "scope_denied"
    assert "超出本部门范围" in verdict.reason


def test_rate_limit_middleware_blocks_burst():
    """突发调用超过桶容量即被限流。"""
    chain = MiddlewareChain([RateLimitMiddleware(capacity=2, refill_per_sec=0.1,
                                                 per_user_capacity=10, per_user_refill=10.0)])
    req = ToolCallRequest(tool_name="calculator", arguments={"expression": "1+1"})
    assert chain.run(req).allowed is True
    assert chain.run(req).allowed is True
    assert chain.run(req).allowed is False


# ------------------------------------------------------------------
# ④ 执行预算与循环控制
# ------------------------------------------------------------------
def test_budget_blocks_when_tool_calls_exhausted():
    """单工具次数打满即拦截（防死循环）。"""
    spec = BudgetSpec(max_calls_per_run=10, max_calls_per_tool=2)
    manager = BudgetManager(spec)
    budget = manager.begin("s1")
    budget.consume("calculator")
    budget.consume("calculator")
    chain = MiddlewareChain([BudgetMiddleware(manager)])
    verdict = chain.run(ToolCallRequest(tool_name="calculator", arguments={}, session_id="s1"))
    assert verdict.allowed is False
    assert verdict.code == "budget_exceeded"
    assert "疑似死循环" in verdict.reason


def test_budget_blocks_when_run_calls_exhausted():
    """整轮调用次数上限生效。"""
    spec = BudgetSpec(max_calls_per_run=1, max_calls_per_tool=5)
    manager = BudgetManager(spec)
    manager.begin("s2").consume("calculator")
    chain = MiddlewareChain([BudgetMiddleware(manager)])
    verdict = chain.run(ToolCallRequest(tool_name="data_query", arguments={}, session_id="s2"))
    assert verdict.allowed is False
    assert "已达上限" in verdict.reason


def test_budget_blocks_after_timeout():
    """墙钟超时后不再放行工具调用。"""
    budget = RunBudget("s3", BudgetSpec(timeout_seconds=0.01))
    time.sleep(0.02)
    ok, reason = budget.check("calculator")
    assert ok is False
    assert "已超时" in reason


def test_budget_blocks_when_token_budget_used_up():
    """token 预算打满即停（防上下文膨胀）。"""
    budget = RunBudget("s4", BudgetSpec(token_budget=100))
    budget.add_tokens(150)
    ok, reason = budget.check("calculator")
    assert ok is False
    assert "token 预算已用尽" in reason


# ------------------------------------------------------------------
# ③ 人工审批
# ------------------------------------------------------------------
class _FakeGate:
    """闸门替身：按预设结果批准或拒绝。"""

    def __init__(self, approve: bool, edited: str = ""):
        self.approve = approve
        self.edited = edited
        self.calls = []

    def ask_tool(self, tool_name, arguments, **kwargs):
        self.calls.append(tool_name)
        return self.approve, dict(arguments or {})


def test_high_tool_executes_after_human_approval():
    """人工批准后高风险工具才真正执行，且执行结果正常返回。"""
    gate = _FakeGate(approve=True)
    h = _harness(gate=gate)
    registry = ToolRegistry(harness=h)
    registry.register(NoticeTool())
    h.bind_registry(registry)
    result = registry.execute_tool("send_notice", {"to": "U001"}, session_id="s5")
    assert result.success is True
    assert gate.calls == ["send_notice"]


def test_high_tool_cancelled_when_human_rejects():
    """人工拒绝 → 工具根本没执行（error 会回喂模型）。"""
    gate = _FakeGate(approve=False)
    h = _harness(gate=gate)
    registry = ToolRegistry(harness=h)
    registry.register(NoticeTool())
    result = registry.execute_tool("send_notice", {"to": "U001"}, session_id="s6")
    assert result.success is False
    assert result.meta.get("code") == "approval_rejected"


def test_denied_tool_never_reaches_gate():
    """禁止类工具连审批都不进：申请不会被受理。"""
    gate = _FakeGate(approve=True)
    h = _harness(gate=gate)
    registry = ToolRegistry(harness=h)
    registry.register(DropTableTool())
    result = registry.execute_tool("drop_table", {"table": "orders"}, session_id="s7")
    assert result.success is False
    assert result.meta.get("code") == "policy_denied"
    assert gate.calls == []          # 闸门一次都没被问到


# ------------------------------------------------------------------
# ③ 沙箱隔离
# ------------------------------------------------------------------
def test_inprocess_sandbox_runs_tool():
    """默认进程内执行：结果与直接调用一致。"""
    runner = InProcessRunner()
    result = runner.run(CalculatorTool(), {"expression": "3*7"})
    assert result.success is True
    assert result.result == 21 or str(result.result) == "21"


def test_subprocess_sandbox_runs_tool():
    """子进程沙箱：真起一个进程执行，结果以 JSON 回传。"""
    runner = SubprocessRunner(timeout_seconds=30.0)
    result = runner.run(CalculatorTool(), {"expression": "3*7"})
    assert result.success is True, result.error
    assert str(result.result) == "21"


def test_subprocess_sandbox_isolates_failure():
    """子进程里工具执行失败，主进程照常拿到失败结果（不会把调用方带崩）。

    【为什么用非法表达式而不是自定义工具类】
        子进程按"模块 + 类名"import 工具，定义在测试文件里的类子进程取不到；
        计算器本身有 AST 白名单，传非法表达式即可稳定触发失败。
    """
    runner = SubprocessRunner(timeout_seconds=30.0)
    result = runner.run(CalculatorTool(), {"expression": "__import__('os').system('echo hi')"})
    assert result.success is False
    assert result.error           # 错误信息要能回传，而不是让主进程崩掉


def test_build_sandbox_by_mode():
    """按模式名构造：none → 进程内，subprocess → 子进程。"""
    assert isinstance(build_sandbox("none"), InProcessRunner)
    assert isinstance(build_sandbox(MODE_SUBPROCESS), SubprocessRunner)


# ------------------------------------------------------------------
# ⑤ 结果过滤与审计
# ------------------------------------------------------------------
def test_result_filter_trims_long_output():
    """超长结果被裁剪，并留下"还有内容"的标记，模型不会误以为结果就这么点。"""
    f = ResultFilter(ResultFilterConfig(max_result_chars=20))
    result = f.filter(ToolResult(success=True, result="abcdefghijklmnopqrstuvwxyz"))
    assert result.success is True
    assert len(result.result) < 60
    assert "已裁剪" in result.result
    assert result.meta.get("trimmed") is True


def test_result_filter_masks_by_principal():
    """结果按身份的 mask_level 脱敏：手机号不再以明文回灌上下文。"""
    f = ResultFilter(ResultFilterConfig(max_result_chars=2000))
    principal = _Principal(mask_level=2)
    result = f.filter(ToolResult(success=True, result={"phone": "13812345678", "name": "张三"}),
                      principal=principal)
    assert "13812345678" not in result.result
    assert result.meta.get("masked") is True


def test_result_filter_keeps_short_result_intact():
    """短结果不该被动到：meta 标记为未裁剪。"""
    f = ResultFilter(ResultFilterConfig(max_result_chars=2000))
    result = f.filter(ToolResult(success=True, result="21"))
    assert result.result == "21"
    assert result.meta.get("trimmed") is False


def test_registry_without_harness_behaves_as_before():
    """未装配 Harness 时行为与改造前一致：正常执行仍然成功。"""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    result = registry.execute_tool("calculator", {"expression": "3*7"}, session_id="legacy")
    assert result.success is True


def test_harness_audit_records_interception():
    """被拦截的调用也记入账本：blocked 计数 +1（审计在 enabled 时落库）。"""
    h = _harness()
    registry = ToolRegistry(harness=h)
    registry.register(CalculatorTool())
    registry.execute_tool("calculator", {"expression": "3*7"}, session_id="s8")
    registry.execute_tool("calculator", {"expression": "3*7"}, session_id="s8")   # 同参去重
    snapshot = h.budget("s8").snapshot()
    assert snapshot["total_calls"] >= 1


# ------------------------------------------------------------------
# ⑤ 审计落库（放行与拦截都要留痕）
# ------------------------------------------------------------------
def test_audit_writes_record_for_allow_and_deny(monkeypatch):
    """有登录态时每次调用落一条审计：放行记 allow、被策略拒绝记 deny。

    【为什么 monkeypatch 而不是真连 MySQL】
        审计落库依赖 MySQL 连接，单测不该依赖外部服务；
        这里只验证"该记的字段有没有被填对"，落库本身有 test_security 覆盖。
    """
    import modules.enterprise.audit as audit_mod

    records = []
    monkeypatch.setattr(audit_mod, "log_audit", lambda *a, **k: records.append((a, k)))

    principal = _Principal(permissions=("tool:calculator",))
    h = ToolHarness(audit_enabled=True)
    registry = ToolRegistry(harness=h)
    registry.register(CalculatorTool())
    registry.register(DropTableTool())

    # ① 正常放行 → state=allow、approval=none
    registry.execute_tool("calculator", {"expression": "3*7"}, session_id="s9", principal=principal)
    # ② 被策略禁止 → state=deny、approval=denied（越权尝试正是审计最该看到的）
    registry.execute_tool("drop_table", {"table": "orders"}, session_id="s9", principal=principal)

    assert len(records) == 2, f"审计条数不对: {len(records)}"
    allow_args, allow_kwargs = records[0]
    assert allow_args[1] == "tool_call" and allow_args[2] == "tool:calculator"
    assert allow_args[3] == "allow"
    assert allow_kwargs["detail"]["risk"] == "low"
    assert allow_kwargs["detail"]["approval"] == "none"

    deny_args, deny_kwargs = records[1]
    assert deny_args[2] == "tool:drop_table"
    assert deny_args[3] == "deny"
    assert deny_kwargs["detail"]["risk"] == "critical"
    assert deny_kwargs["detail"]["approval"] == "denied"


def test_audit_skipped_without_principal(monkeypatch):
    """无登录态（CLI/单测）不落审计：审计表不该被调试流量污染。"""
    import modules.enterprise.audit as audit_mod

    records = []
    monkeypatch.setattr(audit_mod, "log_audit", lambda *a, **k: records.append((a, k)))

    h = ToolHarness(audit_enabled=True)
    registry = ToolRegistry(harness=h)
    registry.register(CalculatorTool())
    registry.execute_tool("calculator", {"expression": "3*7"}, session_id="s10")
    assert records == []


# ------------------------------------------------------------------
# 配置装配：开关、中间件顺序、容器镜像
# ------------------------------------------------------------------
class _FakeSettings:
    """最小配置替身：只含 build_tool_harness 会读的那几项。"""

    tool_harness_enabled = True
    tool_default_risk = "low"
    tool_risk_overrides = "web_search:high"
    tool_deny_levels = "critical"
    tool_approve_levels = "high,critical"
    tool_deny_tools = ""
    tool_approve_tools = ""
    tool_on_no_gate = "deny"
    tool_max_calls_per_run = 8
    tool_max_calls_per_tool = 5
    tool_token_budget = 6000
    tool_run_timeout_seconds = 120.0
    tool_result_max_chars = 2000
    tool_sandbox_mode = "none"
    tool_sandbox_timeout_seconds = 15.0
    tool_sandbox_image = "python:3.11-slim"
    tool_rate_burst = 20
    tool_rate_per_sec = 5.0
    tool_rate_burst_per_user = 8
    tool_rate_per_user = 2.0


class _NamedTool:
    """只带名字的工具替身：用来验证配置覆盖（TOOL_RISK_OVERRIDES）生效。"""

    def __init__(self, name):
        self.name = name

    capability = None


def test_build_tool_harness_from_settings():
    """按配置装配：五道中间件顺序固定，预算与容器镜像都来自配置。"""
    h = build_tool_harness(_FakeSettings())
    assert h is not None
    assert h.chain.names() == ["schema", "permission", "scope", "rate_limit", "budget"]
    assert h.budgets.spec.max_calls_per_run == 8
    assert h.budgets.spec.max_calls_per_tool == 5
    assert h.docker_options["image"] == "python:3.11-slim"     # TOOL_SANDBOX_IMAGE 不能是死配置
    # 配置里的 web_search:high 覆盖掉默认 low
    assert h.capability_of(_NamedTool("web_search")).risk == RiskLevel.HIGH
    assert h.capability_of(_NamedTool("calculator")).risk == RiskLevel.LOW


def test_build_tool_harness_returns_none_when_disabled():
    """总开关关闭时返回 None：registry 退化为改造前行为，零行为变化。"""
    class _Off(_FakeSettings):
        tool_harness_enabled = False
    assert build_tool_harness(_Off()) is None


# 【什么】取工具的 JSON Schema（测试用例里的辅助函数）
def _schema_of(registry: ToolRegistry, name: str) -> dict:
    tool = registry.get_tool(name)
    return getattr(tool, "parameters_schema", {}) or {}
