"""
工具 Harness 演示（离线可跑，不依赖 LLM / MySQL / Milvus）
==========================================================
一次跑完五件事：
    ① 工具注册与能力声明（风险等级 / 副作用 / 沙箱 / 权限 / 业务范围）
    ② 调用前中间件校验（参数 / 权限 / 业务范围 / 限流 / 预算）
    ③ 沙箱隔离（none / subprocess / docker / remote）
    ④ 执行预算与循环控制（次数 / token / 超时）
    ⑤ 结果过滤 + 审计日志（裁剪 / 脱敏 / 完整留痕）

运行：
    python examples/tool_harness_demo.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.harness import (
    BudgetManager, BudgetSpec, MiddlewareChain, RiskLevel, ToolCallRequest, ToolCapability,
    ToolHarness, ToolPolicy,
)
from modules.harness.middleware import (
    BudgetMiddleware, PermissionMiddleware, RateLimitMiddleware, SchemaMiddleware, ScopeMiddleware,
)
from modules.harness.result_filter import ResultFilter, ResultFilterConfig
from modules.harness.sandbox import InProcessRunner, RemoteSandboxRunner, SubprocessRunner, build_sandbox
from modules.tools.base_tool import BaseTool, ToolResult
from modules.tools.calculator import CalculatorTool
from modules.tools.tool_registry import ToolRegistry


# ----------------------------------------------------------------------
# 演示用工具：一个只读、一个高风险（发通知）、一个禁止级（删表）
# ----------------------------------------------------------------------
class NoticeTool(BaseTool):
    capability = {"risk": "high", "side_effect": True, "description": "对外发通知，有副作用"}

    def __init__(self):
        super().__init__(
            name="send_notice", description="向外部发通知",
            parameters={"properties": {"to": {"type": "string"}, "text": {"type": "string"}},
                        "required": ["to"]},
        )

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, result=f"已发送给 {kwargs.get('to')}")


class DropTableTool(BaseTool):
    capability = {"risk": "critical", "side_effect": True, "sandbox": "subprocess"}

    def __init__(self):
        super().__init__(
            name="drop_table", description="删除数据表（演示，不真删）",
            parameters={"properties": {"table": {"type": "string"}}, "required": ["table"]},
        )

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, result="dropped")


class _Principal:
    """最小身份替身（真实场景由 RBAC 解析 API Key 得到）。"""

    def __init__(self, permissions, dept_code="D_FIN", user_code="U007", mask_level=0, scope="dept"):
        self.permissions = set(permissions)
        self.dept_code = dept_code
        self.user_code = user_code
        self.mask_level = mask_level
        self._scope = scope

    def can(self, perm: str) -> bool:
        if not perm or not self.permissions:
            return False
        if perm in self.permissions:
            return True
        return any(g.endswith("*") and perm.startswith(g[:-1]) for g in self.permissions)

    def scope_for(self, data_type: str) -> str:
        return self._scope


def line(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def main() -> None:
    # ------------------------------------------------------------------
    # ① 工具注册与能力声明
    # ------------------------------------------------------------------
    line("① 工具注册与能力声明：高风险禁止执行 / 需人工审批")

    # 捕获审计写入（真实环境落 MySQL sys_audit_log，这里收集起来展示）
    audit_rows = []
    try:
        from modules.enterprise import audit as _audit_mod
        def _collect(*a, **kw):
            row = {"action": a[1] if len(a) > 1 else "", "resource": a[2] if len(a) > 2 else "",
                   "result": a[3] if len(a) > 3 else ""}
            row.update(kw)
            audit_rows.append(row)
        _audit_mod.log_audit = _collect
    except Exception:
        pass

    policy = ToolPolicy(
        deny_levels=("critical",),          # critical：一律禁止
        approve_levels=("high",),           # high：需人工批准
        deny_tools=("drop_*",),             # 名字黑名单，优先于等级
        on_no_gate="deny",                  # 需审批但没闸门 → 拒绝（安全优先）
    )
    spec = BudgetSpec(max_calls_per_run=6, max_calls_per_tool=3, token_budget=4000, timeout_seconds=60.0)
    budgets = BudgetManager(spec)

    harness = ToolHarness(
        policy=policy, budgets=budgets,
        result_filter=ResultFilter(ResultFilterConfig(max_result_chars=120)),
        default_sandbox="none", default_timeout=15.0,
    )
    registry = ToolRegistry(harness=harness)
    registry.register_batch([CalculatorTool(), NoticeTool(), DropTableTool()])
    harness.bind_registry(registry)
    harness.chain = MiddlewareChain([
        SchemaMiddleware(lambda n: getattr(registry.get_tool(n), "parameters_schema", {}) or {}),
        PermissionMiddleware(),
        ScopeMiddleware(),
        RateLimitMiddleware(capacity=20, refill_per_sec=5.0, per_user_capacity=8, per_user_refill=2.0),
        BudgetMiddleware(budgets),
    ])

    print("已登记能力：")
    for name, cap in harness.capabilities().items():
        print(f"  {name:14s} risk={cap['risk']:8s} 需审批={str(cap['needs_approval']):5s} "
              f"沙箱={cap['sandbox']:10s} 副作用={cap['side_effect']}")
    print("\n策略裁决：")
    for name in ("calculator", "send_notice", "drop_table"):
        cap = harness.capability_of(registry.get_tool(name))
        print(f"  {name:14s} → {harness.policy.decide(cap).action:8s} {harness.policy.decide(cap).reason}")

    principal = _Principal(permissions=("tool:calculator", "tool:send_notice"))
    harness.begin_run("demo-session")

    # ------------------------------------------------------------------
    # ② 调用前中间件校验
    # ------------------------------------------------------------------
    line("② 调用前中间件：模型给错了参数，真实服务收不到请求")
    cases = [
        ("幻觉参数（多出 expr）", "calculator", {"expression": "3*7", "expr": "3*7"}),
        ("必填缺失", "calculator", {}),
        ("越权工具（身份无权限）", "drop_table", {"table": "orders"}),
        ("正常调用", "calculator", {"expression": "3*7"}),
    ]
    for title, name, args in cases:
        req = ToolCallRequest(tool_name=name, arguments=args, session_id="demo-session",
                              principal=principal, capability=harness.capability_of(registry.get_tool(name)))
        verdict = harness.before_call(req)
        flag = "放行" if verdict.allowed else f"拦截[{verdict.code}]"
        print(f"  {title:22s} → {flag}  {verdict.reason}")

    print("\n业务范围（财务身份查别的部门）：")
    scope_chain = MiddlewareChain([ScopeMiddleware()])
    req = ToolCallRequest(tool_name="data_query", arguments={"data_type": "orders", "dept_code": "D_HR"},
                          capability=ToolCapability(name="data_query", data_scopes=("orders",)),
                          principal=_Principal(permissions=("tool:data_query",)))
    print(f"  {'越界参数 dept_code=D_HR':22s} → {scope_chain.run(req).reason}")

    # ------------------------------------------------------------------
    # ③ 沙箱隔离
    # ------------------------------------------------------------------
    line("③ 沙箱隔离：工具逻辑出错，影响锁在沙箱里")
    print(f"  进程内   : {InProcessRunner().run(CalculatorTool(), {'expression': '21*2'})}")
    print(f"  子进程   : {SubprocessRunner(timeout_seconds=30.0).run(CalculatorTool(), {'expression': '21*2'})}")
    bad = SubprocessRunner(timeout_seconds=30.0).run(CalculatorTool(), {"expression": "__import__('os')"})
    print(f"  子进程失败: success={bad.success} error={bad.error[:60]}（主进程未受影响）")
    print(f"  按模式构造: none→{type(build_sandbox('none')).__name__}, "
          f"subprocess→{type(build_sandbox('subprocess')).__name__}")
    # 远程沙箱：配了 TOOL_SANDBOX_REMOTE_URL 才真连；没配就提示怎么起模拟服务
    #   （连不上会自动降级本机子进程，所以演示不会卡死）
    remote_url = str(os.getenv("TOOL_SANDBOX_REMOTE_URL", "") or "").strip()
    if remote_url:
        remote = RemoteSandboxRunner(endpoint=remote_url, timeout_seconds=30.0)
        print(f"  远程     : 健康检查={remote.available()} "
              f"{remote.run(CalculatorTool(), {'expression': '21*2'})}")
    else:
        print("  远程     : 未配置 TOOL_SANDBOX_REMOTE_URL，跳过"
              "（起模拟服务：uvicorn modules.harness.remote_sandbox_server:app --port 8899）")

    # ------------------------------------------------------------------
    # ④ 执行预算与循环控制
    # ------------------------------------------------------------------
    line("④ 执行预算：次数 / token / 超时，任一打满即停")
    budgets.begin("budget-demo", BudgetSpec(max_calls_per_run=3, max_calls_per_tool=2, timeout_seconds=60.0))
    for i in range(1, 5):
        req = ToolCallRequest(tool_name="calculator", arguments={"expression": f"{i}+{i}"},
                              session_id="budget-demo", principal=principal,
                              capability=harness.capability_of(registry.get_tool("calculator")))
        verdict = harness.before_call(req)
        if verdict.allowed:
            budgets.get("budget-demo").consume("calculator", 10)
        print(f"  第 {i} 次调用 → {'放行' if verdict.allowed else '拦截: ' + verdict.reason}")
    print(f"  账本快照: {json.dumps(budgets.get('budget-demo').snapshot(), ensure_ascii=False)}")

    # ------------------------------------------------------------------
    # ⑤ 结果过滤 + 审计
    # ------------------------------------------------------------------
    line("⑤ 结果过滤 + 审计日志")
    long_result = ToolResult(success=True, result="差旅报销标准：" + "一线城市每晚不超过600元。" * 20)
    filtered = harness.result_filter.filter(long_result, principal=None)
    print(f"  裁剪前 {len(str(long_result.result))} 字符 → 裁剪后 {len(str(filtered.result))} 字符")
    print(f"  文本尾部: ...{str(filtered.result)[-28:]}")
    masked = harness.result_filter.filter(
        ToolResult(success=True, result={"phone": "13812345678", "name": "张三"}),
        principal=_Principal(permissions=(), mask_level=2))
    print(f"  脱敏后  : {masked.result}")

    print("\n  审计记录（真实环境落 MySQL sys_audit_log）：")
    req = ToolCallRequest(tool_name="drop_table", arguments={"table": "orders"}, session_id="demo-session",
                          principal=principal, capability=harness.capability_of(registry.get_tool("drop_table")))
    harness.record_intercepted(req, req.capability, "工具 drop_table 风险等级为 critical，已配置为禁止执行",
                               code="policy_denied")
    for row in audit_rows:
        print(f"    action={row.get('action')} resource={row.get('resource')} result={row.get('result')}")
        print(f"    detail={json.dumps(row.get('detail', {}), ensure_ascii=False)[:160]}...")
    if not audit_rows:
        print("    （审计模块未装配，真实环境会写入 sys_audit_log；"
              "拦截/放行/失败三种状态分别记为 deny / allow / error）")

    line("演示结束")
    print("提示：TOOL_HARNESS_ENABLED=false 可整体关闭，行为回到接入前。")


if __name__ == "__main__":
    main()
