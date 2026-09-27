"""
调用前中间件
============
模型给出 tool call 之后、真正下发之前，统一在这里过一遍。

【为什么要有这一层】
    模型的 tool call 是"生成"出来的，不是"校验"出来的：
        - 幻觉出不存在的参数名（把 expression 写成 expr）
        - 越权调一个它其实没权限的工具（权限过滤漏了 / 名字拼得差不多）
        - 用别人的部门码查数据（参数里带 dept_code=D_HR）
        - 短时间狂调同一个工具
    这些都不能等下发到真实服务再报错——那时副作用已经发生了。

【五道中间件，顺序即优先级】
    1. SchemaMiddleware      参数合法性（必填 / 未知参数 / 类型）
    2. PermissionMiddleware  用户权限（principal.can）
    3. ScopeMiddleware       业务范围（部门域 / 本人域）
    4. RateLimitMiddleware   调用限流（令牌桶）
    5. BudgetMiddleware      执行预算（次数 / token / 超时）

    顺序为什么是这样：先判"参数对不对"（最便宜、最常见），
    再判"你有没有资格"，最后才动用全局资源（配额与限流）。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from modules.common.logger import get_logger
from modules.harness.budget import BudgetManager, BudgetSpec
from modules.harness.models import ToolCallRequest, ToolVerdict

_logger = get_logger("harness.middleware")

# 服务端注入的私有参数前缀：这些不在 schema 里，模型给不出来，校验时跳过
_PRIVATE_PREFIX = "_"

# JSON Schema 类型 → Python 类型
_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


# 【什么】中间件基类：要么放行，要么给出拦截码与原因
# 【作用】新增一道校验只需实现 check()，不用改调用链
class ToolMiddleware:
    """调用前中间件基类。check 返回 None 表示放行，返回 (code, reason) 表示拦截。"""

    name = "base"

    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        return None


# 【什么】参数合法性校验
# 【作用】把幻觉参数挡在门外：缺失必填、多出未知参数、类型不对，一律拦截
class SchemaMiddleware(ToolMiddleware):
    """按工具的 JSON Schema 校验模型生成的参数。

    【输入输出对照】
        schema: {"properties": {"expression": {"type": "string"}}, "required": ["expression"]}
        ✓ {"expression": "3*7"}         → 放行
        ✗ {"expr": "3*7"}               → 拦截：缺少必填参数 ['expression']
        ✗ {"expression": "3*7", "x": 1} → 拦截：出现未知参数 ['x']
        ✗ {"expression": 123}           → 拦截：参数 expression 类型应为 string
    """

    name = "schema"

    def __init__(self, schema_of):
        # schema_of: 可调用对象，入参工具名，返回 JSON Schema dict（由 ToolRegistry 提供）
        self._schema_of = schema_of

    def _validate(self, schema: dict, arguments: dict) -> Optional[tuple[str, str]]:
        properties = (schema or {}).get("properties", {}) or {}
        required = (schema or {}).get("required", []) or []

        missing = [r for r in required if r not in arguments or arguments[r] is None]
        if missing:
            return "schema_invalid", f"缺少必填参数 {missing}"

        unknown = [k for k in arguments if k not in properties and not str(k).startswith(_PRIVATE_PREFIX)]
        if unknown and properties:
            return "schema_invalid", f"出现未知参数 {unknown}（请只使用工具声明的参数）"

        # 【key 和 spec 各是什么】properties 是字典，items() 拆成一对一对遍历：
        #   key  = 字段名（下面用它去 arguments 里取模型传的值）
        #   spec = 这个字段的规格说明书（也是字典），本轮只从里面取 "type"
        #
        #   以 data_query 为例（它声明了两个字段，data_query.py:100-110）：
        #     properties = {
        #       "data_type": {"type": "string", "enum": [...], "description": "数据类型..."},
        #       "record_id": {"type": "string", "description": "记录ID，留空返回全部"},
        #     }
        #   循环走两轮：
        #     第 1 轮  key="data_type"  spec={"type": "string", "enum": [...], ...}
        #     第 2 轮  key="record_id"  spec={"type": "string", "description": ...}
        #   spec 里的 enum / description 这一层不参与校验：enum 不查，description 是给模型看的
        for key, spec in properties.items():
            # 【为什么模型没传就跳过】properties 里列的字段不等于都必填：
            #   data_query 的 required 只有 ["data_type"]，record_id 是选填的，
            #   不传它是合法用法（表示"查全部"），不能在这里误报类型错误。
            #   必填已在上面 83 行单独查过，这一轮只管「传了的值类型对不对」。
            if key not in arguments or arguments[key] is None:
                continue
            expected = (spec or {}).get("type")
            pytype = _TYPE_MAP.get(expected)
            if pytype is None or expected == "number":
                continue
            value = arguments[key]
            if expected == "boolean":
                if not isinstance(value, bool):
                    return "schema_invalid", f"参数 {key} 类型应为 boolean"
            elif not isinstance(value, pytype) or (expected in ("integer", "number") and isinstance(value, bool)):
                return "schema_invalid", f"参数 {key} 类型应为 {expected}"
        return None

    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        schema = self._schema_of(req.tool_name) if self._schema_of else {}
        if not schema:
            return None                      # 拿不到 schema（如未注册工具）→ 交由后续环节处理
        return self._validate(schema, req.arguments or {})


# 【什么】用户权限校验
# 【作用】模型看不到未授权工具，但工具名可能被"拼出来"，这里再补一道服务端判定
class PermissionMiddleware(ToolMiddleware):
    """权限校验：principal.can('tool:<name>')，以及能力声明里要求的额外权限码。

    【为什么 principal 为 None 时直接放行】
        CLI 调试 / 单元测试没有登录态，此时拦截等于让整条链路不可用；
        权限的正式入口是 API 层的 RBAC，这里只做二次确认，不做唯一判定。
    """

    name = "permission"

    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        principal = req.principal
        if principal is None:
            return None
        can = getattr(principal, "can", None)
        if can is None:
            return None
        if not can(f"tool:{req.tool_name}"):
            return "permission_denied", f"当前身份无权调用工具 {req.tool_name}"
        cap = req.capability
        for perm in getattr(cap, "permissions", ()) or ():
            if not can(perm):
                return "permission_denied", f"调用 {req.tool_name} 需要权限 {perm}，当前身份不具备"
        return None


# 【什么】业务范围校验（部门域 / 本人域）
# 【作用】拦住"用自己的身份查别人的数据"这类越界参数
class ScopeMiddleware(ToolMiddleware):
    """业务范围校验：参数里的 dept_code / user_code 不得超出当前身份的可见范围。

    【判定规则】
        scope_for(dt) == "self" → 参数里出现的 user_code 必须是本人
        scope_for(dt) == "dept" → 参数里出现的 dept_code 必须是本部门
        scope_for(dt) == "all"  → 不限制

    【输入输出对照】
        财务专员（scope=dept, dept=D_FIN）：
        ✓ {"dept_code": "D_FIN"} → 放行
        ✗ {"dept_code": "D_HR"}  → 拦截：dept_code=D_HR 超出本部门范围(D_FIN)
    """

    name = "scope"

    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        principal = req.principal
        cap = req.capability
        if principal is None or cap is None:
            return None
        scope_for = getattr(principal, "scope_for", None)
        if scope_for is None:
            return None
        args = req.arguments or {}
        for dt in getattr(cap, "data_scopes", ()) or ():
            try:
                scope = scope_for(dt)
            except Exception:
                continue
            if scope == "all":
                continue
            if scope == "dept":
                want = str(args.get("dept_code") or "")
                mine = str(getattr(principal, "dept_code", "") or "")
                if want and mine and want != mine:
                    return "scope_denied", f"dept_code={want} 超出本部门范围({mine})"
            elif scope == "self":
                want = str(args.get("user_code") or "")
                mine = str(getattr(principal, "user_code", "") or "")
                if want and mine and want != mine:
                    return "scope_denied", f"user_code={want} 超出本人数据范围"
        return None


# 【什么】调用限流
# 【作用】防止模型在几秒内狂调工具（含 MCP 子进程被反复拉起）
class RateLimitMiddleware(ToolMiddleware):
    """令牌桶限流：全局一个桶，每个用户一个桶。

    【为什么不自己实现令牌桶】
        modules.orchestration.control.RateLimiter 已经是双层的（全局 + 每发送者），
        复用它避免两份限流口径不一致。
    """

    name = "rate_limit"

    def __init__(self, capacity: int = 20, refill_per_sec: float = 5.0, per_user_capacity: int = 8,
                 per_user_refill: float = 2.0):
        from modules.orchestration.control import RateLimiter
        self._limiter = RateLimiter(
            capacity=max(1, int(capacity)),
            refill_per_sec=max(0.1, float(refill_per_sec)),
            per_sender_capacity=max(1, int(per_user_capacity)),
            per_sender_refill=max(0.1, float(per_user_refill)),
        )

    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        sender = str(getattr(req.principal, "user_code", "") or req.user_id or req.session_id or "")
        ok, by = self._limiter.acquire(sender or "anonymous", 1)
        if ok:
            return None
        return "rate_limited", f"工具调用过于频繁（被 {by} 限流），请稍后再试"


# 【什么】执行预算校验
# 【作用】次数 / token / 墙钟超时三道闸门，防死循环与上下文膨胀
class BudgetMiddleware(ToolMiddleware):
    """预算校验：读会话账本，任一配额打满即拦截。"""

    name = "budget"

    def __init__(self, budgets: BudgetManager):
        self._budgets = budgets

    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        budget = self._budgets.get(req.session_id or "default")
        ok, reason = budget.check(req.tool_name)
        if ok:
            return None
        budget.record_block()
        return "budget_exceeded", reason


# 【什么】中间件链
# 【作用】按固定顺序跑，首个拦截即返回；全通过才放行
class MiddlewareChain:
    """按顺序执行中间件，返回最终裁决。

    【输入输出对照】
        chain = MiddlewareChain([SchemaMiddleware(schema_of)])
        chain.run(req_ok)  → ToolVerdict(allowed=True,  action='allow')
        chain.run(req_bad) → ToolVerdict(allowed=False, code='schema_invalid', reason='缺少必填参数 [...]')
    """

    def __init__(self, middlewares: Optional[List[ToolMiddleware]] = None):
        self.middlewares: List[ToolMiddleware] = list(middlewares or [])

    def add(self, middleware: ToolMiddleware) -> "MiddlewareChain":
        self.middlewares.append(middleware)
        return self

    def run(self, req: ToolCallRequest) -> ToolVerdict:
        for mw in self.middlewares:
            try:
                hit = mw.check(req)
            except Exception as e:                       # 中间件自身出错不能中断整条链路
                _logger.warning(f"中间件 {mw.name} 异常，按放行处理: {e}")
                continue
            if hit:
                code, reason = hit
                _logger.warning(f"工具调用被拦截 [{code}]: {req.tool_name} | {reason}")
                return ToolVerdict.block(code, reason)
        return ToolVerdict.ok(req.arguments)

    # 【什么】已装配的中间件名
    # 【作用】启动日志与自检用
    def names(self) -> List[str]:
        return [mw.name for mw in self.middlewares]
