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
        # 【作用】越界就返回 (拦截码, 中文原因)；没越界返回 None（= 放行，交给下一道中间件）
        principal = req.principal                                       # 当前请求的身份快照；None = 未启用权限体系
        cap = req.capability                                            # 这把工具登记的能力，里面带着它涉及哪些数据域
        if principal is None or cap is None:
            return None                                                 # 【目的】没有身份或没有能力台账就不校验，行为与未接入权限时一致
        scope_for = getattr(principal, "scope_for", None)               # 取"问身份：某类数据你能看多宽"的那个方法
        if scope_for is None:
            return None                                                 # 【目的】身份对象没带这个方法就跳过，不强迫所有调用方都实现它
        args = req.arguments or {}                                      # 模型实际传进来的参数，越界就藏在这里
        for dt in getattr(cap, "data_scopes", ()) or ():                # 逐个数据域查：工具声明了 users/orders 就每一个都要过
            try:
                scope = scope_for(dt)                                   # 问一句：这个人对这类数据的可见范围是 all / dept / self
            except Exception:
                continue                                                # 【目的】身份算范围时出错只跳过这一个域，不让一次异常拖垮整次调用
            if scope == "all":
                continue                                                # 全域可见，不用比参数
            if scope == "dept":
                want = str(args.get("dept_code") or "")                 # 参数里想查哪个部门
                mine = str(getattr(principal, "dept_code", "") or "")   # 这个人属于哪个部门
                if want and mine and want != mine:                      # 两边都有值且不相等才判越界：有一边为空就放行，避免误伤
                    return "scope_denied", f"dept_code={want} 超出本部门范围({mine})"
            elif scope == "self":
                want = str(args.get("user_code") or "")                 # 参数里想查哪个人的数据
                mine = str(getattr(principal, "user_code", "") or "")   # 这个人自己的工号
                if want and mine and want != mine:                      # 同上：只拦"明确指向别人"的情况
                    return "scope_denied", f"user_code={want} 超出本人数据范围"
        return None                                                     # 所有数据域都过完、没有越界 → 放行


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
        from modules.orchestration.control import RateLimiter   # 放函数里：按需加载，避免顶层循环导入
        # ------------------------------------------------------------------
        # 【这里自己不算限流】真正的算法在 RateLimiter 里，本方法只负责"把它建好"。
        #   两组桶：capacity/refill 是全局桶（所有人共用），per_user_* 是每人一个桶。
        #   max() 不是随手写的，是下限保护：
        #     容量保底 1   —— 传 0 会让桶永远是空的 → 所有调用全被拦 = 整站不可用
        #     速率保底 0.1 —— 传 0 令牌永不回生     → 打完 8 次就永久封死，只能重启进程
        # ------------------------------------------------------------------
        self._limiter = RateLimiter(
            capacity=max(1, int(capacity)),                      # 全系统瞬时最多连打几次
            refill_per_sec=max(0.1, float(refill_per_sec)),      # 之后每秒回血几个 = 长期平均速率
            per_sender_capacity=max(1, int(per_user_capacity)),  # 同一个人最多连打几次
            per_sender_refill=max(0.1, float(per_user_refill)),  # 之后每秒回血几个
        )

    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        # ------------------------------------------------------------------
        # 【三步】认出是谁 → 问桶要 1 枚令牌 → 把结果翻译成中间件契约
        #   契约（基类 ToolMiddleware.check 定死）：None = 放行，(code, reason) = 拦截
        #   "限流的记忆"在 self._limiter 这个实例属性里——桶内记着剩余令牌数与上次补充时间，
        #   所以本中间件必须长期持有、全局单例，每次 check 都扣在同一个桶上。
        # ------------------------------------------------------------------
        sender = str(getattr(req.principal, "user_code", "") or req.user_id or req.session_id or "")
        #  ↑ 用 getattr 而不是 req.principal.user_code：principal 可能是 None（CLI / 单测无登录态），
        #    直接取属性会崩。后半段是三级兜底：工号 → user_id → session_id，都没有就退回 anonymous
        ok, by = self._limiter.acquire(sender or "anonymous", 1)
        #  ↑ 一次调用 = 扣 1 枚令牌；判两次：全局桶先扣，再扣这个人自己的桶，任一扣不动就拦
        #    by = "global"（整体太猛）或 "sender:U007"（某个人太猛），写进原因里便于日志定位
        #    "anonymous"：没身份时所有人共用一个桶——宁可共用，也不跳过限流
        if ok:
            return None                                          # 放行（令牌已在 acquire 里扣掉，这里只是翻译结果）
        return "rate_limited", f"工具调用过于频繁（被 {by} 限流），请稍后再试"


# 【什么】执行预算校验
# 【作用】次数 / token / 墙钟超时三道闸门，防死循环与上下文膨胀
class BudgetMiddleware(ToolMiddleware):
    """预算校验：读会话账本，任一配额打满即拦截。"""

    name = "budget"

    def __init__(self, budgets: BudgetManager):
        # 【这里不记账】账本按会话存在 BudgetManager 里，本中间件只是"拿账本查一下"的那道闸
        self._budgets = budgets

    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        # ------------------------------------------------------------------
        # 契约与 rate_limit 那道一致：None = 放行，(code, reason) = 拦截
        #   【和限流的区别】限流看"速率"（令牌会按秒回血，等一会儿就恢复）；
        #   预算看"总量"（次数 / token / 墙钟），打满不会自己恢复，只能 begin() 重开一轮。
        # ------------------------------------------------------------------
        budget = self._budgets.get(req.session_id or "default")
        #  ↑ 按会话取账本：BudgetManager.get 内部兜底，没 begin 过也会用默认 spec 现开一本
        #    "default" 是无会话时的兜底 key（CLI / 单测），避免用 None 当 key 让所有无会话调用挤在一起
        ok, reason = budget.check(req.tool_name)
        #  ↑ 四道闸门任一打满就返回 False + 中文原因：
        #    超时 / 总次数上限 / 单工具次数上限（防死循环）/ 上下文 token 预算
        #    注意 check 只"问"，不扣配额——真正记账在调用成功后的 consume()
        if ok:
            return None                                  # 放行
        budget.record_block()                            # 被拦才记一次：blocked 高 = 配额偏紧或模型在打转
        return "budget_exceeded", reason                 # 原因直接复用账本给的，不在这里二次拼装


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
                
                # 工具出错 还会继续执行其他的
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
