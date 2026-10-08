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
from __future__ import annotations           # 允许类型注解里用未导入的名字（Python 3.7+ 兼容写法）

import time                                   # 标准库时间模块（预留给限流/计时类逻辑使用）
from typing import Any, Dict, List, Optional  # 标准库类型：Any 任意 / Dict 字典 / List 列表 / Optional 可空

from modules.common.logger import get_logger          # 统一日志工厂
from modules.harness.budget import BudgetManager, BudgetSpec  # 预算管理器与预算规格（BudgetMiddleware 依赖）
from modules.harness.models import ToolCallRequest, ToolVerdict  # 调用请求模型与裁决单模型

_logger = get_logger("harness.middleware")   # 模块级 logger，名字 = "harness.middleware"，便于按模块过滤日志

# 服务端注入的私有参数前缀：这些不在 schema 里，模型给不出来，校验时跳过
_PRIVATE_PREFIX = "_"                        # 私有参数前缀常量；如 _dept_code / _mask_level

# JSON Schema 类型 → Python 类型
_TYPE_MAP = {
    "string": str,                           # schema 的 string → Python str
    "number": (int, float),                  # schema 的 number → int 或 float 都算
    "integer": int,                          # schema 的 integer → Python int
    "boolean": bool,                         # schema 的 boolean → Python bool
    "object": dict,                          # schema 的 object → Python dict
    "array": list,                           # schema 的 array → Python list
}


# 【什么】中间件基类：要么放行，要么给出拦截码与原因
# 【作用】新增一道校验只需实现 check()，不用改调用链
# 【为什么】统一契约（None=放行 / (code, reason)=拦截）让链条可以无脑顺序执行
class ToolMiddleware:
    """调用前中间件基类。check 返回 None 表示放行，返回 (code, reason) 表示拦截。"""

    name = "base"                            # 中间件名字；子类各自覆盖，用于日志定位是哪一道拦的

    # 【什么】基类默认校验
    # 【作用】默认放行；子类覆盖实现各自校验逻辑
    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        return None                          # None = 放行（子类返回 (code, reason) 即拦截）


# 【什么】参数合法性校验
# 【作用】把幻觉参数挡在门外：缺失必填、多出未知参数、类型不对，一律拦截
# 【为什么】模型的参数是"生成"的，幻觉参数到真实服务才报错就晚了（副作用已发生）
class SchemaMiddleware(ToolMiddleware):
    """按工具的 JSON Schema 校验模型生成的参数。

    【输入输出对照】
        schema: {"properties": {"expression": {"type": "string"}}, "required": ["expression"]}
        ✓ {"expression": "3*7"}         → 放行
        ✗ {"expr": "3*7"}               → 拦截：缺少必填参数 ['expression']
        ✗ {"expression": "3*7", "x": 1} → 拦截：出现未知参数 ['x']
        ✗ {"expression": 123}           → 拦截：参数 expression 类型应为 string
    """

    name = "schema"                          # 本道中间件的名字，拦截日志里会带它

    # 【什么】构造函数
    # 【作用】注入"按工具名取 schema"的取数函数（由 Harness 在 bind_registry 时装配）
    def __init__(self, schema_of):
        # schema_of: 可调用对象，入参工具名，返回 JSON Schema dict（由 ToolRegistry 提供）
        self._schema_of = schema_of          # 保存取数函数；校验时按工具名现取 schema

    # 【什么】对照 schema 校验实参
    # 【作用】依次检查：必填缺失 → 未知参数 → 已传参数的类型
    # 【为什么】三类错误分开报，回喂模型的中文原因才具体可改
    def _validate(self, schema: dict, arguments: dict) -> Optional[tuple[str, str]]:
        properties = (schema or {}).get("properties", {}) or {}   # 取字段规格表；schema 为空/缺字段都兜底成空 dict
        required = (schema or {}).get("required", []) or []       # 取必填字段列表；缺失时兜底成空列表

        missing = [r for r in required if r not in arguments or arguments[r] is None]  # 必填里"没传"或"传了 None"的都算缺失
        if missing:                                                # 有缺失必填
            return "schema_invalid", f"缺少必填参数 {missing}"     # 拦截：告诉模型缺了哪些必填参数

        unknown = [k for k in arguments if k not in properties and not str(k).startswith(_PRIVATE_PREFIX)]  # 模型传了但 schema 没声明、且不是私有参数的键
        if unknown and properties:                                 # 有未知参数且 schema 确实声明了字段集（空 properties 不判未知，等于不校验）
            return "schema_invalid", f"出现未知参数 {unknown}（请只使用工具声明的参数）"  # 拦截：幻觉参数挡在门外

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
        for key, spec in properties.items():                       # 逐字段检查"传了的值类型对不对"
            # 【为什么模型没传就跳过】properties 里列的字段不等于都必填：
            #   data_query 的 required 只有 ["data_type"]，record_id 是选填的，
            #   不传它是合法用法（表示"查全部"），不能在这里误报类型错误。
            #   必填已在上面 83 行单独查过，这一轮只管「传了的值类型对不对」。
            if key not in arguments or arguments[key] is None:     # 该字段没传或传了 None
                continue                                           # 跳过：选填不传是合法用法
            expected = (spec or {}).get("type")                    # 取该字段声明的 schema 类型
            pytype = _TYPE_MAP.get(expected)                       # 映射成 Python 类型
            if pytype is None or expected == "number":             # 未知类型不校验；number 单独放行（int/float 都算，统一由下面分支处理）
                continue                                           # 跳过该字段的类型检查
            value = arguments[key]                                 # 模型实际传入的值
            if expected == "boolean":                              # bool 是 int 的子类，必须单独判，否则 True 会被当成合法 integer
                if not isinstance(value, bool):                    # 值不是布尔
                    return "schema_invalid", f"参数 {key} 类型应为 boolean"  # 拦截：类型不对
            elif not isinstance(value, pytype) or (expected in ("integer", "number") and isinstance(value, bool)):  # 类型不符，或数字类型收到了 bool（True/False 不算数字）
                return "schema_invalid", f"参数 {key} 类型应为 {expected}"   # 拦截：告诉模型期望类型
        return None                                                # 三类检查全过 → 放行

    # 【什么】SchemaMiddleware 的入口
    # 【作用】取 schema 后交给 _validate；取不到 schema 时跳过本道校验
    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        schema = self._schema_of(req.tool_name) if self._schema_of else {}  # 按工具名取 schema；没注入取数函数就当空
        if not schema:                                             # schema 为空（未注册工具/未绑 registry）
            return None                      # 拿不到 schema（如未注册工具）→ 交由后续环节处理
        return self._validate(schema, req.arguments or {})         # 实参兜底成空 dict 后做校验


# 【什么】用户权限校验
# 【作用】模型看不到未授权工具，但工具名可能被"拼出来"，这里再补一道服务端判定
# 【为什么】前端/提示词层的隐藏只是"不展示"，不能当安全边界；服务端必须独立再判一次
class PermissionMiddleware(ToolMiddleware):
    """权限校验：principal.can('tool:<name>')，以及能力声明里要求的额外权限码。

    【为什么 principal 为 None 时直接放行】
        CLI 调试 / 单元测试没有登录态，此时拦截等于让整条链路不可用；
        权限的正式入口是 API 层的 RBAC，这里只做二次确认，不做唯一判定。
    """

    name = "permission"                      # 本道中间件的名字

    # 【什么】权限校验入口
    # 【作用】先查"能不能调这个工具"，再查能力声明里要求的每一个额外权限码
    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        principal = req.principal                        # 当前请求的身份快照
        if principal is None:                            # 无登录态（CLI/单测）
            return None                                  # 放行：正式权限判定在 API 层 RBAC
        can = getattr(principal, "can", None)            # 取身份对象的权限判定方法
        if can is None:                                  # 身份对象没实现 can()
            return None                                  # 放行：不强迫所有身份对象都实现该方法
        if not can(f"tool:{req.tool_name}"):             # 查"tool:<名字>"这个权限点
            return "permission_denied", f"当前身份无权调用工具 {req.tool_name}"  # 拦截：无该工具权限
        cap = req.capability                             # 该工具登记的能力声明
        for perm in getattr(cap, "permissions", ()) or ():  # 遍历能力声明要求的额外权限码
            if not can(perm):                            # 缺其中任何一个权限码
                return "permission_denied", f"调用 {req.tool_name} 需要权限 {perm}，当前身份不具备"  # 拦截：缺具体权限码
        return None                                      # 工具权限 + 额外权限码全过 → 放行


# 【什么】业务范围校验（部门域 / 本人域）
# 【作用】拦住"用自己的身份查别人的数据"这类越界参数
# 【为什么】有工具权限 ≠ 能看所有数据；权限答"能不能调"，范围答"能看谁的"
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

    name = "scope"                           # 本道中间件的名字

    # 【什么】范围校验入口
    # 【作用】按工具声明的每个数据域，逐一比对参数里的 dept_code / user_code 是否越界
    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        # 【作用】越界就返回 (拦截码, 中文原因)；没越界返回 None（= 放行，交给下一道中间件）
        principal = req.principal                                       # 当前请求的身份快照；None = 未启用权限体系
        cap = req.capability                                            # 这把工具登记的能力，里面带着它涉及哪些数据域
        if principal is None or cap is None:                            # 没身份或没能力台账
            return None                                                 # 【目的】没有身份或没有能力台账就不校验，行为与未接入权限时一致
        scope_for = getattr(principal, "scope_for", None)               # 取"问身份：某类数据你能看多宽"的那个方法
        if scope_for is None:                                           # 身份对象没实现 scope_for
            return None                                                 # 【目的】身份对象没带这个方法就跳过，不强迫所有调用方都实现它
        args = req.arguments or {}                                      # 模型实际传进来的参数，越界就藏在这里
        for dt in getattr(cap, "data_scopes", ()) or ():                # 逐个数据域查：工具声明了 users/orders 就每一个都要过
            try:                                                        # scope_for 是身份模块的代码，异常类型不可控
                scope = scope_for(dt)                                   # 问一句：这个人对这类数据的可见范围是 all / dept / self
            except Exception:                                           # 算范围时抛错
                continue                                                # 【目的】身份算范围时出错只跳过这一个域，不让一次异常拖垮整次调用
            if scope == "all":                                          # 全域可见
                continue                                                # 全域可见，不用比参数
            if scope == "dept":                                         # 只能看本部门
                want = str(args.get("dept_code") or "")                 # 参数里想查哪个部门
                mine = str(getattr(principal, "dept_code", "") or "")   # 这个人属于哪个部门
                if want and mine and want != mine:                      # 两边都有值且不相等才判越界：有一边为空就放行，避免误伤
                    return "scope_denied", f"dept_code={want} 超出本部门范围({mine})"  # 拦截：查了别的部门
            elif scope == "self":                                       # 只能看本人
                want = str(args.get("user_code") or "")                 # 参数里想查哪个人的数据
                mine = str(getattr(principal, "user_code", "") or "")   # 这个人自己的工号
                if want and mine and want != mine:                      # 同上：只拦"明确指向别人"的情况
                    return "scope_denied", f"user_code={want} 超出本人数据范围"        # 拦截：查了别人的数据
        return None                                                     # 所有数据域都过完、没有越界 → 放行


# 【什么】调用限流
# 【作用】防止模型在几秒内狂调工具（含 MCP 子进程被反复拉起）
# 【为什么】模型可能陷入"生成→调用→再生成"的快循环，没有限流会把下游服务打爆
class RateLimitMiddleware(ToolMiddleware):
    """令牌桶限流：全局一个桶，每个用户一个桶。

    【为什么不自己实现令牌桶】
        modules.orchestration.control.RateLimiter 已经是双层的（全局 + 每发送者），
        复用它避免两份限流口径不一致。
    """

    name = "rate_limit"                      # 本道中间件的名字

    # 【什么】构造函数
    # 【作用】按容量/速率建双层令牌桶（全局桶 + 每人一个桶）
    # 【为什么】两组 max() 是下限保护：容量 0 会全拦、速率 0 会永久封死
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

    # 【什么】限流校验入口
    # 【作用】认出是谁 → 问两个桶各要 1 枚令牌 → 扣不动就拦
    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        # ------------------------------------------------------------------
        # 【三步】认出是谁 → 问桶要 1 枚令牌 → 把结果翻译成中间件契约
        #   契约（基类 ToolMiddleware.check 定死）：None = 放行，(code, reason) = 拦截
        #   "限流的记忆"在 self._limiter 这个实例属性里——桶内记着剩余令牌数与上次补充时间，
        #   所以本中间件必须长期持有、全局单例，每次 check 都扣在同一个桶上。
        # ------------------------------------------------------------------
        sender = str(getattr(req.principal, "user_code", "") or req.user_id or req.session_id or "")  # 发送者标识：工号 → user_id → session_id 三级兜底
        #  ↑ 用 getattr 而不是 req.principal.user_code：principal 可能是 None（CLI / 单测无登录态），
        #    直接取属性会崩。后半段是三级兜底：工号 → user_id → session_id，都没有就退回 anonymous
        ok, by = self._limiter.acquire(sender or "anonymous", 1)  # 扣 1 枚令牌；ok=是否扣成功，by=哪个桶扣不动
        #  ↑ 一次调用 = 扣 1 枚令牌；判两次：全局桶先扣，再扣这个人自己的桶，任一扣不动就拦
        #    by = "global"（整体太猛）或 "sender:U007"（某个人太猛），写进原因里便于日志定位
        #    "anonymous"：没身份时所有人共用一个桶——宁可共用，也不跳过限流
        if ok:                                                   # 两个桶都扣到了令牌
            return None                                          # 放行（令牌已在 acquire 里扣掉，这里只是翻译结果）
        return "rate_limited", f"工具调用过于频繁（被 {by} 限流），请稍后再试"  # 拦截：告诉模型是谁的桶打满了


# 【什么】执行预算校验
# 【作用】次数 / token / 墙钟超时三道闸门，防死循环与上下文膨胀
# 【为什么】限流管"速率"（会回血），预算管"总量"（打满不恢复），两者互补缺一不可
class BudgetMiddleware(ToolMiddleware):
    """预算校验：读会话账本，任一配额打满即拦截。"""

    name = "budget"                          # 本道中间件的名字

    # 【什么】构造函数
    # 【作用】注入预算管理器；本中间件只查账不记账
    def __init__(self, budgets: BudgetManager):
        # 【这里不记账】账本按会话存在 BudgetManager 里，本中间件只是"拿账本查一下"的那道闸
        self._budgets = budgets                          # 保存预算管理器引用

    # 【什么】预算校验入口
    # 【作用】按会话取账本问一句"配额够不够"；被拦时记一次 blocked
    def check(self, req: ToolCallRequest) -> Optional[tuple[str, str]]:
        # ------------------------------------------------------------------
        # 契约与 rate_limit 那道一致：None = 放行，(code, reason) = 拦截
        #   【和限流的区别】限流看"速率"（令牌会按秒回血，等一会儿就恢复）；
        #   预算看"总量"（次数 / token / 墙钟），打满不会自己恢复，只能 begin() 重开一轮。
        # ------------------------------------------------------------------
        budget = self._budgets.get(req.session_id or "default")  # 按会话取账本；空会话兜底成 "default"
        #  ↑ 按会话取账本：BudgetManager.get 内部兜底，没 begin 过也会用默认 spec 现开一本
        #    "default" 是无会话时的兜底 key（CLI / 单测），避免用 None 当 key 让所有无会话调用挤在一起
        ok, reason = budget.check(req.tool_name)         # 问账本：这个工具还能不能再调一次
        #  ↑ 四道闸门任一打满就返回 False + 中文原因：
        #    超时 / 总次数上限 / 单工具次数上限（防死循环）/ 上下文 token 预算
        #    注意 check 只"问"，不扣配额——真正记账在调用成功后的 consume()
        if ok:                                           # 配额还够
            return None                                  # 放行
        budget.record_block()                            # 被拦才记一次：blocked 高 = 配额偏紧或模型在打转
        return "budget_exceeded", reason                 # 原因直接复用账本给的，不在这里二次拼装


# 【什么】中间件链
# 【作用】按固定顺序跑，首个拦截即返回；全通过才放行
# 【为什么】短路返回让拦截原因精确指向"第一道没过的是谁"，也省掉后续无谓检查
class MiddlewareChain:
    """按顺序执行中间件，返回最终裁决。

    【输入输出对照】
        chain = MiddlewareChain([SchemaMiddleware(schema_of)])
        chain.run(req_ok)  → ToolVerdict(allowed=True,  action='allow')
        chain.run(req_bad) → ToolVerdict(allowed=False, code='schema_invalid', reason='缺少必填参数 [...]')
    """

    # 【什么】构造函数
    # 【作用】接收中间件列表；list() 拷贝防外部篡改
    def __init__(self, middlewares: Optional[List[ToolMiddleware]] = None):
        self.middlewares: List[ToolMiddleware] = list(middlewares or [])  # 中间件列表；None 兜底成空链

    # 【什么】追加一道中间件
    # 【作用】链式调用装配；返回 self 便于连续 add
    def add(self, middleware: ToolMiddleware) -> "MiddlewareChain":
        self.middlewares.append(middleware)              # 追加到链尾（执行顺序 = 装配顺序）
        return self                                      # 返回自己支持链式调用

    # 【什么】按顺序跑整条链
    # 【作用】任一中间件返回 (code, reason) 就短路成 ToolVerdict.block；全过返回 ToolVerdict.ok
    # 【为什么】中间件自身异常按放行处理：一道安检坏了不该把整站锁死（fail-open 仅限中间件自身故障）
    def run(self, req: ToolCallRequest) -> ToolVerdict:
        for mw in self.middlewares:                      # 按装配顺序逐道过
            try:                                         # 包住中间件自身代码，异常类型不可控
                # 【作用】中间件校验：根据请求判断是否拦截
                hit = mw.check(req)                      # None = 放行；(code, reason) = 拦截
            except Exception as e:                       # 中间件自身出错不能中断整条链路
                _logger.warning(f"中间件 {mw.name} 异常，按放行处理: {e}")  # 告警后继续下一道
                
                # 工具出错 还会继续执行其他的
                continue                                 # 跳过这道异常中间件，继续下一道
            if hit:                                      # 该道返回了拦截结果
                code, reason = hit                       # 拆出拦截码与中文原因
                _logger.warning(f"工具调用被拦截 [{code}]: {req.tool_name} | {reason}")  # 告警：哪道拦的、拦的谁、为什么
                return ToolVerdict.block(code, reason)   # 短路返回拦截裁决单，后续中间件不再执行
        return ToolVerdict.ok(req.arguments)             # 五道全过 → 放行裁决单（带原参数）

    # 【什么】已装配的中间件名
    # 【作用】启动日志与自检用
    # 【为什么】让运维确认"五道都在、顺序没错"，不用翻代码
    def names(self) -> List[str]:
        return [mw.name for mw in self.middlewares]      # 按执行顺序返回各中间件的名字
