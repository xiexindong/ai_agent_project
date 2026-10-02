"""
工具 Harness（总装）
====================
把「能力声明 → 调用前校验 → 人工审批 → 沙箱执行 → 结果过滤 → 审计」串成一条链。

【Harness 在这里的职责边界】
    Agent 负责"想调什么"，Harness 负责"能不能调、在哪调、调完怎么办"。
    模型自始至终只有"发起申请"的权利：
        - 高风险工具：必须人工批准，未经批准一律不下发
        - 禁止类工具：配置层面封死，模型连申请都不会被受理
        - 参数/权限/范围/限流/预算任一不过：直接拦截，真实服务收不到请求

【一次调用的五个阶段】
    before_call   策略裁决 + 五道中间件 + 人工审批
    execute       按能力声明选择沙箱执行
    after_call    结果裁剪脱敏 + token 记账 + 审计落库

【完整数据流：一次工具调用从头到尾 12 步】
    0. 注册期   ToolRegistry.register(tool) → capability_of() 读声明 → policy.effective()
                → 写进 _capabilities 台账（解析只做一次，调用期只读台账）
    1. Agent    ReAct 判定要调工具 → prepare() 开一本预算账 begin_run(session_id)
    2. Registry execute_tool(name, params) → get_tool → capability_of 取台账
    3. Registry 同参去重 / 单工具限次（这是 registry 自己的账，与 Harness 预算并行）
    4. Harness  before_call(req)：
                ① policy.decide → deny/approve/allow
                ② 五道中间件（schema → permission → scope → rate_limit → budget）
                ③ 需审批的走 HumanGate（可改参数）
    5. Harness  execute：按 cap.sandbox 选工位 → none / subprocess / docker / remote
    6. 执行      进程内 tool.run() / 子进程 worker / 容器 / 远端 HTTP（传的都是"类坐标"）
    7. Harness  after_call：ResultFilter 脱敏 + 裁剪 → estimate_tokens 记账
    8. Harness  审计落库（放行 / 拒绝 / 失败三种 state），detail 带风险/审批/沙箱/预算
    9. Registry 成功结果进 _result_cache，重复调用直接回传（省一轮 LLM）
   10. Agent    结果回灌 prompt → LLM 决定收尾还是再调一次 → 回到第 1 步
   11. 收尾     新一轮提问 reset_session() → end_run() 销账，配额不被上轮吃掉

【默认行为保证】
    enabled=False 或未装配 harness 时，ToolRegistry 走原有流程，行为与改造前一致。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from modules.common.logger import get_logger
from modules.harness.budget import BudgetManager, BudgetSpec, RunBudget
from modules.harness.middleware import (
    BudgetMiddleware, MiddlewareChain, PermissionMiddleware, RateLimitMiddleware,
    SchemaMiddleware, ScopeMiddleware,
)
from modules.harness.models import (
    ACTION_APPROVE, ACTION_DENY, RiskLevel, ToolCallRequest, ToolCapability, ToolPolicy, ToolVerdict,
)
from modules.harness.result_filter import ResultFilter, ResultFilterConfig
from modules.harness.sandbox import MODE_DOCKER, MODE_NONE, MODE_REMOTE, InProcessRunner, SandboxRunner, build_sandbox
from modules.tools.base_tool import ToolResult

_logger = get_logger("harness")


# 【什么】工具运行时框架
# 【作用】工具调用的唯一"安全与预算"入口；registry 只管找工具，安全判断全在它
class ToolHarness:
    """工具 Harness：能力登记、调用前校验、人工审批、沙箱执行、结果过滤与审计。

    【快速上手】
        h = build_tool_harness()
        h.bind_registry(registry)                       # 让 SchemaMiddleware 能取到 JSON Schema
        h.begin_run("session-1")                        # 开一本预算账
        result = h.call(tool, {"expression": "3*7"}, session_id="session-1", principal=None)
    """

    def __init__(
        self,
        policy: Optional[ToolPolicy] = None,
        budgets: Optional[BudgetManager] = None,
        chain: Optional[MiddlewareChain] = None,
        result_filter: Optional[ResultFilter] = None,
        gate: Any = None,                       # HumanGate；None = 不启用人工审批
        enabled: bool = True,
        audit_enabled: bool = True,
        default_sandbox: str = MODE_NONE,
        default_timeout: float = 15.0,
        docker_options: Optional[dict] = None,  # 容器沙箱参数（image/network/memory/cpus）
        remote_options: Optional[dict] = None,  # 远程沙箱参数（endpoint/token），mode=remote 时生效
    ):
        self.policy = policy or ToolPolicy()
        self.budgets = budgets or BudgetManager()
        self.chain = chain or MiddlewareChain()
        self.result_filter = result_filter or ResultFilter()
        self.gate = gate
        self.enabled = bool(enabled)
        self.audit_enabled = bool(audit_enabled)
        self.default_sandbox = str(default_sandbox or MODE_NONE)
        self.default_timeout = float(default_timeout or 0.0) or 15.0
        # ------------------------------------------------------------------
        # 【容器沙箱参数】来自 TOOL_SANDBOX_IMAGE 等配置，只在 mode=docker 时下发
        #   【不这么写会怎样】把这组参数无条件传给 build_sandbox，
        #   mode=subprocess 时会因 SubprocessRunner 不认 image 抛 TypeError。
        # ------------------------------------------------------------------
        self.docker_options = dict(docker_options or {})
        # 【远程沙箱参数】endpoint/token；与容器参数同理，只在 mode=remote 时下发
        self.remote_options = dict(remote_options or {})
        self._registry: Any = None
        self._capabilities: Dict[str, ToolCapability] = {}
        self._runners: Dict[str, Any] = {MODE_NONE: InProcessRunner()}

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    # 【什么】绑定工具注册表
    # 【作用】SchemaMiddleware 需要按工具名取 JSON Schema，这里延迟注入避免循环依赖
    def bind_registry(self, registry: Any) -> None:
        self._registry = registry
        for mw in self.chain.middlewares:
            if isinstance(mw, SchemaMiddleware):
                mw._schema_of = self._schema_of

    def _schema_of(self, tool_name: str) -> dict:
        if self._registry is None:
            return {}
        tool = self._registry.get_tool(tool_name)
        if tool is None:
            return {}
        schema = getattr(tool, "parameters_schema", None)
        return schema if isinstance(schema, dict) else {}

    # ------------------------------------------------------------------
    # ① 能力声明与登记
    # ------------------------------------------------------------------
    # 【什么】读取并归一化一个工具的能力声明
    # 【作用】工具可以声明 capability（属性/方法/dict），没声明就按默认等级兜底
    def capability_of(self, tool: Any) -> ToolCapability:
        raw = getattr(tool, "capability", None)   # 取工具挂出的能力声明，可能是属性/方法/dict，没挂返回 None
                                                  # 例子：raw = {"risk": "high", "side_effect": True}
        # ------------------------------------------------------------------
        # 【if callable(raw)】判断"这份声明是方法还是值"
        #   工具可以把 capability 写成三种形态：属性（ToolCapability 实例）/ 方法 / dict。
        #   属性与 dict 已经是值，直接往下走 isinstance 判断即可；
        #   只有方法形态必须先调一次（见下面 raw = raw()），否则拿到的是方法对象本身。
        #
        # 【try 包的是谁的代码】包的是工具作者写的 capability 方法
        #   它可能读配置、查库、取环境变量，抛出来的类型不可控
        #   （KeyError / AttributeError / 连接错都可能有），
        #   所以这里捕获 Exception 而不是具体某一类异常。
        #
        # 【输入输出对照】
        #   raw = <bound method X.capability>   → callable=True  → 进 try，调用后拿到声明
        #   raw = ToolCapability(risk=LOW)      → callable=False → 走 isinstance 的工具实例分支
        #   raw = {"risk": "high"}              → callable=False → 走 dict 分支（按字段解析）
        #   raw = None                          → callable=False → 走 policy.default_risk 兜底
        #
        # 【边界】判断顺序固定：先 callable，再 None / isinstance，最后兜底。
        #   顺序调整（例如把 isinstance 提到 callable 前面）会让方法形态漏判，
        #   声明静默失效——不报错，但风险等级回落到默认值。
        # ------------------------------------------------------------------
        
        # 它是 Python 的一个内置函数，专门用来检查一个对象能不能被“调用”
        if callable(raw):
            try:
                # ------------------------------------------------------------------
                # 【raw 在这一行的作用】从"能取到声明的东西"变成"声明本身"
                #   raw 是工具挂出来的 capability，可能是属性、方法、dict 三选一：
                #   callable(raw) 为真说明工具把它写成了方法，
                #   不调用就拿不到里面的内容，下面 142/144 行的 isinstance 判断会全部落空，
                #   结果是按 155 行的默认风险等级兜底（工具声明白写了）。
                #
                # 【输入输出对照】
                #   调用前：raw = <bound method CalculatorTool.capability>   # 只是个方法对象
                #   调用后：raw = ToolCapability(name="calculator", risk=RiskLevel.LOW)
                #   方法内部抛异常时：raw = None → 走 140 行按 policy.default_risk 兜底
                #
                # 【边界】这一步只降级不中断：
                #   一个工具的声明写错，不该让整批工具都注册不进来
                # ------------------------------------------------------------------
                raw = raw()
            except Exception as e:
                _logger.warning(f"工具能力声明读取失败: {getattr(tool, 'name', '?')} | {e}")
                raw = None
        name = getattr(tool, "name", "") or ""  # 取工具挂出的名字，没挂返回空字符串
        if raw is None:
            cap: ToolCapability = ToolCapability(name=name, risk=self.policy.default_risk)
            # isinstance() 是 Python 里最常用的内置函数之一，用来判断一个对象是不是某个类型
        elif isinstance(raw, ToolCapability):
            cap = raw
        elif isinstance(raw, dict):
            cap = ToolCapability(name=raw.get("name", name), risk=RiskLevel.parse(raw.get("risk")),
                                 side_effect=bool(raw.get("side_effect", False)),
                                 sandbox=str(raw.get("sandbox", MODE_NONE)),
                                 requires_approval=bool(raw.get("requires_approval", False)),
                                 permissions=tuple(raw.get("permissions", ()) or ()),
                                 data_scopes=tuple(raw.get("data_scopes", ()) or ()),
                                 max_result_chars=int(raw.get("max_result_chars", 0) or 0),
                                 timeout_seconds=float(raw.get("timeout_seconds", 0) or 0),
                                 description=str(raw.get("description", "") or ""))
        else:
            cap = ToolCapability(name=name, risk=self.policy.default_risk)
        return self.policy.effective(cap)

    # 【什么】登记一个工具的能力
    # 【作用】注册时调一次，之后裁决不再重复解析；返回生效后的能力
    def register(self, tool: Any) -> ToolCapability:
        # 【作用】像给工具办一张通行证：工具自己填的、配置里改的、没填就按默认的，
        #         三份说法在这里一次性定下来（只算这一次，之后都查表）
        cap = self.capability_of(tool)
        # 【作用】存进能力台账，之后裁决直接按工具名查表，不再重复解析
        self._capabilities[cap.name] = cap
        if cap.needs_approval or cap.risk.rank >= RiskLevel.HIGH.rank:
            _logger.info(f"工具能力登记: {cap.name} | 风险={cap.risk.value} | "
                         f"需人工批准={cap.needs_approval} | 沙箱={cap.sandbox}")
        return cap

    # 【什么】已登记能力清单
    # 【作用】/tools 或自检接口展示"每个工具是什么风险等级"
    def capabilities(self) -> Dict[str, dict]:
        return {name: cap.to_dict() for name, cap in self._capabilities.items()}

    # ------------------------------------------------------------------
    # ④ 预算：开账 / 销账 / 取账
    # ------------------------------------------------------------------
    def begin_run(self, session_id: str, spec: Optional[BudgetSpec] = None) -> RunBudget:
        return self.budgets.begin(session_id or "default", spec)

    def end_run(self, session_id: str) -> None:
        self.budgets.end(session_id or "default")

    def budget(self, session_id: str) -> RunBudget:
        return self.budgets.get(session_id or "default")

    # ------------------------------------------------------------------
    # ① + ② 调用前：策略裁决 + 中间件 + 人工审批
    # ------------------------------------------------------------------
    def before_call(self, req: ToolCallRequest) -> ToolVerdict:
        """调用前校验：这是"能不能调"的唯一准入判定入口。

        不过就是不过，真实服务不会收到请求 —— 副作用不会先发生再回滚。
        顺序固定：① 策略裁决（配置层面封死）→ ② 五道中间件 → ③ 人工审批。
        """
        # ------------------------------------------------------------------
        # 【这三关在干什么】"让不让调"的三道关卡，顺序固定不能换
        #   比喻：工具出门干活前，先查黑名单（策略）→ 过安检（中间件）→
        #         高风险的再找人签字（人工批准）；任一关不过就出不了门。
        #
        # 【顺序为什么不能换】按"拦住它的代价"从低到高排：
        #   策略只查一张表（最便宜）→ 中间件要校验参数/权限/配额（中等）→
        #   人工要等人（最贵）。反过来先把人叫来、再发现工具在黑名单里，就白等一场。
        #
        # 【输出】三选一：block（拦截 + 中文原因）/ 转人工 / ok（放行）
        #
        # 【输入 → 输出对照】同一个 before_call，三种工具三种结果（实测）
        #   ① 输入：data_query（被配置提级成 high，且没开人工通道）
        #      输出：allowed=False | action=approve | code=approval_required
        #            reason="工具 data_query 风险等级为 high，需人工批准；
        #                    当前未启用人工审批通道，已拒绝执行（模型无权直接调用）"
        #   ② 输入：calculator（low，无覆盖）
        #      输出：allowed=True  | action=allow | code=（空）→ 直接放行
        #   ③ 输入：drop_table（名字命中禁止名单）
        #      输出：allowed=False | action=deny   | code=policy_denied
        #            reason="工具 drop_table 已配置为禁止执行"
        # ------------------------------------------------------------------
        # 【作用】总开关关着就直接放行：Harness 是加固项，不是依赖项
        if not self.enabled:
            return ToolVerdict.ok(req.arguments)
        # 【作用】取这把工具的通行证：请求自带的优先，没有就查注册期台账，都没有就按默认等级临时造一张
        cap = req.capability or self._capabilities.get(req.tool_name) or ToolCapability(name=req.tool_name)

        decision = self.policy.decide(cap)
        if decision.denied:                                     # ① 配置层面禁止执行
            _logger.warning(f"工具被策略禁止: {req.tool_name} | {decision.reason}")
            return ToolVerdict.block(code="policy_denied", reason=decision.reason, action=ACTION_DENY)

        # 【作用】五道安检：参数 → 权限 → 数据范围 → 限流 → 预算，任一不过直接拦下
        verdict = self.chain.run(req)                           # ② 五道中间件
        if not verdict.allowed:
            return verdict

        if decision.action == ACTION_APPROVE:                   # ③ 高风险：人工批准
            return self._ask_human(req, verdict.arguments, decision.reason)
        return verdict

    # 【什么】向人工闸门申请批准
    # 【作用】模型只能发起申请；批准通过才放行，且人工可以改参数（modify）
    def _ask_human(self, req: ToolCallRequest, arguments: dict, reason: str) -> ToolVerdict:
        if self.gate is None:
            if str(self.policy.on_no_gate).lower() == "allow":
                _logger.warning(f"工具 {req.tool_name} 需人工批准但未启用闸门，按配置放行")
                return ToolVerdict.ok(arguments, action=ACTION_APPROVE, note="需人工批准但未启用闸门")
            return ToolVerdict.block(
                "approval_required",
                f"{reason}；当前未启用人工审批通道，已拒绝执行（模型无权直接调用）",
                action=ACTION_APPROVE,
            )
        try:
            approved, final_args = self.gate.ask_tool(
                req.tool_name, arguments, agent="react", session_id=req.session_id,
            )
        except Exception as e:
            _logger.warning(f"人工审批异常，按拒绝处理: {e}")
            approved, final_args = False, arguments
        if not approved:
            return ToolVerdict.block("approval_rejected", f"工具 {req.tool_name} 未获人工批准，已取消执行",
                                     action=ACTION_APPROVE)
        return ToolVerdict.ok(final_args, action=ACTION_APPROVE, note="人工已批准")

    # ------------------------------------------------------------------
    # ③ 沙箱执行
    # ------------------------------------------------------------------
    def _runner_for(self, mode: str):
        # 【先查缓存】同一个 mode 只建一次执行器：
        #   一是省掉重复构造的开销；
        #   二是 runner 是有状态对象（远程执行器的连接配置、子进程执行器的环境白名单等），
        #   每次新建会让"上一次的状态"凭空消失
        mode = str(mode or MODE_NONE)
        runner = self._runners.get(mode)
        if runner is None:
            kwargs: Dict[str, Any] = {"timeout_seconds": self.default_timeout}
            if mode == MODE_DOCKER:             # image/network/memory/cpus 只有容器沙箱认
                kwargs.update(self.docker_options)
            elif mode == MODE_REMOTE:            # endpoint/token 只有远程沙箱认
                kwargs.update(self.remote_options)
            #  ↑ update = 把"该模式专属参数"并进 kwargs，最后一次性 **kwargs 下发给 build_sandbox
            #    【为什么不合成一个全量大字典】三种 runner 的构造签名不同：
            #    把 image 传给 SubprocessRunner 会直接 TypeError: unexpected keyword argument，
            #    所以按 mode 分档下发（详见上方 97 行注释）
            #    【副作用】update 同名 key 会覆盖：docker_options 里若带了 timeout_seconds，
            #    会盖掉上面的 default_timeout —— 这是留给调用方单独调某个沙箱超时的口子
            runner: SandboxRunner = build_sandbox(mode, **kwargs)
            self._runners[mode] = runner          # 存缓存，下次同 mode 直接复用
        return runner

    def execute(self, tool: Any, params: dict, cap: Optional[ToolCapability] = None) -> ToolResult:
        """选工位并执行：工具在哪跑，由能力声明说了算，不由调用方说了算。

        【工位优先级】工具声明 cap.sandbox > 全局 default_sandbox；超时同理。
            none        进程内，零开销（只读工具）
            subprocess  本机子进程，超时可杀、环境白名单（有副作用）
            docker      容器，网络隔离 + 内存限制（不可信逻辑）
            remote      远程沙箱服务，工具不在本机跑（跨机隔离）
        """
        cap = cap or self.capability_of(tool)
        mode = cap.sandbox or self.default_sandbox                  # 工具没声明就用全局默认
        timeout = cap.timeout_seconds or self.default_timeout
        return self._runner_for(mode).run(tool, dict(params or {}), timeout)

    # ------------------------------------------------------------------
    # ⑤ 结果过滤 + 审计
    # ------------------------------------------------------------------
    def after_call(self, result: ToolResult, req: ToolCallRequest,
                   cap: Optional[ToolCapability] = None) -> ToolResult:
        """调用后处理：脱敏裁剪 → token 记账 → 审计。

        【为什么必须有这一段】工具返回什么 ≠ 模型看到什么：
            不裁剪 → 几千字回灌 prompt，几轮吃满上下文窗口
            不脱敏 → 工具按自己权限取回的明文（手机号/薪资）直接进 prompt
            不记账 → 预算形同虚设（预算卡的是"回灌的 token"，不是"调用的 token"）
        """
        cap = cap or req.capability
        limit = int(getattr(cap, "max_result_chars", 0) or 0)      # 工具可单独声明裁剪上限；0 = 用全局
        result = self.result_filter.filter(result, req.principal, max_chars=limit)
        try:
            from modules.common.token_counter import estimate_tokens
            tokens = estimate_tokens(str(result.result if result.success else result.error))
        except Exception:
            tokens = 0
        self.budget(req.session_id).consume(req.tool_name, tokens)   # token 记进预算账
        if self.audit_enabled:
            self._audit(req, result, cap, approved=True)
        return result

    # 【什么】落一条工具调用审计
    # 【作用】放行、拒绝、失败都记，detail 里带上风险等级/审批/沙箱/预算
    def _audit(self, req: ToolCallRequest, result: Optional[ToolResult],
               cap: Optional[ToolCapability], approved: bool) -> None:
        if req.principal is None:
            return                                     # 无登录态（CLI/单测）不落审计
        try:
            from modules.enterprise import audit
            success = bool(result and result.success)
            state = "allow" if (approved and success) else ("deny" if not approved else "error")
            audit.log_audit(
                getattr(req.principal, "user_code", ""),
                "tool_call",
                f"tool:{req.tool_name}",
                state,
                dept_code=getattr(req.principal, "dept_code", ""),
                session_id=req.session_id,
                detail={
                    "arguments": req.arguments,
                    "risk": getattr(cap, "risk", RiskLevel.LOW).value if cap else RiskLevel.LOW.value,
                    "sandbox": getattr(cap, "sandbox", MODE_NONE) if cap else MODE_NONE,
                    # 未获批准（被策略/中间件/人工拒绝）记 denied；需审批且已通过记 approved；其余记 none
                    "approval": ("denied" if not approved
                                 else ("approved" if (cap and cap.needs_approval) else "none")),
                    "iteration": req.iteration,
                    "duration_ms": getattr(result, "duration_ms", 0) if result else 0,
                    "error": (result.error if result and not result.success else ""),
                    "budget": self.budget(req.session_id).snapshot(),
                },
            )
        except Exception as e:
            _logger.warning(f"工具审计写入失败: {e}")

    # 【什么】记录一次被拦截的调用
    # 【作用】拦截也要留痕（审计 + 预算计数），否则"被拒的越权尝试"无从追查
    def record_intercepted(self, req: ToolCallRequest, cap: Optional[ToolCapability],
                           reason: str, code: str = "") -> None:
        self.budget(req.session_id).record_block()
        if self.audit_enabled:
            self._audit(req, ToolResult(success=False, error=reason), cap, approved=False)
        _logger.info(f"工具调用已拦截: {req.tool_name} | code={code} | {reason}")

    # ------------------------------------------------------------------
    # 一次完整调用
    # ------------------------------------------------------------------
    def call(self, tool: Any, params: dict, *, session_id: str = "", user_id: str = "",
             principal: Any = None, context: Optional[dict] = None, iteration: int = 0) -> ToolResult:
        """完整调用链：before_call → execute → after_call。

        【三个阶段各自干什么】
            before_call  决定"能不能调、用哪套参数调"（拦截在这里发生，真实服务收不到请求）
            execute      决定"在哪调"（进程内 / 子进程 / 容器 / 远程）
            after_call   决定"调完给模型看什么"（脱敏裁剪）+ 记账 + 审计

        Returns:
            ToolResult。被拦截时 success=False 且 error 是可直接回喂模型的中文原因。
        """
        # 【阶段 0】取能力台账：工具声明 → 配置覆盖 → 策略兜底，三级后的最终值
        cap = self.capability_of(tool)

        # 【阶段 0.5】打包成请求对象：中间件全程只读它，改参数只通过 verdict.arguments 输出
        req = ToolCallRequest(
            tool_name=getattr(tool, "name", ""), arguments=dict(params or {}), session_id=session_id,
            user_id=user_id, principal=principal, capability=cap, iteration=iteration,
            context=dict(context or {}),
        )

        # 【阶段 1】before_call：策略裁决 + 五道中间件 + 人工审批
        verdict = self.before_call(req)
        if not verdict.allowed:
            # 【拦截也要留痕的原因】"被拒的越权尝试"正是安全审计最该看到的东西，
            #   放行、拒绝、失败三种 state 都在审计里，才能还原一次完整的攻击/误用链路
            if self.audit_enabled:
                self._audit(req, ToolResult(success=False, error=verdict.reason), cap, approved=False)
            self.budget(session_id).record_block()
            # 【为什么 error 用中文原因】这是要回喂模型的文本，模型据此改参数或换工具，
            #   code 是给机器（前端/日志）用的，risk 让前端知道要不要提示"该操作需审批"
            return ToolResult(success=False, error=verdict.reason, meta={"code": verdict.code, "risk": cap.risk.value})

        # 【阶段 2】execute：参数可能被人工改过，所以用 verdict.arguments 而不是原始 params
        call_params = dict(verdict.arguments or {})
        if context:
            # 【为什么私有参数最后注入】_dept_code / _mask_level 这类是服务端按登录态算的，
            #   不出现在 schema 里，模型给不出来；后写入保证模型伪造的同名参数盖不住它
            call_params.update(context)
        result = self.execute(tool, call_params, cap)

        # 【为什么审计前把 arguments 换回原始入参】审计要看"模型到底想干什么"，
        #   注入项是服务端行为，混进去就看不出模型有没有试图伪造部门码
        req.arguments = dict(params or {})

        # 【阶段 3】after_call：脱敏裁剪 → token 记账 → 审计
        return self.after_call(result, req, cap)

    # 【什么】运行时快照
    # 【作用】观测：装了哪些中间件、各工具什么风险、各会话预算用了多少
    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "policy": {
                "default_risk": self.policy.default_risk.value,
                "deny_levels": list(self.policy.deny_levels),
                "approve_levels": list(self.policy.approve_levels),
                "deny_tools": list(self.policy.deny_tools),
                "approve_tools": list(self.policy.approve_tools),
                "on_no_gate": self.policy.on_no_gate,
            },
            "middlewares": self.chain.names(),
            "gate_enabled": self.gate is not None,
            "default_sandbox": self.default_sandbox,
            "docker_options": dict(self.docker_options),
            "capabilities": self.capabilities(),
            "budgets": self.budgets.stats(),
        }


# 【什么】按配置装配一个 Harness
# 【作用】server / main 一行接入；未启用时返回 None，registry 走原流程
def build_tool_harness(settings: Any = None, gate: Any = None) -> Optional[ToolHarness]:
    """按 Settings 装配 ToolHarness。

    【配置项 → 组件】
        TOOL_HARNESS_ENABLED      总开关；关掉返回 None（零行为变化）
        TOOL_DEFAULT_RISK/...     策略裁决
        TOOL_MAX_CALLS_PER_RUN... 预算
        TOOL_SANDBOX_MODE         默认隔离方式（工具可单独声明覆盖）
        TOOL_SANDBOX_IMAGE        容器沙箱镜像（mode=docker 时生效）
        TOOL_SANDBOX_REMOTE_URL   远程沙箱服务地址（mode=remote 时生效，空则降级子进程）
        TOOL_SANDBOX_REMOTE_TOKEN 远程沙箱鉴权令牌（可空）
        TOOL_RESULT_MAX_CHARS     结果裁剪上限
    """
    if settings is None:
        from config import get_settings
        settings = get_settings()
    if not bool(getattr(settings, "tool_harness_enabled", True)):
        return None

    policy = ToolPolicy.from_settings(settings)
    spec = BudgetSpec(
        max_calls_per_run=int(getattr(settings, "tool_max_calls_per_run", 8) or 8),
        max_calls_per_tool=int(getattr(settings, "tool_max_calls_per_tool", 5) or 5),
        token_budget=int(getattr(settings, "tool_token_budget", 6000) or 6000),
        timeout_seconds=float(getattr(settings, "tool_run_timeout_seconds", 120.0) or 120.0),
        max_result_chars=int(getattr(settings, "tool_result_max_chars", 2000) or 2000),
    )
    budgets = BudgetManager(spec)
    chain = MiddlewareChain([
        SchemaMiddleware(lambda name: _schema_of_default(name, None)),   # 占位，bind_registry 后生效
        PermissionMiddleware(),
        ScopeMiddleware(),
        RateLimitMiddleware(
            capacity=int(getattr(settings, "tool_rate_burst", 20) or 20),
            refill_per_sec=float(getattr(settings, "tool_rate_per_sec", 5.0) or 5.0),
            per_user_capacity=int(getattr(settings, "tool_rate_burst_per_user", 8) or 8),
            per_user_refill=float(getattr(settings, "tool_rate_per_user", 2.0) or 2.0),
        ),
        BudgetMiddleware(budgets),
    ])
    result_filter = ResultFilter(ResultFilterConfig(max_result_chars=spec.max_result_chars))
    return ToolHarness(
        policy=policy, budgets=budgets, chain=chain, result_filter=result_filter, gate=gate,
        enabled=True, audit_enabled=True,
        default_sandbox=str(getattr(settings, "tool_sandbox_mode", MODE_NONE) or MODE_NONE),
        default_timeout=float(getattr(settings, "tool_sandbox_timeout_seconds", 15.0) or 15.0),
        docker_options={
            "image": str(getattr(settings, "tool_sandbox_image", "") or "") or "python:3.11-slim",
        },
        remote_options={
            "endpoint": str(getattr(settings, "tool_sandbox_remote_url", "") or ""),
            "token": str(getattr(settings, "tool_sandbox_remote_token", "") or ""),
            "timeout_seconds": float(getattr(settings, "tool_sandbox_remote_timeout", 15.0) or 15.0),
        },
    )


def _schema_of_default(tool_name: str, _unused: Any = None) -> dict:
    """未绑定 registry 时的占位取 schema 函数（恒返回空 → 跳过 schema 校验）。"""
    return {}
