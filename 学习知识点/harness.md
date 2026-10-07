## 一句话

**一本台账 + 一条三段链**：注册期把工具能力写进台账；调用期固定走 `before_call → execute → after_call`，三段都只**读**台账。

> 完整端到端流程（含流程图、拦截码表、配置映射）见 `docs/Harness执行流程.md`。

---

## 骨架索引（先记住这张表）

| 阶段 | 入口代码 | 干什么 | 不过的后果 |
|---|---|---|---|
| 注册期 | `tool_registry.py:95` → `harness.py:210` | 写能力台账 | — |
| ① 裁决 | `harness.py:248` → `models.py:211` | 禁止 / 要审批 / 放行 | `policy_denied` |
| ② 中间件 | `harness.py:298` → `middleware.py:302` | 参数/权限/范围/限流/预算 | 5 种拦截码 |
| ③ 人工审批 | `harness.py:302` → `harness.py:311` | 高风险问人 | `approval_required` / `approval_rejected` |
| ④ 沙箱执行 | `harness.py:360` → `sandbox.py:205` | 选工位执行 | 超时/崩溃隔离在沙箱里 |
| ⑤ 结果过滤 | `harness.py:401` → `result_filter.py:67` | 脱敏 + 裁剪 | — |
| ⑥ 记账审计 | `harness.py:419` → `budget.py:91` / `harness.py:426` | token 记账 + 落审计 | — |

串起这六步的是 `tool_registry.py:226` 的 `execute_tool`。

---

## 阶段 0：注册期 —— 写台账

```
ToolRegistry.register(tool)                  tool_registry.py:73
    └─ self.harness.register(tool)           tool_registry.py:88
        └─ capability_of(tool)               harness.py:141   读工具的 capability 声明
            └─ policy.effective(cap)         models.py:190    配置覆盖压过工具声明
                └─ _capabilities[name]=cap   harness.py:215   台账落定
```

能力从哪儿读（`harness.py:141`，三种写法都认）：

```141:184:modules/harness/harness.py
    def capability_of(self, tool: Any) -> ToolCapability:
        raw = getattr(tool, "capability", None)
        if callable(raw):
            try:
                raw = raw()
```

注册时就定死裁决结果（`models.py:211`），调用期不再重复解析：

```211:242:modules/harness/models.py
    def decide(self, cap: ToolCapability) -> ToolDecision:
        name = cap.name
        if self._match(name, self.deny_tools):
            return ToolDecision(ACTION_DENY, f"工具 {name} 已配置为禁止执行", cap.risk.value)
        if str(cap.risk.value) in self.deny_levels:
            return ToolDecision(ACTION_DENY, f"工具 {name} 风险等级为 {cap.risk.value}，已配置为禁止执行", cap.risk.value)
        if self._match(name, self.approve_tools):
            return ToolDecision(ACTION_APPROVE, f"工具 {name} 已配置为需人工批准", cap.risk.value)
        if str(cap.risk.value) in self.approve_levels or cap.requires_approval:
            return ToolDecision(ACTION_APPROVE, f"工具 {name} 风险等级为 {cap.risk.value}，需人工批准", cap.risk.value)
        return ToolDecision(ACTION_ALLOW, "", cap.risk.value)
```

四行 if 就是全部裁决逻辑，顺序固定：**名字黑名单 → 等级黑名单 → 审批名单 → 等级 → allow**。

---

## 阶段 1：`before_call` —— 三道闸门（整个骨架的核心）

```248:294:modules/harness/harness.py
    def before_call(self, req: ToolCallRequest) -> ToolVerdict:
        """调用前校验：不过就是不过，真实服务不会收到请求。"""
        if not self.enabled:
            return ToolVerdict.ok(req.arguments)
        cap = req.capability or self._capabilities.get(req.tool_name) or ToolCapability(name=req.tool_name)

        decision = self.policy.decide(cap)
        if decision.denied:                                     # ① 配置层面禁止执行
            _logger.warning(f"工具被策略禁止: {req.tool_name} | {decision.reason}")
            return ToolVerdict.block("policy_denied", decision.reason, action=ACTION_DENY)

        verdict = self.chain.run(req)                           # ② 五道中间件
        if not verdict.allowed:
            return verdict

        if decision.action == ACTION_APPROVE:                   # ③ 高风险：人工批准
            return self._ask_human(req, verdict.arguments, decision.reason)
        return verdict
```

### ② 五道中间件：一行循环跑完

```302:316:modules/harness/middleware.py
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
```

`for` 顺序 = `middlewares` 列表顺序 = 优先级。五道各自的判定代码：

| # | 类（行号） | 判定的那几行 | 拦截码 |
|---|---|---|---|
| 1 | `SchemaMiddleware` `middleware.py:62` | `79` `_validate`：必填缺失 / 未知参数 / 类型 | `schema_invalid` |
| 2 | `PermissionMiddleware` `middleware.py:132` | `149` `can(f"tool:{name}")` + `152` 能力要求的权限码 | `permission_denied` |
| 3 | `ScopeMiddleware` `middleware.py:160` | `196` `want != mine`（dept）/ `201`（user） | `scope_denied` |
| 4 | `RateLimitMiddleware` `middleware.py:208` | `245` `limiter.acquire(sender, 1)` | `rate_limited` |
| 5 | `BudgetMiddleware` `middleware.py:256` | `274` `budget.check(tool_name)` | `budget_exceeded` |

### ③ 人工审批：模型只能"申请"

```301:308:modules/harness/harness.py
    def _ask_human(self, req: ToolCallRequest, arguments: dict, reason: str) -> ToolVerdict:
        if self.gate is None:
            if str(self.policy.on_no_gate).lower() == "allow":
                _logger.warning(f"工具 {req.tool_name} 需人工批准但未启用闸门，按配置放行")
                return ToolVerdict.ok(arguments, action=ACTION_APPROVE, note="需人工批准但未启用闸门")
            return ToolVerdict.block(
                "approval_required",
```

关键点：`gate is None` 时默认 `deny`——**没装审批通道 = 高风险工具一律不执行**，而不是"没人审就放行"。

---

## 阶段 2：`execute` —— 按台账选工位

```350:354:modules/harness/harness.py
    def execute(self, tool: Any, params: dict, cap: Optional[ToolCapability] = None) -> ToolResult:
        cap = cap or self.capability_of(tool)
        mode = cap.sandbox or self.default_sandbox
        timeout = cap.timeout_seconds or self.default_timeout
        return self._runner_for(mode).run(tool, dict(params or {}), timeout)
```

`_runner_for`（`harness.py:337`）按 mode 取执行器，`build_sandbox`（`sandbox.py`）四选一：

| mode | 类 | 关键代码 | 隔离手段 |
|---|---|---|---|
| `none` | `sandbox.py:67` | `70` 直接 `tool.run()` | 无（零开销） |
| `subprocess` | `sandbox.py:76` | `101` 环境变量白名单 / `125` `subprocess.run(timeout=)` | 超时可杀、崩了不影响主进程 |
| `docker` | `sandbox.py:181` | `244` `--network none --memory` / `217` 不可用时 `super().run()` 降级 | 网络与资源隔离 |
| `remote` | `RemoteSandboxRunner` | `_post()` 把同一份类坐标 POST 到远端 `/run`，失败降级 `self._local.run()` | 执行搬到另一台机器（模拟对端 `remote_sandbox_server.py`） |

跨进程传的是**类坐标**不是对象（`sandbox.py:108`），子进程 `sandbox_worker.py:44` 按坐标 import 后 `run()`，结果一行 JSON 回传。

---

## 阶段 3：`after_call` —— 过滤 + 记账 + 留痕

```391:406:modules/harness/harness.py
    def after_call(self, result: ToolResult, req: ToolCallRequest,
                   cap: Optional[ToolCapability] = None) -> ToolResult:
        cap = cap or req.capability
        limit = int(getattr(cap, "max_result_chars", 0) or 0)
        result = self.result_filter.filter(result, req.principal, max_chars=limit)
        try:
            from modules.common.token_counter import estimate_tokens
            tokens = estimate_tokens(str(result.result if result.success else result.error))
        except Exception:
            tokens = 0
        self.budget(req.session_id).consume(req.tool_name, tokens)   # token 记进预算账
        if self.audit_enabled:
            self._audit(req, result, cap)
```

三步各自的代码落点：

| 步 | 代码 | 逻辑 |
|---|---|---|
| 脱敏 | `result_filter.py:93-98` | `mask_level > 0` 且结果是 dict/list → `mask_record()`（复用企业模块，不另造口径） |
| 裁剪 | `result_filter.py:102-105` | 超长就切到 `max_result_chars` 并追加"已裁剪"提示 |
| 记账 | `budget.py:91` `consume()` | `total_calls += 1`、`per_tool[name] += 1`、`tokens += n` |
| 审计 | `harness.py:426` `_audit()` | 写 MySQL；`approval` 记 `none`/`approved`/`denied` |

---

## 谁把三段串起来：`ToolRegistry.execute_tool`

```317:336:modules/tools/tool_registry.py
        cap = self.harness.capability_of(tool) if self.harness is not None else None
        req = ToolCallRequest(
            tool_name=name, arguments=dict(params or {}), session_id=session_id, user_id=user_id,
            principal=principal, capability=cap, iteration=iteration, context=dict(context or {}),
        )
        if self.harness is not None:
            verdict = self.harness.before_call(req)
            if not verdict.allowed:
                self.harness.record_intercepted(req, cap, verdict.reason, verdict.code)
                return ToolResult(success=False, error=verdict.reason,
                    meta={"code": verdict.code, "risk": cap.risk.value if cap else ""},
                )
            params = verdict.arguments          # 人工批准时可能改过参数（modify）
```

```374:387:modules/tools/tool_registry.py
        if self.harness is not None:
            result = self.harness.execute(tool, call_params, cap)
        else:
            result = tool.run(call_params)

        self._call_counts[name] += 1   # 只在真正发起执行后计数：失败调用不占配额
        if session_id:
            self._call_records[session_id].add(dedup_key)

        # ③ 结果过滤（裁剪 + 脱敏）与审计；过滤后再缓存，保证重复调用回传的是同一份文本
        if self.harness is not None:
            result = self.harness.after_call(result, req, cap)
```

`self.harness is not None` 这层判断出现了 3 次——这就是"未装配时零行为变化"的实现方式。

---

## 反例：骨架换个顺序会怎样

| 改动 | 后果 |
|---|---|
| 把 `policy.decide` 挪到 `chain.run` 之后 | critical 工具先被参数校验放过后才判禁止——虽然仍拦得住，但白跑五道中间件；更糟的是如果未来加了"有副作用"的中间件，就会在禁止工具上产生副作用 |
| 先 `[:k]` 截断再判阈值（同参去重的类比） | 高分项被截掉后永久丢失 |
| `consume()` 放在 `before_call` 里 | 一次幻觉参数就把额度打满，模型后面正常调用也调不动 |
| `gate is None` 时放行 | 没装审批通道 = 所有 high/critical 工具裸奔，等于能力声明形同虚设 |

---

## 简单总结

1. **台账**：`ToolCapability`（`models.py:64`）+ `ToolPolicy`（`models.py:123`）+ `RunBudget`（`budget.py:44`），注册期写定、调用期只读。
2. **三段链**：`before_call`（`harness.py:248`）= 裁决 + 5 道中间件 + 人工审批；`execute`（`harness.py:360`）= 按 `cap.sandbox` 选执行器；`after_call`（`harness.py:401`）= 脱敏裁剪 + token 记账 + 审计。
3. **串联点**：`tool_registry.py:327/374/386`，三处 `harness is not None` 决定"装不装都可用"。
4. **拦截码是关节**：`policy_denied`、`schema_invalid`、`permission_denied`、`scope_denied`、`rate_limited`、`budget_exceeded`、`approval_required`、`approval_rejected`——每个码对应一处代码 + 一条"被拦下"的测试。
5. **降级写死在代码里**：`harness.py:277`（enabled=False 直接放行）、`middleware.py:307`（中间件报错按放行）、`sandbox.py:217`（docker 降级子进程）、`harness.py:312`（无闸门默认 deny）。

要我接着把「五道中间件的判定顺序为什么是这样」按这同样的方式——每道给出代码行号 + 反例——拆开讲吗？