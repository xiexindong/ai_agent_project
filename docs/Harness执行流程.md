# Harness 执行流程（端到端梳理）

> 目标：把「模型说要调工具」到「结果回灌 prompt」之间发生的一切，按顺序讲清楚。
> 代码位置：`modules/harness/*`，串起来的地方是 `modules/tools/tool_registry.py` 的 `execute_tool`。

---

## 一句话

**Agent 负责想「调哪个工具」，Harness 负责定「让不让调、在哪调、调完给模型看什么」。**

模型自始至终只有**发起申请**的权利：禁止类工具它连申请都不被受理，高风险工具必须等人批准，参数/权限/范围/限流/预算任一不过就被拦在真实服务之外。

---

## 全景流程

```
用户提问
  │
  ▼
ReActAgent.prepare()                    开一本预算账 begin_run(session_id)
  │
  ▼
LLM → 生成 tool_call(name, arguments)
  │
  ▼
ToolRegistry.execute_tool()  ────────────────────────────────────────────────┐
  │                                                                          │
  │ ① get_tool(name)                   找不到 → "工具不存在"                  │
  │ ② capability_of(tool)              取能力台账（注册期已解析好）           │
  │ ③ harness.before_call(req)         ← 准入判定（唯一入口）                │
  │      ① policy.decide               deny  → 拦截 policy_denied            │
  │      ② 五道中间件                   任一不过 → 拦截（见拦截码表）         │
  │      ③ HumanGate（需审批时）        拒绝 → 拦截 approval_rejected         │
  │ ④ 单工具限次 / 同参去重（registry 自己的账）命中缓存 → 直接回传           │
  │ ⑤ harness.execute(tool, params)    按 cap.sandbox 选工位                 │
  │      none        进程内 tool.run()                                       │
  │      subprocess  本机 worker 子进程（stdin/stdout 传 JSON）              │
  │      docker      容器（网络隔离 + 内存限制）                              │
  │      remote      → POST 远端 /run → 远端起 worker → 回 JSON              │
  │ ⑥ harness.after_call(result)                                             │
  │      脱敏（mask_level）→ 裁剪（max_result_chars）→ token 记账 → 审计      │
  │ ⑦ 成功结果进 _result_cache                                               │
  └──────────────────────────────────────────────────────────────────────────┘
  │
  ▼
ToolResult 回灌 prompt → LLM 决定收尾 or 再调一次（回到 execute_tool）
  │
  ▼
新一轮提问：reset_session() → end_run() 销账，配额不被上一轮吃掉
```

---

## 六阶段总表

| # | 阶段 | 入口 | 干什么 | 不过的后果 |
|---|---|---|---|---|
| 0 | 注册期 | `ToolRegistry.register` → `ToolHarness.register` | 读能力声明，写进台账 | — |
| ① | 策略裁决 | `ToolPolicy.decide` | 禁止 / 要审批 / 放行 | `policy_denied` |
| ② | 中间件 | `MiddlewareChain.run` | 参数/权限/范围/限流/预算 | 5 种拦截码 |
| ③ | 人工审批 | `ToolHarness._ask_human` | 高风险问人，人可改参数 | `approval_required` / `approval_rejected` |
| ④ | 沙箱执行 | `ToolHarness.execute` | 按能力声明选工位 | 超时/崩溃被隔离在工位里 |
| ⑤ | 结果过滤 | `ResultFilter.filter` | 脱敏 + 裁剪 | — |
| ⑥ | 记账审计 | `RunBudget.consume` / `ToolHarness._audit` | token 记账 + 落审计 | — |

---

## 阶段 0：注册期（解析只做一次）

```
register(tool) → capability_of(tool) → policy.effective(cap) → _capabilities[name] = cap
```

能力三级来源，**后者压前者**：工具自己声明（`BaseTool.capability`）→ 配置覆盖（`TOOL_RISK_OVERRIDES`）→ 策略兜底（`TOOL_DEFAULT_RISK`）。

台账字段（`ToolCapability`）：

| 字段 | 含义 | 谁在读 |
|---|---|---|
| `risk` | low/medium/high/critical | 策略裁决（直接跑 / 要审批 / 禁止） |
| `side_effect` | 会不会写外部系统 | 记录与展示，不决定是否审批 |
| `sandbox` | none/subprocess/docker/remote | 执行阶段选工位 |
| `requires_approval` | 显式要求人工批准 | 策略裁决 |
| `permissions` | 需要的权限码 | PermissionMiddleware |
| `data_scopes` | 涉及的业务数据类型 | ScopeMiddleware |
| `max_result_chars` / `timeout_seconds` | 裁剪上限 / 超时 | ResultFilter / 执行器 |

---

## 阶段 ①：策略裁决（四行 if，顺序固定）

```
名字黑名单 deny_tools → 等级黑名单 deny_levels → 审批名单 approve_tools → 审批等级 approve_levels → allow
```

**名字优先于等级**：可以把某个 medium 工具单独提级为需审批，也可以把某个 high 工具加入禁止名单。

| 例子 | 裁决 |
|---|---|
| `calculator`（low） | allow，直接跑 |
| `send_notice`（high） | approve，需人工批准 |
| `drop_table`（critical） | deny，禁止执行 |
| `drop_index`（medium，名字命中 `drop_*`） | deny（名字盖过等级） |

---

## 阶段 ②：五道中间件（顺序即优先级）

| 序 | 中间件 | 拦什么 | 拦截码 |
|---|---|---|---|
| 1 | `SchemaMiddleware` | 缺必填 / 多未知参数 / 类型不对 | `schema_invalid` |
| 2 | `PermissionMiddleware` | 身份没有 `tool:<name>` 权限或缺少能力要求的权限码 | `permission_denied` |
| 3 | `ScopeMiddleware` | 参数里的 `dept_code` / `user_code` 越出可见范围 | `scope_denied` |
| 4 | `RateLimitMiddleware` | 短时间狂调（全局桶 + 每用户桶） | `rate_limited` |
| 5 | `BudgetMiddleware` | 次数 / token / 墙钟配额打满 | `budget_exceeded` |

**为什么是这个顺序**：先判"参数对不对"（最便宜、最常见），再判"你有没有资格"，最后才动用全局资源（配额与限流）。

两条例外规则：
- `principal is None`（CLI 调试 / 单测）→ 权限与范围直接放行，否则整条链路不可用
- 中间件自身抛异常 → 记 warning 按**放行**处理，一道校验的 bug 不该让所有工具瘫痪

---

## 阶段 ③：人工审批（高风险）

三条路径（`ToolHarness._ask_human`）：

| 情况 | 行为 |
|---|---|
| 有闸门 | `gate.ask_tool(...)` → 批准放行（人可改参数）/ 拒绝拦截 |
| 无闸门 + `TOOL_ON_NO_GATE=allow` | 记 warning 放行 |
| 无闸门 + `TOOL_ON_NO_GATE=deny`（默认） | 拦截 `approval_required`（**安全优先**：宁可不跑，也不让模型自己跑） |

审批异常按**拒绝**处理——人工通道是保险丝，坏了不能当它不存在。

---

## 阶段 ④：沙箱执行（工具在哪跑）

工具声明 `cap.sandbox` 优先，没声明用 `TOOL_SANDBOX_MODE`。

| 模式 | 怎么跑 | 适用 | 不可用会怎样 |
|---|---|---|---|
| `none` | 进程内 `tool.run()` | 只读工具 | — |
| `subprocess` | 本机 worker 子进程，stdin/stdout 传 JSON | 有副作用 | — |
| `docker` | 容器 `--network=none` + 内存限制 | 不可信逻辑 | 降级子进程 + warning |
| `remote` | POST 远端 `/run`，远端起 worker | 工具不在本机跑 | 降级子进程 + warning（`fallback=False` 时直接失败） |

**跨进程传的是"类坐标"，不是 pickled 对象**：

```json
{"module": "modules.tools.calculator", "class": "CalculatorTool", "params": {"expression": "3*7"}}
```

worker 按坐标 `import` 后 `run()`，结果以**一行 JSON** 回传；工具自己的 `print` 不污染结果行（主进程取最后一个以 `{` 开头的合法 JSON 行）。

远端服务怎么起（本机模拟）：`python main.py sandbox-server --port 8899`，客户端配 `TOOL_SANDBOX_MODE=remote` + `TOOL_SANDBOX_REMOTE_URL=http://127.0.0.1:8899/run`。远端服务与子进程沙箱是**同一份协议**，换真远程时调用方一行不用改。

---

## 阶段 ⑤：结果过滤（工具返回什么 ≠ 模型看到什么）

三步：脱敏（按 `principal.mask_level`）→ 裁剪（按 `max_result_chars`）→ 标注（`meta` 里写清楚被改过）。

| meta 字段 | 含义 |
|---|---|
| `trimmed` | 被裁剪过（文本末尾带 `…(结果已裁剪，共 N 字符，已保留前 M 字符)`，模型知道后面还有） |
| `masked` / `mask_level` | 被脱敏过 |
| `filtered` / `chars` | 过滤标记与最终字符数 |
| `error_trimmed` | 错误信息被裁剪（上限 500 字符） |

失败结果只裁错误信息，不动 result。

---

## 阶段 ⑥：记账与审计

**三个配额**（`BudgetSpec`）：`max_calls_per_run` 8、`max_calls_per_tool` 5、`token_budget` 6000、墙钟 `timeout_seconds` 120s。
被拦截的调用**不消耗**配额，只记 `blocked` 计数（观测用：blocked 高说明配额偏紧或模型在打转）。

**审计三种 state**：`allow`（放行且成功）/ `deny`（未获批准）/ `error`（放行但执行失败）。
放行、拒绝、失败**都记**——越权尝试正是安全审计最该看到的。detail 带：`risk` / `sandbox` / `approval`（denied|approved|none）/ `iteration` / `duration_ms` / `budget` 快照。
`principal is None`（CLI/单测）不落审计。

---

## 两本账：registry 去重 vs Harness 预算

| 账本 | 位置 | 管什么 |
|---|---|---|
| `_call_records` / `_call_counts` | `ToolRegistry` | 同一会话里别打转（同参去重 + 单工具限次） |
| `RunBudget` | `modules/harness/budget.py` | 这一轮总共能花多少（次数 / token / 墙钟） |

两本都过才放行，互不替代：去重挡"完全一样的重复调用"，预算挡"换着花样打转"。
成功结果进缓存后，重复调用**直接回传上次结果**（省一轮 LLM，也让模型有依据收尾，而不是误以为工具坏了）。

---

## 拦截码总表（前端 / 日志按这个分流）

| code | 来源 | 模型看到的提示 |
|---|---|---|
| `policy_denied` | 策略裁决 | 工具已配置为禁止执行 |
| `schema_invalid` | Schema | 缺少必填参数 / 出现未知参数 / 类型不对 |
| `permission_denied` | Permission | 当前身份无权调用该工具 |
| `scope_denied` | Scope | dept_code/user_code 超出可见范围 |
| `rate_limited` | RateLimit | 调用过于频繁，请稍后再试 |
| `budget_exceeded` | Budget | 次数/token/墙钟配额已用尽 |
| `approval_required` | 人工闸门缺失 | 需人工批准但未启用审批通道，已拒绝 |
| `approval_rejected` | 人工闸门 | 未获人工批准，已取消执行 |

被拦截时 `ToolResult.success=False`，`error` 是可直接回喂模型的中文原因，模型据此改参数或换工具。

---

## 配置项 → 组件

| 配置 | 默认 | 落到哪个组件 |
|---|---|---|
| `TOOL_HARNESS_ENABLED` | true | 总开关；关掉返回 `None`，行为与改造前一致 |
| `TOOL_DEFAULT_RISK` / `TOOL_RISK_OVERRIDES` | low / 空 | `ToolPolicy` 兜底与覆盖 |
| `TOOL_DENY_LEVELS` / `TOOL_APPROVE_LEVELS` | critical / high,critical | `ToolPolicy.decide` |
| `TOOL_DENY_TOOLS` / `TOOL_APPROVE_TOOLS` | 空 | 名字名单，支持 `drop_*` 前缀 |
| `TOOL_ON_NO_GATE` | deny | 需审批但无闸门时：deny / allow |
| `TOOL_MAX_CALLS_PER_RUN` / `_PER_TOOL` / `TOOL_TOKEN_BUDGET` / `TOOL_RUN_TIMEOUT_SECONDS` | 8 / 5 / 6000 / 120 | `BudgetSpec` |
| `TOOL_RESULT_MAX_CHARS` | 2000 | `ResultFilterConfig` |
| `TOOL_RATE_*`（4 个） | 20 / 5 / 8 / 2 | `RateLimitMiddleware`（全局桶 + 每用户桶） |
| `TOOL_SANDBOX_MODE` / `_TIMEOUT_SECONDS` / `_IMAGE` | none / 15 / python:3.11-slim | `build_sandbox` |
| `TOOL_SANDBOX_REMOTE_URL` / `_TOKEN` / `_TIMEOUT` | 空 / 空 / 15 | `RemoteSandboxRunner` |

---

## 远程链路在流程里的位置（两种"远程"，别混）

| 远程的东西 | 发生在哪 | 怎么开 |
|---|---|---|
| **远程沙箱**（工具在哪跑） | 阶段 ④ 执行 | `python main.py sandbox-server --port 8899` + `TOOL_SANDBOX_MODE=remote` |
| **远程 MCP**（工具从哪来） | 阶段 前的工具发现（目录/召回/注册） | `python main.py mcp-http-server --port 8800` + `MCP_REMOTE_URL` |

前者是"执行搬走"，后者是"工具从远端来"；两者都能用项目命令一键起，启动日志会打印 `传输={'xxx':'http'}` 与远程地址。

---

## 观测

- `ToolHarness.stats()`：开关 / 策略 / 中间件清单 / 闸门 / 默认沙箱 / 各工具能力 / 各会话预算
- 启动日志：`工具 Harness 已装配: 中间件=[...] | 闸门=开/关`，`工具沙箱: 默认模式=... | 远程地址=...`
- 每次拦截：warning `工具调用已拦截: <tool> | code=... | 原因`
- 审计表：`tool_call` 记录，detail 含风险/审批/沙箱/预算快照

---

## 降级与默认行为

| 情况 | 行为 |
|---|---|
| `TOOL_HARNESS_ENABLED=false` | `build_tool_harness()` 返回 `None`，registry 走原流程，**零行为变化** |
| 装配抛异常 | 同上，记 warning 后按无治理运行 |
| docker 不可用 | 降级子进程 + warning |
| 远程沙箱连不上/超时/非 2xx | 降级子进程 + warning（`fallback=False` 时直接失败） |
| 工具在远端跑失败（`success=false`） | 不算服务故障，原样返回失败结果，不降级 |

一句话：**Harness 是加固项，不是依赖项**——任何一环坏了，最坏的结果是"治理降级"，不是"不可用"。

---

## 文件索引

| 文件 | 职责 |
|---|---|
| `modules/harness/models.py` | 能力声明、策略裁决、请求/裁决的数据结构 |
| `modules/harness/budget.py` | 预算账本（次数 / token / 墙钟） |
| `modules/harness/middleware.py` | 五道调用前中间件 + 中间件链 |
| `modules/harness/sandbox.py` | 四种执行工位（none/subprocess/docker/remote） |
| `modules/harness/sandbox_worker.py` | 子进程（容器）里真正跑工具的那一段 |
| `modules/harness/remote_sandbox_server.py` | 模拟远程沙箱服务（本机可起） |
| `modules/harness/result_filter.py` | 结果脱敏 + 裁剪 |
| `modules/harness/harness.py` | 总装 `ToolHarness` + `build_tool_harness()` |
| `modules/tools/tool_registry.py` | 串起整条链的 `execute_tool` |
| `modules/mcp/mcp_http_server.py` / `mcp_http_client.py` | 远程 MCP（工具从远端来） |
