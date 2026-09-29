# 工具 Harness：让模型只有「发起申请」的权利

> Agent 负责想「调哪个工具」，Harness 负责「让不让调、在哪调、调完怎么处理」。
> 模型自始至终拿不到"直接执行高风险操作"的权限。

**端到端流程见 [`Harness执行流程.md`](./Harness执行流程.md)**（从提问到回灌，六阶段 + 拦截码 + 配置映射）。
本篇讲「每层是什么、为什么这么设计」。

> **别混名**：本篇讲**工具侧** Harness（单个工具让不让调、在哪调）。
> 多 Agent 的团队运行时是另一个同名文件 `modules/multiagent/harness.py`，
> 骨架见 [`多AgentHarness执行流程骨架.md`](./多AgentHarness执行流程骨架.md)。

---

## 一、为什么需要这一层

只靠"工具注册表 + ReAct 循环"时，有五件事没人管：

| 缺口 | 后果 |
|---|---|
| 工具只有名字和描述 | 删库、改生产、外发通知和算个数混在一起，模型一视同仁地调 |
| 参数由模型生成 | 幻觉出的参数名/越权的工具名直接下发到真实服务，副作用已经发生才报错 |
| 工具在进程内直接跑 | 工具逻辑出错、死循环，污染的是主进程与外部系统 |
| 循环只有轮次上限 | 同一个工具反复调、结果一遍遍回灌，上下文膨胀到爆 |
| 结果直接回灌 | 几千字全进 prompt、越权字段（手机号/薪资）以明文进上下文 |

Harness 就是补这五项的那层。

---

## 二、五项能力

### ① 工具注册与能力声明

工具在 `BaseTool.capability` 上声明自己"是什么"，Harness 据此裁决。三种写法等价：

```python
capability = {"risk": "high", "side_effect": True, "sandbox": "subprocess"}
capability = ToolCapability(name="send_notice", risk=RiskLevel.HIGH, side_effect=True)
def capability(self): return ToolCapability(...)      # 需要动态计算时
```

`ToolCapability` 字段（`modules/harness/models.py:65`）：

| 字段 | 含义 | 默认 |
|---|---|---|
| `risk` | 风险等级 | `low` |
| `side_effect` | 会写外部系统吗 | `False` |
| `sandbox` | 隔离方式 | `none` |
| `requires_approval` | 显式要求人工批准 | `False` |
| `permissions` | 需要的权限码，如 `("data:orders:all",)` | `()` |
| `data_scopes` | 涉及的业务数据类型，如 `("orders",)` | `()` |
| `max_result_chars` / `timeout_seconds` | 单独覆盖裁剪上限与超时 | `0`（用全局） |

四档风险等级：

| 等级 | 含义 | 典型工具 |
|---|---|---|
| `low` | 只读、无副作用 | calculator、data_query |
| `medium` | 有副作用但可回滚（MCP 远程工具默认这档） | web_search |
| `high` | 影响外部系统 | send_notice、下单 |
| `critical` | 不可逆或影响生产 | drop_table、改生产配置 |

策略裁决顺序（`ToolPolicy.decide`，`models.py:209`）——**名字优先于等级**：

```
deny_tools（黑名单，支持 drop_*） → deny_levels → approve_tools → approve_levels / requires_approval → allow
```

| 工具 | 等级 | 裁决 |
|---|---|---|
| `calculator` | low | `allow` 直接执行 |
| `send_notice` | high | `approve` 必须人工批准 |
| `drop_table` | critical | `deny` 禁止执行（模型连申请都不受理） |
| `drop_index` | medium + 黑名单 `drop_*` | `deny`（名字规则盖过等级） |

高风险且没启用 HITL 时按 `TOOL_ON_NO_GATE` 处理，默认 `deny`——宁可不做，也不能做错。

### ② 调用前中间件

模型输出 tool call 之后、下发之前，固定顺序过五道（`modules/harness/middleware.py`）：

| # | 中间件 | 拦什么 | 拦截码 |
|---|---|---|---|
| 1 | `SchemaMiddleware` | 必填缺失 / 幻觉出的未知参数 / 类型不对 | `schema_invalid` |
| 2 | `PermissionMiddleware` | `principal.can('tool:<name>')` 与能力声明要求的权限码 | `permission_denied` |
| 3 | `ScopeMiddleware` | 参数里的 `dept_code`/`user_code` 超出本人可见范围 | `scope_denied` |
| 4 | `RateLimitMiddleware` | 令牌桶（全局 + 每用户，复用 `orchestration.control.RateLimiter`） | `rate_limited` |
| 5 | `BudgetMiddleware` | 次数 / token / 墙钟超时 | `budget_exceeded` |

顺序是先便宜后昂贵：先判"参数对不对"，再判"你有没有资格"，最后才动用全局资源。
首个拦截即返回，`execute` 根本不会发生——**真实服务收不到这次请求**。

服务端注入的私有参数（`_data_scope` / `_dept_code` / `_mask_level`，下划线开头）不参与 schema 校验：它们不在 schema 里，模型给不出来，也伪造不了。

### ③ 沙箱隔离

四档，按工具声明选择（`modules/harness/sandbox.py`）：

| 模式 | 做法 | 适用 |
|---|---|---|
| `none` | 进程内直接 `tool.run()`（默认，零开销） | 只读工具 |
| `subprocess` | 独立子进程：超时可杀、环境变量只剩白名单 | 有副作用的工具 |
| `docker` | 容器：`--network=none` + 内存/CPU 限制 | 不可信逻辑 |
| `remote` | 远程沙箱服务：同一份类坐标 POST 给另一台机器上常驻的执行服务 | 想在本机之外跑（模拟实现见下） |

跨进程执行传的是"类坐标"，不是 pickled 对象：

```json
{"module": "modules.tools.calculator", "class": "CalculatorTool", "params": {"expression": "3*7"}}
```

子进程（`sandbox_worker.py`）按坐标 import 后 `run()`，结果以**一行 JSON** 回传；主进程取 stdout 里最后一个合法 JSON 行，工具自己的 `print` 不会污染结果。
docker 不可用时自动降级为子进程并在日志 warning——沙箱是加固项，不该成为可用性瓶颈。

#### 远程沙箱（`mode=remote`）

`RemoteSandboxRunner` 传的是**同一份类坐标**，只是把"管道"换成 HTTP：

```
主进程 ──POST {"module","class","params"}──▶ 远端 /run ──▶ 起 worker 子进程执行
      ◀──────── {"success","result","error","duration_ms"} ──
```

没有第二台机器也能跑通——项目自带一个可本地启动的模拟远端服务：

```bash
python main.py sandbox-server --host 127.0.0.1 --port 8899     # 远程部署时 --host 0.0.0.0
# 等价于 uvicorn modules.harness.remote_sandbox_server:app --host 127.0.0.1 --port 8899

# 客户端配置
TOOL_SANDBOX_MODE=remote
TOOL_SANDBOX_REMOTE_URL=http://127.0.0.1:8899/run
TOOL_SANDBOX_REMOTE_TOKEN=dev-token        # 可选，服务端设了 REMOTE_SANDBOX_TOKEN 才需要
```

它收到请求后在本机起一个 `sandbox_worker` 子进程执行，**协议与子进程沙箱完全一致**，换真远程时把这一层换成容器/沙箱池即可，调用方一行不用改。

三点约定：

| 情况 | 行为 |
|---|---|
| 远端连不上 / 超时 / 非 2xx | 默认降级本机子进程并 warning；`fallback=False` 时直接返回失败（宁可不跑，也不能在本机跑） |
| 工具在远端跑失败（`success=false`） | 不算服务故障，原样返回失败结果，不降级 |
| 没配 `TOOL_SANDBOX_REMOTE_URL` | 直接走本机子进程，连网络都不碰 |

> 为什么用标准库 `urllib` 而不是 requests/httpx：沙箱是基础组件，不该再挂一层网络依赖。

### ④ 执行预算与循环控制

`BudgetSpec`（`modules/harness/budget.py:34`）四个配额，任一打满即停：

| 配额 | 配置 | 默认 | 防什么 |
|---|---|---|---|
| `max_calls_per_run` | `TOOL_MAX_CALLS_PER_RUN` | 8 | 整体失控 |
| `max_calls_per_tool` | `TOOL_MAX_CALLS_PER_TOOL` | 5 | 死循环反复调同一个工具 |
| `token_budget` | `TOOL_TOKEN_BUDGET` | 6000 | 上下文膨胀 |
| `timeout_seconds` | `TOOL_RUN_TIMEOUT_SECONDS` | 120 | 卡死 |

账本按 `session_id` 隔离，`begin_run()` 开账、`end_run()` 销账；被拦截的调用**不消耗配额**（一次参数错误不该把额度浪费掉）。
ReAct 侧还会看这本账：超时或 token 打满时直接收尾出答案，不再问模型（`react_agent.py:329`）——这是上下文膨胀的最后一道防线。

### ⑤ 结果过滤 + 审计

`ResultFilter`（`result_filter.py`）三步：脱敏 → 裁剪 → 标注。

```
输入 result="abcdefghijklmnopqrstuvwxyz"  (max_result_chars=20)
输出 result="abcdefghijklmnopqrst…(结果已裁剪，共 26 字符，已保留前 20 字符)"

输入 result={"phone":"13812345678"}  principal.mask_level=2
输出 result='{"phone": "138****5678"}'    ← 复用 modules.enterprise.masking，不另造口径
```

裁剪标记是故意留的：模型看到它就知道"后面还有，但没给我"，不会误以为结果只有这么点。
`meta` 里写清楚 `trimmed` / `masked` / `chars`，便于排查。

审计（`ToolHarness._audit`）：有登录态时每次调用落一条 MySQL 审计，**放行、拒绝、失败都记**：

| detail 字段 | 内容 |
|---|---|
| `arguments` | 模型原始入参（不含服务端注入项） |
| `risk` / `sandbox` | 风险等级与执行环境 |
| `approval` | `none` / `approved` / `denied` |
| `iteration` / `duration_ms` / `error` | 轮次、耗时、失败原因 |
| `budget` | 当时预算快照 |

越权尝试正是安全审计最该看到的东西，所以被拦截的调用**也要留痕**。

---

## 三、一次调用的完整链路

```
模型输出 tool_call
      │
      ├─ ① 策略裁决      deny → 拦截（不进审批、不执行）
      │
      ├─ ② 五道中间件    schema → permission → scope → rate_limit → budget
      │                  任一不过 → 拦截 + 审计 + blocked+1
      │
      ├─ ③ 人工审批      high/critical 才问；批准才放行，人可以改参数
      │
      ├─ ④ 沙箱执行      none / subprocess / docker
      │
      └─ ⑤ 结果过滤      脱敏 → 裁剪 → token 记账 → 审计落库
```

---

## 四、配置项（`.env`）

| 配置 | 默认 | 作用 |
|---|---|---|
| `TOOL_HARNESS_ENABLED` | `true` | 总开关；`false` = 回到接入前的行为 |
| `TOOL_DEFAULT_RISK` | `low` | 工具未声明时的兜底等级 |
| `TOOL_RISK_OVERRIDES` | 空 | 按名字提级，如 `web_search:medium,mysql_exec:critical` |
| `TOOL_DENY_LEVELS` | `critical` | 这些等级一律禁止执行 |
| `TOOL_APPROVE_LEVELS` | `high,critical` | 这些等级需人工批准 |
| `TOOL_DENY_TOOLS` / `TOOL_APPROVE_TOOLS` | 空 | 名字黑白名单，支持 `drop_*` 前缀 |
| `TOOL_ON_NO_GATE` | `deny` | 需审批但未启用 HITL 时怎么办 |
| `TOOL_MAX_CALLS_PER_RUN` / `_PER_TOOL` | 8 / 5 | 调用次数上限 |
| `TOOL_TOKEN_BUDGET` | 6000 | 上下文 token 预算 |
| `TOOL_RUN_TIMEOUT_SECONDS` | 120 | 墙钟超时 |
| `TOOL_RESULT_MAX_CHARS` | 2000 | 结果裁剪上限 |
| `TOOL_SANDBOX_MODE` | `none` | 默认隔离方式 |
| `TOOL_SANDBOX_TIMEOUT_SECONDS` | 15 | 单次执行超时 |
| `TOOL_SANDBOX_IMAGE` | `python:3.11-slim` | 容器沙箱镜像（`mode=docker`） |
| `TOOL_SANDBOX_REMOTE_URL` | 空 | 远程沙箱 `/run` 地址（`mode=remote`），空则降级本机子进程 |
| `TOOL_SANDBOX_REMOTE_TOKEN` | 空 | 远程沙箱鉴权令牌（写到 `X-Sandbox-Token` 头） |
| `TOOL_SANDBOX_REMOTE_TIMEOUT` | 15 | 单次远程执行超时（秒） |
| `TOOL_RATE_PER_SEC` / `TOOL_RATE_BURST` | 5 / 20 | 全局令牌桶 |
| `TOOL_RATE_PER_USER` / `TOOL_RATE_BURST_PER_USER` | 2 / 8 | 单用户令牌桶 |

---

## 五、接入方式

生产（`modules/api/server.py:215`）与 CLI（`main.py:86`）同一套装配：

```python
from modules.harness import build_tool_harness
harness = build_tool_harness(s, gate=gate)        # 开关关闭 / 装配失败 → None
registry = ToolRegistry(harness=harness)
harness.bind_registry(registry)                   # 让 SchemaMiddleware 取到 JSON Schema
```

`gate` 只在开启 HITL 时注入；没开 HITL 时高风险工具按 `TOOL_ON_NO_GATE` 处理。

单测里不想接 registry 也行：拿不到 schema 时 `SchemaMiddleware` 自动跳过。

---

## 六、开关与降级

| 情况 | 行为 |
|---|---|
| `TOOL_HARNESS_ENABLED=false` | `build_tool_harness()` 返回 `None`，registry 走改造前流程，**零行为变化** |
| 装配抛异常 | 记 warning，`harness=None`，服务照常起来 |
| docker 不可用 | 容器沙箱降级为子进程 |
| 远程沙箱连不上 / 超时 / 非 2xx | 默认降级本机子进程并 warning；要求强隔离时可关掉降级（`fallback=False`）直接失败 |
| 中间件自身抛异常 | 按放行处理并记 warning，不让一个校验环节搞挂整条链路 |
| 审计写库失败 | 记 warning，不影响调用结果 |
| 无登录态（CLI/单测） | 不落审计，避免调试流量污染审计表 |

---

## 七、测试

`tests/test_tool_harness.py`：每个闸门都有一条"**被拦下**"的用例——闸门失效时系统看起来是正常的（照常返回答案），只有被拦截的那一侧能证明它真的生效。

| 分组 | 覆盖 |
|---|---|
| 能力声明与策略 | 工具声明生效、critical 禁止、high 无闸门被拒、黑名单盖过等级、配置提级 |
| 中间件 | 幻觉参数、必填缺失、越权、跨部门、突发限流 |
| 预算 | 单工具打满、整轮打满、超时、token 打满 |
| 人工审批 | 批准后执行、拒绝后不执行、禁止类连闸门都不进 |
| 沙箱 | 进程内、子进程、子进程失败不崩主进程、按模式构造 |
| 远程沙箱 | `tests/test_sandbox_remote.py`：真起一个 HTTP 服务跑通远程执行 / 工具报错 / 连不上降级 / 令牌鉴权 |
| 结果过滤 | 裁剪、脱敏、短结果不动 |
| 审计 | 放行记 allow、拦截记 deny、无登录态不记 |
| 配置装配 | 中间件顺序、预算与镜像来自配置、总开关关闭返回 None |

端到端演示：`python examples/tool_harness_demo.py`（五项能力各打印一遍实测结果）。

---

## 八、文件清单

| 文件 | 职责 |
|---|---|
| `modules/harness/models.py` | 能力声明、策略裁决、请求与裁决的数据契约 |
| `modules/harness/budget.py` | 执行预算：次数 / token / 墙钟超时 |
| `modules/harness/middleware.py` | 五道调用前中间件与中间件链 |
| `modules/harness/sandbox.py` | 沙箱执行：none / subprocess / docker |
| `modules/harness/sandbox_worker.py` | 子进程（容器）里真正跑工具的那一段 |
| `modules/harness/remote_sandbox_server.py` | 模拟远程沙箱服务（本机可起，给 `mode=remote` 当对端） |
| `docs/Harness执行流程.md` | 端到端执行流程梳理（六阶段 / 拦截码 / 配置映射 / 降级表） |
| `modules/harness/result_filter.py` | 结果裁剪 + 脱敏 |
| `modules/harness/harness.py` | 总装 `ToolHarness` + `build_tool_harness()` |
| `modules/tools/tool_registry.py` | 调用入口：串联裁决 → 执行 → 过滤 → 审计 |
| `tests/test_tool_harness.py` | 上述测试 |
| `examples/tool_harness_demo.py` | 端到端演示 |
