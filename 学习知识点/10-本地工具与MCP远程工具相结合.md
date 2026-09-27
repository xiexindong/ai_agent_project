# 10 · 本地工具与 MCP 远程工具：为什么两个都要

> 对应代码：`main.py:89-117`（装配）、`modules/tools/tool_registry.py`、`modules/mcp/mcp_client.py:203-251`、`modules/agent/react_agent.py:187-192`

**一句话（专业版）**：`ToolRegistry` 是一个注册表，被装配阶段**装填两次**——先 `register_batch(get_default_tools())` 装本地进程内工具，再逐个 `register(MCPToolAdapter(...))` 装 MCP 子进程工具的适配器；两者都实现 `BaseTool` 接口，因此 `Agent` 与 `ToolRegistry` 完全不感知工具的物理位置。

**一句话（人话版）**：公司里有**正式员工**（坐你工位旁边，随叫随到）也有**外包团队**（在隔壁楼，只能发工单）。这两拨人**登记在同一本通讯录里**，格式一模一样。调度员（LLM）翻通讯录派活，**他根本不知道也无需知道这活是内部干还是外包干**。

> 与 [06-MCP接入全流程](./06-MCP接入全流程.md) 的分工：06 讲**MCP 这一条线怎么接**（拉起→握手→发现→适配→调用）；本文讲**两条线怎么并进同一张表、为什么要这么并，以及本项目在这个设计上目前有哪 3 处没走通**。

---

## 0. 先讲人话：沿用 06 的「外包」比喻

| 现实世界 | 代码里的东西 |
|---|---|
| 正式员工（工位就在你旁边） | 本地工具 `CalculatorTool` / `DataQueryTool` |
| 外包团队（隔壁楼，发工单联系） | MCP 子进程 `mcp_server.py` |
| **一本**通讯录，两种人都登记在上面 | `ToolRegistry._tools`（同一个 dict） |
| 派遣函（格式统一，不区分内外） | `BaseTool` 接口（`name` / `tool_type` / `run()`） |
| 外包员工在通讯录里的"对接人" | `MCPToolAdapter`（适配器，是个壳，不是真人） |
| 调度员只看通讯录派活 | LLM / Function Calling（`get_openai_tools_schema()`） |
| 外包集体失联 → 正式员工顶上 | `try/except` 降级（`main.py:114-117`） |

---

## 1. 两段装配代码：看着像，其实不同

```python
# 第②步：本地工具（main.py:91-92）
tool_registry: ToolRegistry = ToolRegistry()
tool_registry.register_batch(tool_registry.get_default_tools())
```

```python
# 第④步：MCP 远程工具（main.py:111-113，外层包在 try 里）
for tool_info in mcp.discover_tools():
    tool_registry.register(MCPToolAdapter(mcp, tool_info))
_logger.info(f"MCP工具已接入: {len(tool_registry.list_tools())} 个工具")
```

| 对比项 | 第②步（本地） | 第④步（MCP 远程） |
|---|---|---|
| 工具从哪来 | `get_default_tools()`：Python **直接 new** 实例 | `discover_tools()`：给子进程发 JSON-RPC `tools/list` 问回来的 **dict** |
| 注册方式 | `register_batch()` **批量**一次搞定 | `register()` **循环逐个** |
| 数量 | **固定 2 个**：`calculator`、`data_query` | **不定**，由 MCP Server 决定；当前 3 个 |
| 注册进去的是什么 | 真工具实例 | `MCPToolAdapter`——**只持有 `mcp` 客户端和工具说明书的壳** |
| 跨进程吗 | ❌ 进程内直调，无 IPC 开销 | ✅ 每次调用 = 写一行 JSON 给 stdin、读回一行 |
| 失败怎么办 | **没有 try**，挂了直接崩 | ✅ 整段 `try/except`，挂了只 `warning` |
| 延迟 import 的动机 | 避免**循环依赖**（`tool_registry` ↔ `calculator`） | 避免**缺依赖时启动崩溃** |

### 注册进去之后，通讯录长这样

| 阶段 | `tool_registry._tools` 内容 | 数量 |
|---|---|---|
| 第②步之前 | `{}` | 0 |
| 第②步之后 | `calculator` → `CalculatorTool`<br>`data_query` → `DataQueryTool` | 2 |
| 第④步之后 | 上面 2 个 **+**<br>`web_search` → `MCPToolAdapter`<br>`get_current_time` → `MCPToolAdapter`<br>`knowledge_stats` → `MCPToolAdapter` | **5** |

> MCP 那 3 个工具定义在 `modules/mcp/mcp_server.py:42-78`。

---

## 2. 相结合的 8 个好处（每条都对得上代码）

| # | 好处 | 代码依据 | 大白话 |
|---|---|---|---|
| 1 | **崩溃隔离** | `mcp_server.py:216` `except Exception: _logger.error` | `requests` 炸了、返回脏数据，只炸子进程，主进程照跑 |
| 2 | **超时隔离** | `mcp_client.py:167` `timeout: float = 60.0`<br>`mcp_server.py:102` `timeout=15` | 网络慢只卡子进程，Agent 主线程不会被拖 60 秒 |
| 3 | **降级保底** | `main.py:114-117` → `MCP降级（本地工具仍可用）` | MCP 全挂，Agent 还能算数、查库。**反之若全放远程，一挂全残** |
| 4 | **接口统一、可互换** | `MCPToolAdapter(BaseTool)`、`main.py:112` | 哪天把 `get_current_time` 挪到本地，Agent 代码**零改动**（部署形态可变，调用方不变） |
| 5 | **语言 / 依赖无关** | `main.py:107` `server_command=[sys.executable, "-m", ...]` | 换成 `["node", "server.js"]` 就是 Node 工具；Python 包版本冲突也隔离了 |
| 6 | **权限分层** | `tool_registry.py:111` `if user_level == "normal" and tool.tool_type == "mcp": continue` | 本地 = 可信基础能力，远程 = 需授权能力，天然一条权限线 |
| 7 | **性能分层** | `calculator` 走 AST 本地求值（微秒级）<br>`web_search` 发 HTTPS 到 Tavily（秒级） | 快的别付 IPC 开销，慢的别阻塞主线程 |
| 8 | **可选依赖** | `mcp_server.py:90-91` 没配 `TAVILY_API_KEY` 就返回文案 | 没 key、没装 `requests`，项目照样能启动 |

### 同一个提问，两种架构的结局

用户问：**「帮我算 12*8，再查一下今天上海天气」**

| | 全部写本地 | 本地 + MCP（本项目） |
|---|---|---|
| 算 12*8 | 本地执行，~0ms | 本地执行，~0ms |
| 查天气 | 主线程发 HTTP，**阻塞 ~2s**；异常直接冲进 Agent 调用栈 | 子进程发 HTTP（15s 超时），主进程继续；失败只影响这一条 |
| 网络库挂了 | **整个 Agent 崩** | 一行 warning，Agent 还能算数 |
| 想换成 Go 写的搜索 | 得改 Python 代码 | 改 `server_command` 一行 |

---

## 3. 为什么能「无感」：适配器模式

`MCPToolAdapter` 继承 `BaseTool` 后，只需实现 `execute()`，把调用翻译成一次 `tools/call`：

```python
# modules/mcp/mcp_client.py:245-251
def execute(self, **kwargs) -> ToolResult:
    result = self._client.call_tool(self.name, kwargs)   # → 写一行 JSON 给子进程
```

于是 `ToolRegistry` 的 3 本账本对本地 / 远程**一视同仁**：

| 账本 | 作用 | 位置 |
|---|---|---|
| `_tools` | 工具名 → 工具实例 | `tool_registry.py:39` |
| `_call_records` | 「工具名 + 参数指纹」集合，防死循环（边界 3） | `tool_registry.py:42` |
| `_call_counts` | 单工具累计调用次数，上限 5（限频） | `tool_registry.py:44-45` |

**这就是「相结合」最大的收益：调用方不关心工具在哪，工具在哪也就可以随便改。**

---

## 4. 本项目现状：设计是对的，但有 3 处没走通

### 问题 ① 日志数的是「总数」，不是「MCP 工具数」

```python
# main.py:113
_logger.info(f"MCP工具已接入: {len(tool_registry.list_tools())} 个工具")
```

`list_tools()` 返回的是 `self._tools.values()`（**全部 5 个**），所以日志会打：

```
MCP工具已接入: 5 个工具     ← 实际 MCP 只有 3 个，另外 2 个是本地的
```

更糟的情况：`discover_tools()` 返回 `[]`（MCP 连上了但没工具）时，只要没抛异常，日志照样打 `MCP工具已接入: 2 个工具`——**看着像成功了**。

### 问题 ② MCP 工具与本地工具同名会「静默覆盖」

```python
# modules/tools/tool_registry.py:58-60
if tool.name in self._tools:
    _logger.warning(f"工具重复注册，覆盖: {tool.name}")
self._tools[tool.name] = tool          # 只 warning，后者覆盖前者
```

当前 3 个 MCP 工具名（`web_search` 等）与本地的 `calculator` / `data_query` 不冲突，所以没暴露。
但哪天 MCP Server 也提供一个 `calculator`，**本地那个 AST 安全计算器就被顶掉**，Agent 会跑去子进程里算 `2+3`。

### 问题 ③（最严重）`user_level` 没透传 → MCP 工具永远不可见

权限过滤规则写在：

```python
# modules/tools/tool_registry.py:109-113
for tool in self._tools.values():
    # 权限检查：普通用户跳过 MCP 远程工具（仅保留本地基础工具）
    if user_level == "normal" and tool.tool_type == "mcp":
        continue
    schemas.append(tool.to_openai_format())
```

而 `MCPToolAdapter.tool_type` **恒等于 `"mcp"`**（`mcp_client.py:224-226`），函数的 `user_level` 默认值又恰好是 `"normal"`（`tool_registry.py:94`）。

调用方却是这么写的：

```python
# modules/agent/react_agent.py:190
tools_schema = self.tool_registry.get_openai_tools_schema()   # ← 没传 user_level
```

**没传 → 用默认 `normal` → 3 个 MCP 工具全被过滤。**

而 `perceive()` 明明收了这个参数（`react_agent.py:75` `user_level: str = "normal"`），就是没往下传。

**后果**：`main.py` 费劲注册了 5 个工具，日志还欢快地打「MCP工具已接入: 5 个工具」，但发给 DeepSeek 的 schema **只有 2 个**——模型**根本不知道**有 `web_search`。用户问「今天上海天气」，Agent 只会回「我没有联网能力」。

> 顺带说明：这也解释了问题 ① 为什么一直没人发现——**MCP 工具从来没真正被调用过**，自然没人去核对那行日志数的对不对。

---

## 5. 修复建议（都是纯改动，不动架构）

### 修 ③：透传 `user_level`（1 行，让 MCP 真正可用）

```diff
# modules/agent/react_agent.py:190
- tools_schema = self.tool_registry.get_openai_tools_schema()
+ tools_schema = self.tool_registry.get_openai_tools_schema(user_level=user_level)
```

### 修 ①：日志记真实数量

```diff
# main.py:111-113
- for tool_info in mcp.discover_tools():
-     tool_registry.register(MCPToolAdapter(mcp, tool_info))
- _logger.info(f"MCP工具已接入: {len(tool_registry.list_tools())} 个工具")
+ mcp_tools = mcp.discover_tools()
+ for tool_info in mcp_tools:
+     tool_registry.register(MCPToolAdapter(mcp, tool_info))
+ _logger.info(
+     f"MCP工具已接入: {len(mcp_tools)} 个（工具总数 {len(tool_registry.list_tools())}）"
+ )
```

### 修 ②：MCP 与本地重名时不覆盖（二选一）

```python
# 方案 A：跳过（保本地优先）
for tool_info in mcp_tools:
    if tool_info["name"] in tool_registry.list_tool_names():
        _logger.warning(f"MCP 工具与本地重名，跳过: {tool_info['name']}")
        continue
    tool_registry.register(MCPToolAdapter(mcp, tool_info))

# 方案 B：加前缀（两个都留，模型可分辨）
adapter = MCPToolAdapter(mcp, tool_info)
if tool_info["name"] in tool_registry.list_tool_names():
    adapter.name = f"mcp_{tool_info['name']}"
tool_registry.register(adapter)
```

> 方案 A 更安全（本地工具是硬依赖，不该被外部顶掉），方案 B 更灵活。本项目建议 A。

---

## 6. 一张表总结

| | 状态 |
|---|---|
| 架构本身（隔离 / 降级 / 可替换 / 性能 / 可选依赖） | ✅ **已生效**，适配器 + 统一注册表的思路是对的 |
| 崩溃隔离、超时隔离、依赖隔离 | ✅ 子进程 + `timeout=60` / `timeout=15` |
| 降级保底 | ✅ `main.py:114-117`，本地 2 个工具始终可用 |
| 权限分层 | ❌ **反向生效**：门槛写好了但没人传等级 → 所有用户都是 `normal` → **MCP 永远不可见** |
| 注册日志 | ⚠️ 数的是总数，会误导 |
| 重名策略 | ⚠️ 静默覆盖，暂无冲突但机制上有隐患 |

**一句话记住**：本地保底、MCP 扩展、适配器抹平——**思路满分，差在 `user_level` 那一行没传下去**。

---

## 7. 真·远程 MCP：HTTP 传输（新增）

前面讲的 MCP 走的是 **stdio**：`subprocess.Popen` 在本机拉起一个 Server 子进程。
那解决的是**进程隔离**，不是远程——Server 还跑在本机。

要"远程机器部署 MCP Server、Agent 当 Client 调它"，把传输层换成 HTTP 即可，
**协议一行不用改**（还是 initialize / tools/list / tools/call，还是 JSON-RPC 2.0）：

```
Agent ──POST /mcp {"jsonrpc":"2.0","method":"tools/call",...}──▶ 远端常驻 MCP Server
     ◀──── {"jsonrpc":"2.0","result":{"content":[{"type":"text","text":"..."}]}} ──
```

### 怎么用

```bash
# 远端机器上（本机模拟就本机跑）
python main.py mcp-http-server --host 127.0.0.1 --port 8800    # 远程部署时 --host 0.0.0.0

# Agent 侧，两种配置二选一
MCP_REMOTE_URL=http://10.0.0.7:8800/mcp
# 或写全
MCP_SERVERS_JSON='[{"name":"remote","transport":"http","endpoint":"http://10.0.0.7:8800/mcp"}]'
```

### 新增的三处代码

| 位置 | 作用 |
|---|---|
| `modules/mcp/mcp_http_client.py` | `MCPHttpClient`：与 `MCPClient` **方法名完全一致**，换传输不改上层 |
| `modules/mcp/mcp_http_server.py` | 模拟远程 Server：把项目自带的 `MCPServer` 挂到 HTTP 上（复用同一份工具与协议处理） |
| `modules/mcp/gateway.py` | `MCPServerConfig` 新增 `transport/endpoint/token`；`_new_client()` 按传输选客户端 |

因为适配器只要求"有 `call_tool`"，`MCPToolAdapter` 接哪个 client 都行：

```python
MCPToolAdapter(MCPClient(...),   info)   # 本机子进程
MCPToolAdapter(MCPHttpClient(...), info) # 远程机器
```

对 `ToolRegistry` 和 ReAct 来说两者完全等价 —— 这就是「适配器抹平」的价值。

### 降级与鉴权

| 情况 | 行为 |
|---|---|
| 远端连不上 / 401 / 500 | 抛 `MCPException`，网关接住后**该 server 不可用**，其余 server 与本地工具照常 |
| 目录建不出来 | `catalog()` 返回空目录，不抛异常（局部故障局部降级） |
| 服务端设了 `MCP_SERVER_TOKEN` | 客户端必须带 `Authorization: Bearer <token>`，否则 401 |
| 工具业务性失败 | 走协议 `isError`，不算 server 故障，不触发熔断 |

### 和「远程沙箱 runner」怎么选

| | 远程沙箱 `mode=remote` | 远程 MCP（HTTP） |
|---|---|---|
| 传什么 | 类坐标 `{"module","class","params"}` | MCP 协议（工具名 + 参数，元数据/错误码协议定好） |
| 对面是什么 | 自己写的执行服务 | 任意语言实现的 MCP Server（生态现成的都能接） |
| 适合 | 把自己的本地工具搬到别处跑 | 接外部/第三方工具服务 |
| 协议成本 | 自研（小） | 0（用标准协议，跨语言跨机器通用） |
