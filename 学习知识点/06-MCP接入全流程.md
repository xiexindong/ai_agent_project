# 06 · MCP 接入全流程

> 对应代码：`main.py:104-117`（装配）、`modules/mcp/mcp_client.py`、`modules/mcp/mcp_server.py`、`modules/mcp/protocol.py`
> 一句话：**用 `sys.executable -m` 拉起一个独立 Python 进程，通过 stdin/stdout 用 JSON-RPC 聊三次天（握手 / 要工具清单 / 完成通知），把拿到的工具说明书包成本地工具的样子塞进注册表；此后每次调用就是往那个进程写一行 JSON、读回一行 JSON。**

---

## 1. 参与三方

| 角色 | 所在进程 | 职责 |
|---|---|---|
| `MCPClient` | 主进程 | 拉起子进程、封装/发送 JSON-RPC、读响应 |
| `MCPServer` | **子进程** | 提供工具实现，死循环监听 stdin |
| `MCPToolAdapter` | 主进程 | 适配器：把远程工具伪装成本地 `BaseTool` |
| `ToolRegistry` | 主进程 | 统一注册表，**本地工具和 MCP 工具混存在同一个字典里** |

通信方式：**stdio 传输 + JSON-RPC 2.0**，一行一个 JSON。

```
主进程 stdin ──写一行 JSON──▶ 子进程
主进程 ◀──print 一行 JSON──── 子进程
```

---

## 2. 全局时序

```
主进程 (main.py cmd_chat)              子进程 (modules/mcp/mcp_server.py)
────────────────────────────────────   ──────────────────────────────────
① Popen 拉起子进程 ─────────────────▶  MCPServer().run()
                                        for line in sys.stdin:   ← 阻塞等待
② initialize ───────────────────────▶  handle_request → 返回 serverInfo
  ◀──── result.serverInfo ────────────
③ notifications/initialized ─────────▶  （通知，不回响应）
④ tools/list ────────────────────────▶  返回 3 个工具的「说明书」
  ◀──── [{name, description, inputSchema}] ──
⑤ 每个工具包成 MCPToolAdapter → register 进 ToolRegistry._tools
                                        （子进程继续等 stdin，全程存活）
─────────────── 以上只在启动做一次 ─────────────────────────────────────

⑥ 对话中 LLM 选中 web_search
   → ToolRegistry.execute_tool
   → BaseTool.run（模板方法）
   → MCPToolAdapter.execute
   → MCPClient.call_tool → tools/call ─▶  _web_search() 真正执行
   ◀──── content:[{type:"text", text:"…"}] ──
   → 拼成 ToolResult 回喂 LLM          （可能重复多轮）

⑦ 退出时 disconnect()：关 stdin → terminate → wait(5)
```

---

## 3. 阶段详解

### 阶段 ① 拉起子进程（构造，还没启动）

```105:109:main.py
        from modules.mcp.mcp_client import MCPClient, MCPToolAdapter
        mcp = MCPClient(
            server_command=[sys.executable, "-m", "modules.mcp.mcp_server"],
            name="enterprise-mcp",
        )
```

| 参数 | 含义 |
|---|---|
| `sys.executable` | **当前 Python 解释器的绝对路径**。不写死 `"python"`，保证子进程和主进程用同一个虚拟环境，否则子进程会找不到 `requests` / `pytz` 等依赖 |
| `-m modules.mcp.mcp_server` | 以模块方式运行 → 触发 `mcp_server.py:220` 的 `MCPServer().run()` |
| `name="enterprise-mcp"` | 纯日志标识，多个 MCP Server 时用于区分 |

这一步**只存命令，进程还没起来**。

### 阶段 ② connect() —— 真正的启动 + 握手

```46:54:modules/mcp/mcp_client.py
            self._process = subprocess.Popen(
                self._server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding="utf-8",
            )
```

三条管道全开：**stdin 写命令、stdout 读结果、stderr 读日志**。
`bufsize=1` = 行缓冲，写完立刻到对端，不攒在缓冲区。
`text=True` + `encoding="utf-8"`：直接读写字符串，不用手动 encode/decode（中文不出乱码）。

随后做两件事：

**(a) 起后台线程读 stderr**

```56:58:modules/mcp/mcp_client.py
            threading.Thread(
                target=self._read_stderr, daemon=True
            ).start()
```

这不是可选优化，是**必需**。管道缓冲区一般 64KB，子进程狂打日志而没人读 → **子进程阻塞在写 stderr，永远不响应请求**。
`daemon=True`：主线程退出时线程直接被杀，不拦着程序退出。

**(b) `initialize` 握手**

主进程写出一行：

```json
{"jsonrpc": "2.0", "method": "initialize",
 "params": {"protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "ai-agent-mcp-client", "version": "1.0.0"}},
 "id": 1}
```

子进程回一行：

```json
{"jsonrpc": "2.0", "id": 1,
 "result": {"protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "ai-agent-mcp-server", "version": "1.0.0"}}}
```

主进程只要看到 `resp["result"]` 存在就置 `_connected = True`，否则抛 `MCPException("MCP握手失败")`。

**协议规定握手后必须补一条 `initialized` 通知**：

```74:74:modules/mcp/mcp_client.py
                self._send_notification("notifications/initialized", {})
```

> Request vs Notification 的区别：
> - **Request** 带 `id`，一问一答（`_send_request` 写完会 `readline()` 阻塞等）
> - **Notification** 无 `id`，单向不期待响应（`_send_notification` 写完就返回）

### 阶段 ③ discover_tools() —— 拿「说明书」

```130:131:modules/mcp/mcp_client.py
        resp = self._send_request("tools/list", {})
        tools = resp.get("result", {}).get("tools", [])
```

请求：`{"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}`

返回的是 **3 个纯字典，没有任何可执行代码**（`mcp_server.py:40-84`）：

```python
{"name": "web_search",
 "description": "网络搜索工具，获取互联网最新信息。输入搜索关键词，返回搜索结果摘要。",
 "inputSchema": {"type": "object",
                 "properties": {"query":      {"type": "string",  ...},
                                "max_results":{"type": "integer", ...}},
                 "required": ["query"]}}
```

| 工具 | 作用 |
|---|---|
| `web_search` | Tavily 联网搜索（需 `TAVILY_API_KEY`，没配则返回提示文本，不报错） |
| `get_current_time` | 取当前时间，支持时区 |
| `knowledge_stats` | 演示用的「企业内部工具」 |

**这就是 MCP 的意义**：工具的**描述**走协议传过来，**实现留在子进程里**。换工具、加工具都不用改主进程一行代码。

### 阶段 ④ 适配器包装 + 注册

```111:113:main.py
        for tool_info in mcp.discover_tools():
            tool_registry.register(MCPToolAdapter(mcp, tool_info))
        _logger.info(f"MCP工具已接入: {len(tool_registry.list_tools())} 个工具")
```

`MCPToolAdapter` 继承 `BaseTool`，把字典「翻译」成工具对象（`mcp_client.py:185-231`）：

| MCP 字段 | 适配器属性 | 用途 |
|---|---|---|
| `name` | `self.name` | Function Calling 匹配用的工具名 |
| `description` | `self.description` | 给 LLM 看，决定选不选它 |
| `inputSchema` | `self.parameters_schema` | 给 LLM 看，决定怎么传参 |
| — | `tool_type` → `"mcp"` | 仅日志区分 local / mcp |

注册后 `ToolRegistry._tools` 的内容（`tool_registry.py:60` 就一行 `self._tools[tool.name] = tool`）：

```
calculator        (local)  ← 进程内直接执行
data_query        (local)
web_search        (mcp)    ← 转发到子进程
get_current_time  (mcp)
knowledge_stats   (mcp)
```

**从这一刻起，Agent 完全分不清哪个是本地的、哪个在别人进程里**——都只认 `name` / `to_openai_format()` / `run()`。这就是**适配器模式**。

> ⚠️ `main.py:113` 打印的是 `len(list_tools())` = **5（总数）**，不是 MCP 的 3 个。日志文案有点误导。

### 阶段 ⑤ 运行期：一次真实调用

LLM 决定要搜东西 → 返回 `tool_calls[0].function.name == "web_search"`：

```
ReActAgent._act_node
  └─ ToolRegistry.execute_tool("web_search", {"query": "…"})
       └─ BaseTool.run(params)                    ← 模板方法 base_tool.py:115
            ① validate_params()  校验 required=["query"] 是否传了
            ② execute(**params)  ← 子类实现
            ③ 计时 + 异常兜底 → 统一封装 ToolResult
                 └─ MCPToolAdapter.execute()      ← mcp_client.py:216
                      └─ MCPClient.call_tool()
                           └─ _send_request("tools/call", {...})
                                └─ 写一行 JSON 到子进程 stdin
                                     ▶ 子进程 _handle_tool_call
                                       → _web_search(**arguments)
                                       → requests.post(tavily)
                                     ◀ 回 {"content":[{"type":"text","text":"…"}],"isError":false}
                           ← 提取所有 type=text 拼成字符串
                      ← ToolResult(success=True, result="标题:…")
```

`tools/call` 报文：

```json
{"jsonrpc":"2.0","method":"tools/call",
 "params":{"name":"web_search","arguments":{"query":"今天天气"}},"id":3}
```

适配器 `execute` 只做一件事——**把 `content` 数组里的文本拼起来**：

```225:231:modules/mcp/mcp_client.py
        for item in result.get("content", []):
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return ToolResult(
            success=True,
            result="\n".join(text_parts) if text_parts else result,
        )
```

拼出来的字符串，和 `calculator` 返回的 `21`，走**同一条路**回喂 LLM。报错也一样：`isError=true` → 抛 `MCPException` → 被 `BaseTool.run` 的 `except` 接住 → 变成 `ToolResult(success=False, error=...)` → LLM 看到错误自己调整策略。

**注意分工**：`MCPToolAdapter` 里没有计时、没有 try/except、没有参数校验——这些全在继承来的 `BaseTool.run` 里。注释也写了「统一由继承来的 BaseTool.run 负责」。

### 阶段 ⑥ 销毁

```217:219:main.py
    if mcp:
        mcp.disconnect()
    pg.close()
```

`disconnect()`（`mcp_client.py:155`）三步：关 stdin（子进程 `for line in sys.stdin` 读到 EOF 自然退出）→ `terminate()` → `wait(timeout=5)`；超时或抛异常就 `kill()`。

---

## 4. 降级设计：为什么整段包在 try 里

```114:117:main.py
    except Exception as e:
        # 全链路降级第 1 道：MCP 挂了不影响本地工具，Agent 仍可用 calculator / data_query 工作
        _logger.warning(f"MCP降级（本地工具仍可用）: {e}")
        mcp = None
```

`except Exception` 抓**全部**异常，包括：

| 失败点 | 触发的异常 |
|---|---|
| `import` 失败 | ImportError |
| 命令不存在 | `FileNotFoundError` → 转成 `MCPException`（`mcp_client.py:78`） |
| 握手无响应 / 超时 | `MCPException("MCP Server 无响应")` |
| 其它 | 统一包装成 `MCPException` |

因为 MCP 是**可选增强**：没有联网搜索，Agent 照样能算数、查库。
`mcp = None` 之后 `217` 行的 `if mcp:` 判空跳过，不会 NPE。

项目里这叫「全链路降级」，同类写法还有第 ⑤ 步 RAG（`main.py:122-127`）——Milvus 挂了照样对话，只是没知识库。

---

## 5. 本项目现状：4 个值得注意的点

| # | 点 | 状态 |
|---|---|---|
| 1 | **stderr 必须有人读** | ✅ 已用守护线程解决（`_read_stderr`）。否则管道写满 → 子进程永久阻塞 |
| 2 | **`timeout` 参数实际没生效** | ⚠️ `_send_request(method, params, timeout=30.0)` 收下了参数，但里面是裸 `readline()`，**没有任何超时机制**。子进程卡死 → 主进程一直等 |
| 3 | **请求严格串行** | ℹ️ 一问一答，`_req_id` 其实没派上用场（没有并发在飞）。够用，但别指望并行调用 |
| 4 | **异常退出时子进程可能变孤儿** | ⚠️ `disconnect()` 只在正常走到 217 行时调用；主循环里抛异常直接崩，子进程不会被 `terminate` |

### 第 2 条想补的话

```python
# 思路：把 readline 挪到线程里，主线程用 queue.get(timeout=...)
def _send_request(self, method, params, timeout=30.0):
    ...
    q: Queue = Queue()
    threading.Thread(target=lambda: q.put(self._process.stdout.readline()), daemon=True).start()
    try:
        line = q.get(timeout=timeout)
    except Empty:
        raise MCPException("MCP响应超时", f"method={method}")
```

（现有代码里已经有 `except Empty:` 分支了，只是从来触发不到——因为 `readline()` 不抛 `Empty`。）

---

## 6. 速查表

| 问题 | 答案 |
|---|---|
| 子进程怎么起的 | `subprocess.Popen([sys.executable, "-m", "modules.mcp.mcp_server"])` |
| 通信格式 | 一行一个 JSON-RPC 2.0 字符串 |
| 启动期发了几个请求 | 3 个：`initialize` / `notifications/initialized` / `tools/list` |
| 拿到的是代码还是描述 | **只有描述**（name + description + inputSchema） |
| 怎么变成工具 | `MCPToolAdapter` 继承 `BaseTool` 包装 |
| 注册到哪 | `ToolRegistry._tools[name]`，和本地工具同一个字典 |
| LLM 看得到区别吗 | 看不到，都是 `to_openai_format()` 出来的函数定义 |
| 调用时走什么方法 | `tools/call`，参数 `{name, arguments}` |
| 结果怎么解析 | 取 `content` 数组里 `type=="text"` 拼接 |
| MCP 挂了会怎样 | 打 warning，`mcp=None`，本地工具照常可用 |
| 子进程什么时候死 | 正常退出 `disconnect()`；或被父进程退出带走（不保证） |
