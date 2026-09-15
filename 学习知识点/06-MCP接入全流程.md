# 06 · MCP 接入全流程

> 对应代码：`main.py:104-117`（装配）、`modules/mcp/mcp_client.py`、`modules/mcp/mcp_server.py`、`modules/mcp/protocol.py`

**一句话（专业版）**：用 `sys.executable -m` 拉起一个独立 Python 进程，通过 stdin/stdout 用 JSON-RPC 聊三次天（握手 / 要工具清单 / 完成通知），把拿到的工具说明书包成本地工具的样子塞进注册表；此后每次调用就是往那个进程写一行 JSON、读回一行 JSON。

**一句话（人话版）**：你嫌自己公司没人会联网搜索，就**外包给隔壁一家公司**。先打个电话确认人家上班（握手），问他们要一份服务价目表（拿工具清单），把价目表抄进你的内部通讯录（注册工具）。以后要搜东西，就按通讯录拨过去派单——**你手下的调度员（LLM）根本不知道这活儿是内部员工干还是外包干**。

---

## 0. 先讲人话：一个比喻贯穿全文

把 MCP 想成**外包团队**，全文所有概念都能对上：

| 现实世界 | MCP 世界 | 代码里的东西 |
|---|---|---|
| 你（小公司老板） | 主进程 | `main.py` / `MCPClient` |
| 外包团队（另一家公司，独立办公室） | 子进程 | `mcp_server.py` 那个新 Python 进程 |
| 你跟外包之间只有一部传真机 | stdio 三根管道 | `stdin` / `stdout` / `stderr` |
| 传真机的"发送格" | 父 → 子 | `stdin` |
| 传真机的"接收格" | 子 → 父 | `stdout` |
| 传真机的"碎碎念/废纸篓" | 日志通道 | `stderr` |
| 先打个电话："你好，我是 X，咱说中文 2024 版？" | `initialize` 握手 | `protocolVersion: "2024-11-05"` |
| "好的，我准备好了"（不用对方回话） | `notifications/initialized` | Notification（无 `id`） |
| 问对方要**服务价目表** | `tools/list` | 拿到 3 个工具的描述 |
| 把价目表抄进你的**通讯录**，格式和内部员工一模一样 | `MCPToolAdapter` + `register` | 适配器模式 |
| 你的**通讯录**（内部员工 + 外包混在一起） | `ToolRegistry._tools` | 同一个字典 |
| 调度员（只看通讯录，不知道谁外包） | LLM / Function Calling | `to_openai_format()` |
| 按价目表**派单** | `tools/call` | 写一行 JSON |
| 对方把活干完，传真回一份**报告** | `content:[{type:"text"}]` | 拼接成 ToolResult |
| 外包公司电话打不通 → 先用内部员工顶着 | try/except 降级 | `mcp = None` |

**记住这张表，后面所有内容都是它的展开。**

---

## 0.5 为什么这么绕？直接 `import` 进来调用不行吗？

这是最容易"云里雾里"的点，先解决它。

直觉写法是这样：

```python
from modules.tools.web_search import web_search   # 直接 import
result = web_search("今天天气")                    # 直接调用，多简单
```

**完全可行，但你放弃了 5 样东西**：

| # | 多进程（MCP）能给你的 | 同进程 import 会怎样 |
|---|---|---|
| 1 | **崩溃隔离**：`web_search` 里 `requests` 卡死 / 段错误 / 内存爆炸，最多子进程挂 | 整个 Agent 跟着一起死，用户对话直接中断 |
| 2 | **语言无关**：子进程说人话（JSON）就行，可以是 Node / Go / Rust 写的 | 只能用 Python，且必须能装进你的解释器 |
| 3 | **依赖隔离**：子进程要 `requests==1.x`，主程序要 `2.x`，各装各的互不冲突 | 依赖打架，经典的"升级一个包崩三个功能" |
| 4 | **权限隔离**：只给子进程开了三根管子，它碰不到主进程内存 | 工具代码能摸到 Agent 的所有状态 |
| 5 | **热插拔 + 生态复用**：改工具只重启子进程；别人写好的 MCP Server 直接接 | 改一行要重启整个服务 |

**一句话**：MCP 不是"为了让远程调用"而存在，是为了**把外部工具关进一个独立的盒子里**——它出事不拖累你，你也不需要知道它怎么实现的。

> 类比：你可以把外包程序员拉进你的工位坐着（import），也可以让他回自己公司、只通过工单系统联系你（MCP）。后者慢一点点，但他上班摸鱼/电脑蓝屏都不影响你。

---

## 0.6 为什么非得是 client / server **两端**？（先澄清：这是另一个问题）

> ⚠ 别和上一节搞混，它们回答的是**两件事**，只是恰好在同一个项目里同时发生：
>
> | 问题 | 答案 | 关键词 |
> |---|---|---|
> | 为什么要**跨进程**？ | 崩溃隔离 / 依赖隔离 / 语言无关（见 §0.5） | **物理边界** |
> | 为什么要分 **client / server 两端**？ | 标准化接口，让"提供方"和"使用方"能互不相识地合作（本节） | **角色分工** |
>
> 两者**可以独立存在**：
> - 只分角色、不跨进程 → 同进程里两个类，直接函数调用（很多 SDK 就这样）
> - 只跨进程、不分角色 → `multiprocessing`，父子跑同一份代码，没有协议
>
> **MCP 两样都占了**，所以看起来"绕上加绕"。

### 诚实的答案：在你这个项目里，两端**确实显得多余**

因为现在 client 和 server **都是你写的、同一台机器、同一种语言、一起启动**。这种情况下分两端，就像一个人左手写纸条递给右手——纯属多余。

**它真正的价值在"别人"和"以后"**：

| 场景 | 有两端 | 只有一端（合成一个模块） |
|---|---|---|
| 用**别人写的** MCP Server（官方 filesystem / github / postgres server） | Client **一行不用改**，换个 `server_command` 就行 | 得把他的代码抄进你的项目，还得改成你的接口 |
| 把**你的工具给别人用** | 别人装上就能调，不用看懂你的实现 | 别人得读你的源码、装你全部依赖 |
| Client 用 Python、Server 用 Go/Node 写 | 完全没问题，只认 JSON | 不可能 |

### "两端"的本质：发布契约 + 消费契约

这才是分两端的真正理由——**中间那个"协议"必须有个东西来提供、有个东西来消费**：

```
   Server 端                      契约                      Client 端
┌──────────────┐                                      ┌──────────────┐
│ 我有什么能力   │  ──发布 tools/list 菜单──▶            │ 我需要什么能力 │
│ 怎么实现(保密) │  ◀──按菜单 tools/call 下单──          │ 怎么用(自己定) │
└──────────────┘        JSON-RPC 2.0                  └──────────────┘
   可以换实现、                标准报文                    可以换界面、
   换语言、换机器               (唯一纽带)                  换调用方
```

**契约一旦定死，两边就能各自独立演化**——Server 把 Tavily 换成 Bing，Client 毫不知情；Client 从命令行换成 Web，Server 毫不知情。

> 类比 **USB 接口**：
> - 为什么要分"主机"和"U 盘"两端？因为只有两端都遵守同一套标准，**任意品牌的 U 盘插任意电脑都能用**。
> - 如果电脑和硬盘焊死在一起（合成一端），那你的硬盘只能在这台电脑上用。
> - 你家里"电脑 + 自己的移动硬盘"看起来确实多此一举——但正是这个标准让你哪天换硬盘不用换电脑。

### 活体对照实验：你项目里两种工具**同时存在**

这是理解"要不要两端"最直接的方式——同一个 Agent 里，两种写法摆在一起：

| | `calculator` / `data_query`（**无两端**） | `web_search`（**有两端**） |
|---|---|---|
| 工具在哪定义 | 直接在 `BaseTool` 子类里写 | 在**子进程**里写 |
| 怎么进注册表 | `register(工具实例)`，一步到位 | `tools/list` 拿描述 → 包装成 Adapter → 注册 |
| 怎么执行 | 同进程**函数调用** | 写一行 JSON → 过管道 → 读回一行 JSON |
| 工具崩了 | **整个 Agent 一起死** | 只死子进程，Agent 活着 |
| 代码量 | 少 | 多（多出 client / protocol / adapter 一整套） |
| 适合什么 | 自己写的、稳定的、纯 Python 的 | 别人的、不稳定的、依赖冲突的、非 Python 的 |

**所以结论不是"两端更好"，而是"两端有代价，换来了复用和隔离"。**

### 什么时候该上两端？

```
工具是你自己写的？
├─ 否 → 上 MCP（你没法把别人的代码塞进自己进程）
└─ 是 → 它稳定吗？依赖会跟主程序打架吗？需要隔离吗？
        ├─ 都还好 → 直接本地工具（calculator 那样，简单快）
        └─ 不稳定 / 依赖冲突 / 要热更新 / 想给别人用 → 上 MCP
```

### 好处先量化：从 N×M 变成 N+M

假设公司里有 3 个都要用工具的东西：**客服机器人（Python）**、**数据看板（Node.js）**、**审批助手（Java，别的部门维护）**，共用一批工具。

**不分成两端**（每个产品自己把工具代码写进去）：

| 要做什么 | 要改几处 |
|---|---|
| 接一个新工具 | **3 处**，三个产品各写一遍实现 |
| 工具供应商换了（Tavily → Bing） | **3 处**，各自改各自的 |
| 工具加一个参数 | **3 处** |
| API Key 轮换 | **3 份配置** |

= **N×M 份实现**，任何变更都要改 N 次。

**分成两端**：

| 要做什么 | 要改几处 |
|---|---|
| 接一个新工具 | 起一个新 Server，**三个 Client 下次 discovery 自动就看见了** |
| 供应商换了（Tavily → Bing） | **1 处**，只改 Server。Client 毫无感知 |
| 加一个参数 | **1 处**，Server 改 `inputSchema`，Client 下次拉取自动生效 |
| Key 轮换 | **1 份**，在 Server 那边 |

= **N+M**，而且**变更永远只落在 Server 一侧**。

> 一句话记住：**两端不是为了"现在方便"，是为了"以后能换"和"出事不连坐"。**

### 实机例子：同一个 Client，换 Server 不改一行

项目里准备了两个实现风格**完全无关**的 Server（都在 `examples/` 下）：

| | 实现 | 依赖 | 提供的工具 |
|---|---|---|---|
| **Server A** | `modules/mcp/mcp_server.py`：类、`get_logger`、`modules.mcp.protocol` | 半个项目 | `web_search` / `get_current_time` / `knowledge_stats` |
| **Server B** | `examples/wild_mcp_server.py`：纯标准库，一个平铺 `for` 循环，没有 class | **零项目依赖** | `string_reverse` |

跑这个脚本（`examples/multi_server_demo.py`）：

```bat
cmd /c "cd /d e:\workspace\AI\AI-project\ai_agent_project && python examples\multi_server_demo.py 2>NUL"
```

真实输出：

```
==============================================================
  Server A (self-built)   <-  -m modules.mcp.mcp_server
==============================================================
connect()       -> True
discover_tools  -> ['web_search', 'get_current_time', 'knowledge_stats']
call_tool       -> {"knowledge_base": "Milvus knowledge_base", "status": "运行中"...

==============================================================
  Server B (wild)         <-  examples/wild_mcp_server.py
==============================================================
connect()       -> True
discover_tools  -> ['string_reverse']
call_tool       -> PCM olleH          ← "Hello MCP" 被反转

==============================================================
Client code unchanged, yet it talks to two totally different servers.
==============================================================
```

** Client 侧从头到尾只有这 4 行，两个 Server 共用一套**：

```python
client = MCPClient(server_command, name=label)   # ← 只有这里换个命令
client.connect()
client.discover_tools()
client.call_tool(...)
```

请重点看 Server B：它**连你项目的 `protocol.py` 都没 import**，手写 `json.dumps` 就完事了——照样接得住。

> **这就是"两端"的全部意义**：`MCPClient` 和 `modules/mcp/mcp_server.py` 之间**没有代码级依赖**，只有报文级约定。
> 所以明天换成 Anthropic 官方的 server（Node 写的）、或者隔壁团队用 Go 写的 server，你的 Client **依然一行不用改**。

---

## 1. 参与三方

| 角色 | 所在进程 | 职责 | 人话 |
|---|---|---|---|
| `MCPClient` | 主进程 | 拉起子进程、封装/发送 JSON-RPC、读响应 | 你的**传真机操作员** |
| `MCPServer` | **子进程** | 提供工具实现，死循环监听 stdin | **外包团队**，守着传真机等单 |
| `MCPToolAdapter` | 主进程 | 适配器：把远程工具伪装成本地 `BaseTool` | **价目表 → 通讯录格式**的转换器 |
| `ToolRegistry` | 主进程 | 统一注册表，**本地工具和 MCP 工具混存在同一个字典里** | 那本**通讯录** |

通信方式：**stdio 传输 + JSON-RPC 2.0**，一行一个 JSON。

```
主进程 stdin ──写一行 JSON──▶ 子进程
主进程 ◀──print 一行 JSON──── 子进程
```

> **人话 + 专业**：管道是**字节流**，本身没有"一条消息"的概念（不像 UDP 有包边界）。
> 所以双方约定：**一条 JSON 就是一行，用换行符 `\n` 当分隔符**——这叫 NDJSON / 行分隔 JSON，也是"帧定界"。
> 就像微信聊天：一条消息一行，**不能把两条消息粘在一行发**，对面会解析失败。

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
─────────── 以上只在启动做一次 ──────────────────────────────────────────

⑥ 对话中 LLM 选中 web_search
   → ToolRegistry.execute_tool
   → BaseTool.run（模板方法）
   → MCPToolAdapter.execute
   → MCPClient.call_tool → tools/call ─▶  _web_search() 真正执行
   ◀──── content:[{type:"text", text:"…"}] ──
   → 拼成 ToolResult 回喂 LLM          （可能重复多轮）

⑦ 退出时 disconnect()：关 stdin → terminate → wait(5)
```

### 翻译成人话

| 步 | 人话 |
|---|---|
| ① | **招人**：给外包公司打个电话把工位开通（`Popen`），对方电话接通后一直举着听筒等你说话（`for line in sys.stdin`） |
| ② | **对暗号**："你好，我是 ai-agent，咱说 2024-11-05 版的行话？" 对方回"行，我是 ai-agent-mcp-server" |
| ③ | **挂个尾句**："那我开始派单了啊"——这句**不用对方回**（所以叫通知） |
| ④ | **要价目表**："你们都能干啥？" 对方传真过来 3 张服务说明 |
| ⑤ | **抄进通讯录**：把 3 张说明抄成内部员工一样的格式，塞进通讯录 |
| ⑥ | **开始派单**：调度员（LLM）翻通讯录说"这个活找 web_search"→ 派单 → 对方干完传真回报告 |
| ⑦ | **下班**：先跟对方说"没事了挂了"（关 stdin）→ 强拆（terminate）→ 等 5 秒，还不走就拔电源（kill） |

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

| 参数 | 含义 | 人话 |
|---|---|---|
| `sys.executable` | **当前 Python 解释器的绝对路径**。不写死 `"python"`，保证子进程和主进程用同一个虚拟环境，否则子进程会找不到 `requests` / `pytz` 等依赖 | 用**你现在正在用的那个 python**，别写死 `"python"`——不然可能叫到系统里另一个 python，它没装依赖，一开工就报错 |
| `-m modules.mcp.mcp_server` | 以模块方式运行 → 触发 `mcp_server.py:220` 的 `MCPServer().run()` | 等价于在终端敲 `python -m modules.mcp.mcp_server` |
| `name="enterprise-mcp"` | 纯日志标识，多个 MCP Server 时用于区分 | 给外包公司起个名，日志里好认 |

这一步**只存命令，进程还没起来**——就像你只记下了对方的电话号码，还没拨。

### 阶段 ② connect() —— 真正的启动 + 握手

```58:83:modules/mcp/mcp_client.py
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

| 管道 | 方向 | 人话 | 踩坑 |
|---|---|---|---|
| `stdin` | 父 → 子 | 传真机的**发送格**：往里塞问题纸条 | 管道**单向**，"说"和"听"必须是两根不同的管子 |
| `stdout` | 子 → 父 | 传真机的**接收格**：拿对方递回来的回答 | ⚠ **协议专线**！子进程任何 `print` / 日志都不许打到这里，否则 `parse_message` 收到非 JSON 的行，整个协议崩掉 |
| `stderr` | 子 → 父 | 对方**自言自语**（日志），不参与协议 | ⚠ 必须有人持续读，见下方 (a) |

其余两个参数：

- `bufsize=1`：**行缓冲**。写完立刻到对端，不攒在缓冲区里（所以 `_send_request` 写完还会跟一行 `flush()`）。
  > 人话：纸条**塞进去就滑到对面**，不是攒够一沓才送一次。
- `text=True` + `encoding="utf-8"`：直接读写字符串，不用手动 encode/decode（中文不出乱码）。
  > 人话：传真机直接认**汉字**，不用你自己先把汉字翻译成拼音。Windows 上不写 `utf-8` 会走系统默认的 GBK，中文**必乱码**。

随后做两件事：

**(a) 起后台线程读 stderr**

```85:87:modules/mcp/mcp_client.py
            threading.Thread(
                target=self._read_stderr, daemon=True
            ).start()
```

这不是可选优化，是**必需**。管道缓冲区一般 64KB，子进程狂打日志而没人读 → **子进程阻塞在写 stderr，永远不响应请求**。

> **人话**：对方的"碎碎念"通道是根**细管子**。你不管它，它说满 64KB 就堵住了，然后它光顾着堵在这儿，**再也不干正事了**——你等它回复，它等你清空管子，**俩人一起干等**，这就是**死锁**。
> 所以专门派个实习生（守护线程）蹲在那儿，对方说一句他就记一句，管子永远不通。

`daemon=True`：主线程退出时线程直接被杀，不拦着程序退出（实习生不加班，公司关门他就走）。

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

> **人话**：这就是**对暗号**。
> - `protocolVersion` = "咱说 2024-11-05 这一版的行话？"（版本对不上就各说各的，直接谈崩）
> - `clientInfo` / `serverInfo` = 互相报个家门
> - `id: 1` = **这句话的编号**。因为可能同时有多句话在飞，回信要带上编号才知道回的是哪句。

主进程只要看到 `resp["result"]` 存在就置 `_connected = True`，否则抛 `MCPException("MCP握手失败")`。

**协议规定握手后必须补一条 `initialized` 通知**：

```86:86:modules/mcp/mcp_client.py
                self._send_notification("notifications/initialized", {})
```

> **人话**：暗号对完，你还得补一句"OK 我开始派单了啊"——这是**礼节**，对方听到就行，**不用回你**。
> 类比微信：发个"收到"，对方回不回无所谓，你也不等。

> Request vs Notification 的区别（**这是 JSON-RPC 的核心**）：
> - **Request** 带 `id`，**一问一答**（`_send_request` 写完会 `readline()` **阻塞死等**）——像打电话，你得举着听筒等对方说话
> - **Notification** 无 `id`，**单向不期待响应**（`_send_notification` 写完就返回）——像发微信，发完把手机放下

### 阶段 ③ discover_tools() —— 拿「说明书」

```159:160:modules/mcp/mcp_client.py
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

> **人话**：拿回来的**不是软件，是一张菜单**。
> 菜单上写着：菜名（`name`）、这道菜是啥（`description`）、要提供什么食材（`inputSchema`）。
> **但你拿不到厨师，也拿不到菜谱**——做菜的人留在对面公司里。

**这就是 MCP 的意义**：工具的**描述**走协议传过来，**实现留在子进程里**。换工具、加工具都不用改主进程一行代码。

> 类比：你不需要把外包程序员挖过来上班，只要拿到他的**服务目录**就能派活。他明天换了个人干，你这边毫无感觉。

### 阶段 ④ 适配器包装 + 注册

```111:113:main.py
        for tool_info in mcp.discover_tools():
            tool_registry.register(MCPToolAdapter(mcp, tool_info))
        _logger.info(f"MCP工具已接入: {len(tool_registry.list_tools())} 个工具")
```

`MCPToolAdapter` 继承 `BaseTool`，把字典「翻译」成工具对象（`mcp_client.py:203-262`）：

| MCP 字段 | 适配器属性 | 用途 | 人话 |
|---|---|---|---|
| `name` | `self.name` | Function Calling 匹配用的工具名 | 通讯录里的**名字** |
| `description` | `self.description` | 给 LLM 看，决定选不选它 | **这人能干啥**（调度员看着决定派不派活） |
| `inputSchema` | `self.parameters_schema` | 给 LLM 看，决定怎么传参 | **派单时要填哪些格子** |
| — | `tool_type` → `"mcp"` | 仅日志区分 local / mcp | 备注栏写个"外包" |

注册后 `ToolRegistry._tools` 的内容（`tool_registry.py:60` 就一行 `self._tools[tool.name] = tool`）：

```
calculator        (local)  ← 进程内直接执行（内部员工）
data_query        (local)
web_search        (mcp)    ← 转发到子进程（外包）
get_current_time  (mcp)
knowledge_stats   (mcp)
```

**从这一刻起，Agent 完全分不清哪个是本地的、哪个在别人进程里**——都只认 `name` / `to_openai_format()` / `run()`。这就是**适配器模式**。

> **人话 + 专业**：`MCPToolAdapter` 就是个**转换插头**。
> 外包的接口是"传真单"（JSON-RPC），你公司的接口是"工单"（`BaseTool`）。转换插头一插，两边就长得一模一样了。
> 这就是设计模式里的**适配器（Adapter）**：不改任何一方，只加一层壳，让两个不兼容的接口能一起工作。

> ⚠️ `main.py:113` 打印的是 `len(list_tools())` = **5（总数）**，不是 MCP 的 3 个。日志文案有点误导。

### 阶段 ⑤ 运行期：一次真实调用

LLM 决定要搜东西 → 返回 `tool_calls[0].function.name == "web_search"`：

```
ReActAgent._act_node
  └─ ToolRegistry.execute_tool("web_search", {"query": "…"})   ← 翻通讯录找人
       └─ BaseTool.run(params)                    ← 模板方法 base_tool.py:115
            ① validate_params()  校验 required=["query"] 是否传了
            ② execute(**params)  ← 子类实现
            ③ 计时 + 异常兜底 → 统一封装 ToolResult
                 └─ MCPToolAdapter.execute()      ← mcp_client.py:245
                      └─ MCPClient.call_tool()
                           └─ _send_request("tools/call", {...})
                                └─ 写一行 JSON 到子进程 stdin   ← 发传真
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

> **人话**：`arguments` 就是**派单时填的格子**。LLM 看到菜单上写着"必填 query"，所以填了 `{"query":"今天天气"}`。

适配器 `execute` 只做一件事——**把 `content` 数组里的文本拼起来**：

```254:260:modules/mcp/mcp_client.py
        for item in result.get("content", []):
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return ToolResult(
            success=True,
            result="\n".join(text_parts) if text_parts else result,
        )
```

> **人话**：对方传真回来的是一个**包裹**（`content` 数组），里面可能装了文字、图片、表格。我们只把其中**文字那部分**掏出来拼成一段话。

拼出来的字符串，和 `calculator` 返回的 `21`，走**同一条路**回喂 LLM。报错也一样：`isError=true` → 抛 `MCPException` → 被 `BaseTool.run` 的 `except` 接住 → 变成 `ToolResult(success=False, error=...)` → LLM 看到错误自己调整策略。

> **人话**：不管是内部员工算出来的 `21`，还是外包搜回来的"今天北京晴，25 度"，**回到调度员手里都是一张写着结果的纸**。他不需要知道纸是怎么来的。
> 失败也一样：外包回一句"KV 没配，干不了"，调度员收到后自己想办法换个工具，不会让整个流程崩掉。

**注意分工**：`MCPToolAdapter` 里没有计时、没有 try/except、没有参数校验——这些全在继承来的 `BaseTool.run` 里。注释也写了「统一由继承来的 BaseTool.run 负责」。

> **人话 + 专业**：这是**模板方法模式**——父类 `BaseTool.run` 定好流程骨架（校验 → 执行 → 计时 → 兜底），子类只填"执行"这一步。
> 好处：所有工具（本地的、外包的）**行为一致**，不会出现"这个工具超时没兜底、那个工具没计时"的坑。

### 阶段 ⑥ 销毁

```217:219:main.py
    if mcp:
        mcp.disconnect()
    pg.close()
```

`disconnect()`（`mcp_client.py:184`）三步：关 stdin（子进程 `for line in sys.stdin` 读到 EOF 自然退出）→ `terminate()` → `wait(timeout=5)`；超时或抛异常就 `kill()`。

> **人话**：辞退外包的三步曲，从礼貌到粗暴：
> 1. **礼貌分手**：把发送格关了（`stdin.close()`）→ 对方发现再也没单子进来（读到 EOF），自己收拾东西走人
> 2. **下逐客令**：`terminate()`（发 SIGTERM，"请你离开"）
> 3. **拔电源**：等 5 秒还不走 → `kill()`（发 SIGKILL，"我不管了你给我消失"）

---

## 4. 降级设计：为什么整段包在 try 里

```114:117:main.py
    except Exception as e:
        # 全链路降级第 1 道：MCP 挂了不影响本地工具，Agent 仍可用 calculator / data_query 工作
        _logger.warning(f"MCP降级（本地工具仍可用）: {e}")
        mcp = None
```

`except Exception` 抓**全部**异常，包括：

| 失败点 | 触发的异常 | 人话 |
|---|---|---|
| `import` 失败 | ImportError | 连外包公司的联系方式都丢了 |
| 命令不存在 | `FileNotFoundError` → 转成 `MCPException`（`mcp_client.py:108-111`） | 电话号码是空号 |
| 握手无响应 / 超时 | `MCPException("MCP Server 无响应")` | 电话拨通了没人接 |
| 其它 | 统一包装成 `MCPException` | — |

因为 MCP 是**可选增强**：没有联网搜索，Agent 照样能算数、查库。
`mcp = None` 之后 `217` 行的 `if mcp:` 判空跳过，不会 NPE。

> **人话**：外包公司今天集体放假？没关系，**内部员工还在**。
> 只是今天不能联网搜索了，算数、查库照样能干。用户体验是"功能少一点"，而不是"直接崩了"。

项目里这叫「全链路降级」，同类写法还有第 ⑤ 步 RAG（`main.py:122-127`）——Milvus 挂了照样对话，只是没知识库。

> 这就像饭店：海鲜卖完了，服务员不会说"关门了"，而是说"今天海鲜没有，您看点别的？"——**核心服务不中断，可选能力缺失**。

---

## 5. 本项目现状：5 个值得注意的点

| # | 点 | 状态 | 人话 |
|---|---|---|---|
| 1 | **stderr 必须有人读** | ✅ 已用守护线程解决（`_read_stderr`）。否则管道写满 → 子进程永久阻塞 | 已派人蹲在碎碎念通道旁边（见阶段②） |
| 2 | **`timeout` 参数实际没生效** | ⚠️ `_send_request(method, params, timeout=30.0)` 收下了参数，但里面是裸 `readline()`，**没有任何超时机制**。子进程卡死 → 主进程一直等 | 你设了"最多等 30 秒"，但电话举起来就**忘了看表**，对方不说话你就一直举着 |
| 3 | **请求严格串行** | ℹ️ 一问一答，`_req_id` 其实没派上用场（没有并发在飞）。够用，但别指望并行调用 | 一次只派一个单，等回执来了才派下一个。不能同时派 3 个单 |
| 4 | **异常退出时子进程可能变孤儿** | ⚠️ `disconnect()` 只在正常走到 217 行时调用；主循环里抛异常直接崩，子进程不会被 `terminate` | 公司突然着火（主进程崩溃），大家跑了，但**外包公司还开着工位在那儿等单** |
| 5 | **中文 Windows 上 MCP 实际失效** | 🔴 子进程按 GBK 写管道、父进程按 UTF-8 读 → `UnicodeDecodeError`，被降级静默吞掉，详见下方「第 5 条」 | **说好的中文传真，结果对方用摩斯电码发过来，你压根读不懂** |

### 第 5 条：Windows 中文环境下 MCP 实际失效（已实测复现）

**现象**：Windows 中文系统上，`initialize` 握手能成功，一调 `tools/list` 就炸：

```
Exception in thread Thread-1 (_read_stderr):
  File "modules/mcp/mcp_client.py", line 119, in _read_stderr
    for line in self._process.stderr:
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc6 in position 98

MCP工具发现异常 | 未知异常: UnicodeDecodeError:
  'utf-8' codec can't decode byte 0xcd in position 88
```

**根因**：`Popen` 里写死的 `encoding="utf-8"` **只管父进程怎么解码**，管不着子进程用什么编码写：

```
子进程（Windows）                      父进程
写 stdout 用系统 locale = GBK/cp936  ✗  读的时候按 encoding="utf-8"
（0xc6 0xcd 就是 GBK 汉字）             解不出来 → 抛异常
```

**为什么偏偏 `tools/list` 才炸**：`initialize` 响应里全是英文（`serverInfo`），碰巧没事；而**工具清单里有中文 `description`**——只要响应含中文，必崩。

**为什么一直没被发现**：异常被两道防线层层吃掉，最终表现只是"少了几个工具"：

```
tools/list 抛 UnicodeDecodeError
  → discover_tools 的 @handle_exception 吞掉，返回 []
  → main.py:114 的 except 兜住，不外传
  → 只打一行 warning：MCP降级（本地工具仍可用）
  → 用户感受：Agent 能用，就是没有联网搜索而已
```

**修复**（任选其一）：

```python
# 方案 A（推荐）：给子进程传环境变量，强制它用 UTF-8 输出
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
subprocess.Popen(self._server_command, ..., env=env)

# 方案 B（临时兜底，中文会变乱码）：父进程容错
subprocess.Popen(..., encoding="utf-8", errors="replace")
```

`examples/multi_server_demo.py` 用的是**方案 A 的等价写法**（启动前设 `os.environ`，`Popen` 不传 `env` 时子进程会继承），所以那个 demo 能正常跑。

> ⚠ 通用教训：**凡是 subprocess + 管道 + 中文，都必须显式约定编码**，不能指望父子进程 locale 恰好一致。

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

> **人话解决思路**：你自己举着电话会忘看表，那就**让实习生举着**，你在旁边掐表。
> 30 秒到了实习生还没递过来回话，你就挂电话——这样 `timeout` 才真的生效。

（现有代码里已经有 `except Empty:` 分支了，只是从来触发不到——因为 `readline()` 不抛 `Empty`。）

---

## 6. 速查表

| 问题 | 答案 |
|---|---|
| 子进程怎么起的 | `subprocess.Popen([sys.executable, "-m", "modules.mcp.mcp_server"])` |
| 通信格式 | 一行一个 JSON-RPC 2.0 字符串（NDJSON，用换行分帧） |
| 启动期发了几个请求 | 3 个：`initialize` / `notifications/initialized` / `tools/list` |
| 拿到的是代码还是描述 | **只有描述**（name + description + inputSchema） |
| 怎么变成工具 | `MCPToolAdapter` 继承 `BaseTool` 包装（适配器模式） |
| 注册到哪 | `ToolRegistry._tools[name]`，和本地工具同一个字典 |
| LLM 看得到区别吗 | 看不到，都是 `to_openai_format()` 出来的函数定义 |
| 调用时走什么方法 | `tools/call`，参数 `{name, arguments}` |
| 结果怎么解析 | 取 `content` 数组里 `type=="text"` 拼接 |
| MCP 挂了会怎样 | 打 warning，`mcp=None`，本地工具照常可用 |
| 子进程什么时候死 | 正常退出 `disconnect()`；或被父进程退出带走（不保证） |
| 为什么不用 import | 崩溃隔离 / 依赖隔离 / 语言无关 / 权限隔离 / 生态复用（见 0.5） |

---

## 7. 三句话背诵版

1. **招人**：`Popen` 起个子进程，三根管道——`stdin` 发单、`stdout` 收报告、`stderr` 听碎碎念（**必须有人听，会堵**）。
2. **对暗号 + 要菜单 + 抄通讯录**：`initialize`（带 `id`，一问一答）→ `initialized`（无 `id`，发完不等）→ `tools/list`（拿描述）→ 包装成 `BaseTool` 注册进 `ToolRegistry`。
3. **派单**：LLM 翻通讯录选中工具 → `tools/call` 写一行 JSON → 拿回 `content` 拼文本 → 和本地工具走同一条路回喂 LLM；全程 `try/except` 兜底，**MCP 挂了只是少几个工具**。
