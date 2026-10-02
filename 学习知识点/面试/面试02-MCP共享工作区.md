# 面试02：MCP 独立会话，为什么多 Agent 能在同一工作区协作？

> 配套代码：`modules/mcp/workspace.py`（新增实现）、`modules/mcp/mcp_client.py:87-95`、`modules/mcp/gateway.py:364-395`、`modules/multiagent/harness.py:256-260`
> 一句话结论：**MCP 的会话隔离是"一个连接一个会话"，多 Agent 协作的解法是把会话的归属从「Agent」提升到「工作区」——同工作区共享同一个 client（同一子进程同一会话），跨工作区各拿各的 client 物理隔离。**

---

## 一、先说结论

| 面试问 | 一句话答 |
|---|---|
| 为什么默认状态互相不可见？ | stdio 传输下每次 `connect` = 拉起一个新子进程，进程之间天然隔离 |
| 多 Agent 怎么共享？ | `MCPWorkspaceManager`：`(workspace, server)` 二元组决定一个共享 client |
| 互不冲突靠什么？ | ① 引用计数管连接生命周期 ② 会话内调用串行化（stdio 单管道）③ 工作区状态板带覆盖留痕 |
| 跨工作区会串吗？ | 不会：不同 workspace 是不同 dict 键，连 client 都是各自独立的子进程 |
| 忘了 release 会怎样？ | refs 永不归零 → 子进程永远残留；用 `session_stats()` 定位泄漏点 |

---

## 二、隔离的根源：往下挖三层

### 第 1 层：代码层——connect 做了什么（mcp_client.py:87-95）

```python
self._process = subprocess.Popen(          # 每次 connect 都拉起一个子进程
    self._server_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, ...)
resp = self._send_request("initialize", {...})   # 子进程里那份状态，只有这个连接看得见
```

| 部分 | 大白话解释 |
|---|---|
| `subprocess.Popen` | 起一个新进程跑 MCP server，stdin/stdout 各接一根管子 |
| `initialize` 握手 | 这根管子的"开卡仪式"，之后 server 里的会话状态只挂在这张卡上 |
| 结论 | Agent A 和 Agent B 各 connect 一次 = 两个进程两份状态，A 建的目录 B 根本看不见 |

### 第 2 层：协议层——为什么"一连接一进程"

MCP 的 stdio 传输约定：**客户端与 server 之间是两条单向管道（客户端 stdin 写入 → server stdout 读出），全双工只有这一对管子**。协议上没有"一个 server 进程服务多个连接"的通道复用设计——想多连一份，就只能再拉一个进程。

### 第 3 层：进程层——为什么进程隔离 = 状态隔离

操作系统保证：两个进程的内存空间互相不可见。MCP server 的会话状态（当前目录、打开的文件句柄、它自己缓存的变量）全住在自己的进程内存里。所以：

```python
# Agent A 的连接：                    # Agent B 的连接：
subprocess #1（内存：cwd=/tmp/a）     subprocess #2（内存：cwd=/）
# A 执行 mkdir /tmp/a/reports
# B 执行 ls /tmp/a        → 空的！B 的进程里 /tmp/a 只是普通目录，
#                            A 会话里"我在 /tmp/a 工作过"的记忆不存在
```

**面试点**：这层隔离不是 MCP 的设计缺陷，而是 stdio 传输的天然属性（http 传输则是一个 server 进程服务多连接，但本项目 MCP server 以 stdio 为主）。

---

## 三、共享的实现：数据结构先看清

`MCPWorkspaceManager` 内部就两张表 + 一把锁（workspace.py:64-67）：

```python
self._sessions: Dict[Tuple[str, str], _WorkspaceSession] = {}
#           键是元组 → 值是会话条目
#           ("ws-x", "fs")  → _WorkspaceSession(client=client#1, refs=2, busy=Lock)
#           ("ws-y", "fs")  → _WorkspaceSession(client=client#2, refs=1, busy=Lock)

self._state: Dict[str, Dict[str, dict]] = {}
#           工作区名 → {键: {"value":..., "owner":..., "prev_owner":..., "prev_value":...}}
#           "ws-x" → {"progress": {"value": "B在做演示", "owner": "agent-b",
#                                   "prev_owner": "agent-a", "prev_value": "A做完调研"}}

self._lock = threading.Lock()   # 保护上面两张表的锁
```

`_WorkspaceSession` 三个字段各司其职：

| 字段 | 类型 | 作用 |
|---|---|---|
| `client` | MCPClient | 共享的客户端对象 = 同一个子进程 = 同一个 MCP 会话 |
| `refs` | int | 当前有几个 Agent 在用；**归零才允许断开** |
| `busy` | Lock | 会话级锁：串行化对这条管道的工具调用 |

### 关键设计：为什么键是元组 `(workspace, server)` 而不是嵌套 dict

```python
# 元组键（本项目）：一次 dict 查找定位
sess = self._sessions[("ws-x", "fs")]

# 嵌套 dict（等价写法）：两次查找
sess = self._sessions["ws-x"]["fs"]
```

两种都能实现"两级映射"，选元组是因为 Python 的 tuple 可哈希、可直接当 dict 键，查找一步到位；且删除时（release 归零）一条 `del self._sessions[key]` 连带清掉整级，不存在"外层 dict 留空壳"的问题。

---

## 四、生命周期：acquire / release 逐行拆

### acquire：加入工作区（workspace.py:89-99）

```python
def acquire(self, workspace: str, server: str) -> Any:
    key = (workspace, server)
    with self._lock:                                # 全局锁：两张表只能一个线程改
        sess = self._sessions.get(key)
        if sess is None:                            # 这个组合第一次有人用
            client = self._factory(server)          # 工厂造 client 并 connect（失败直接抛异常）
            sess = _WorkspaceSession(client)
            self._sessions[key] = sess
            _logger.info(...)
        sess.refs += 1                              # 不论新建还是复用，引用 +1
        return sess.client                          # 返回的是同一个对象（同区）
```

| 行 | 大白话解释 |
|---|---|
| `with self._lock` | 两个 Agent 同时加入同一工作区时，防止造出两个 client（一个查表造、一个也查表也造）——查表和登记必须锁内一口气完成 |
| `self._factory(server)` | 只传 server 名不传 workspace：**client 本身不知道工作区**，工作区只是管理器账本上的键 |
| `sess.refs += 1` | 复用路径唯一动作：不加锁对象、不新连接，只把计数 +1 |
| `return sess.client` | 同区两个 Agent 拿到的 `id()` 相同——这就是"共享"的全部实现 |

### release：退出工作区（workspace.py:103-120）

```python
def release(self, workspace: str, server: str) -> None:
    key = (workspace, server)
    with self._lock:
        sess = self._sessions.get(key)
        if sess is None:
            return                                  # 重复 release：静默返回，不报错
        sess.refs -= 1
        if sess.refs <= 0:                          # 最后一个退出
            del self._sessions[key]                 # 账本上先除名（锁内）
            client = sess.client                    # 把 client 拿出来，锁外再断
        else:
            return                                  # 还有人在用，什么都不做
    # ↓ 已经出了 with self._lock（锁已释放）
    try:
        client.disconnect()                         # 真正断开：等子进程退出，可能阻塞几秒
    except Exception as e:
        _logger.warning(f"...断开失败(忽略): {e}")
```

| 行 | 大白话解释 |
|---|---|
| `if sess is None: return` | release 一个不存在的会话 = 幂等设计：重复退出不炸、不倒扣 |
| `del` 在锁内、`disconnect` 在锁外 | **除名要快（锁内），断开可能慢（锁外）**。disconnect 要等子进程退出，攥着全局锁等的话，其他工作区的 acquire 全被卡住 |
| `except Exception` 只记日志 | 断不开（子进程僵死）时不能拖死调用方：残留一个进程好过抛异常中断 Agent 流程 |

### 生命周期状态机

```
                acquire 首次
   (不存在) ───────────────────→ refs=1, client 已 connect
                                    │
                 acquire 复用 ──→ refs+1  ←── release（还有人用）refs-1
                                    │
                                    │ release 时 refs 减到 0
                                    ↓
                              del 出账本（锁内）→ disconnect（锁外）→ 回到 (不存在)
```

### 输入 → 输出对照（本机实测）

```python
m = MCPWorkspaceManager(client_factory=factory)    # factory 每次造一个新的 FakeClient

c1 = m.acquire("ws-x", "fs")      # 第 1 个 Agent 加入 → 新建 client#1，refs=1
c2 = m.acquire("ws-x", "fs")      # 第 2 个 Agent 加入 → 复用 client#1，refs=2
c3 = m.acquire("ws-y", "fs")      # 另一个工作区   → 新建 client#2，refs=1

# 运行结果:
# same-workspace share: True        ← id(c1) == id(c2)，同一个对象
# cross-workspace isolated: True    ← id(c1) != id(c3)，完全不同
# refs: {'ws-x/fs': 2, 'ws-y/fs': 1}

m.release("ws-x", "fs")            # 一个 Agent 退出
# after 1 release, connected still: True   ← refs=1，还有人在用，连接保持
m.release("ws-x", "fs")            # 最后一个退出
# after last release, closed: True         ← refs=0，真正 disconnect，子进程退出
```

---

## 五、call_tool：为什么还需要一把会话锁

```python
def call_tool(self, workspace, server, tool, arguments) -> dict:
    key = (workspace, server)
    with self._lock:
        sess = self._sessions.get(key)
        if sess is None:                    # 分支①：便捷用法（没加入也能调）
            client = self.acquire(workspace, server)     # 自动加入
            sess = self._sessions[key]
            try:
                with sess.busy:                          # 会话锁
                    return client.call_tool(tool, arguments) or {}
            finally:
                self.release(workspace, server)          # 调完自动退出
    with sess.busy:                          # 分支②：已在会话里（正常路径）
        return sess.client.call_tool(tool, arguments) or {}
```

### 两个分支

| 分支 | 触发条件 | 行为 |
|---|---|---|
| ① 便捷用法 | 没有先 acquire 就直接调 | 临时加入 → 调用 → 临时退出（acquire/release 配对，refs 涨了又落） |
| ② 正常路径 | 已在会话里 | 只拿会话锁排队调用，不动引用计数 |

### 为什么全局锁之外还要会话锁（busy）

两把锁管的不是同一件事：

| 锁 | 管什么 | 不管什么 |
|---|---|---|
| `self._lock`（全局） | 两张表（_sessions/_state）的读写 | 不拦工具调用本身 |
| `sess.busy`（会话级） | 这条 stdio 管道的写入串行化 | 不拦别的会话 |

stdio 是**单管道**：两个线程同时往同一根管子写 JSON-RPC 请求，字节流会交叉：

```python
# 不加 busy 锁时（示意）：
线程A写: {"method":"tools/call","params":{"name":"mkdir"...}}
线程B写: {"method":"tools/call","params":{"name":"ls"...}}
# 管子里实际可能变成：
# {"method":"tools/call","params":{"name":"mkdir"{"method":"tools/c...
# server 解析出乱码或错位，且不一定报错——静默出错比崩溃更难查

# 加锁后（本机实测，3 个线程同会话并发调用 6 次）：
# 结果：6/6 全部成功，调用在日志里首尾相接、无交叉
```

### 排队时序推演（同工作区 3 个线程同时调工具）

```
时间 →
t1: 拿到 busy ──── call_tool("mkdir") ──── 还锁
t2:      等 busy ──────────┬拿到 busy ──── call_tool("write") ──── 还锁
t3:      等 busy ──────────┴──────────────────┬拿到 busy ──── call_tool("ls") ──── 还锁
# 任意时刻管道里只有一个请求在飞——这就是"串行化"的含义
```

注意跨工作区不排队：ws-x 的调用和 ws-y 的调用走不同管道、不同 busy 锁，互不等待。

---

## 六、状态板：黑板的数据结构与读写

### 写入（set_state, workspace.py:145-159）

每次写入都把**上一个持有者**留底在条目里：

```python
board[key] = {
    "value": value,                                   # 新值（生效）
    "owner": owner,                                   # 这次是谁写的
    "prev_owner": (old or {}).get("owner", ""),       # 上次是谁写的（首写为 ""）
    "prev_value": (old or {}).get("value"),           # 上次的值（首写为 None）
}
```

```python
# 实测：两个 Agent 先后写同一个键
m.set_state("ws-x", "progress", "A做完调研", owner="agent-a")
m.set_state("ws-x", "progress", "B在做演示", owner="agent-b")
# 日志: 工作区 ws-x 键 'progress' 被覆盖: agent-a → agent-b（旧值已留存）

m._state["ws-x"]["progress"]
# → {"value": "B在做演示", "owner": "agent-b",
#    "prev_owner": "agent-a", "prev_value": "A做完调研"}

m.get_state("ws-x", "progress")       # → "B在做演示"（新值生效）
m.snapshot("ws-x")
# → {"progress": {"value": "B在做演示", "owner": "agent-b"}}
#   snapshot 只给 value+owner，不给 prev（对外只要现状，追溯留给自己）
```

| 设计 | 不这么写会怎样 |
|---|---|
| 覆盖前先留 `prev_owner/prev_value` | "B 覆盖了 A 的产出"无据可查，两个 Agent 各执一词时没有裁判证据 |
| 不同 owner 覆盖才告警 | 同一个 Agent 反复更新自己的进度是正常行为，不告警；跨人覆盖才是冲突信号 |
| 键空间按 workspace 物理分开（不同 dict） | 共用一个全局 dict 时，键名撞了就跨项目串数据 |

### 读取（get_state / snapshot）

| 方法 | 返回 | 适用 |
|---|---|---|
| `get_state(ws, key)` | 单个值（不存在给 default） | 偷看某个进度 |
| `snapshot(ws)` | 全部 {键: {value, owner}} | **新 Agent 加入时一次性同步全部上下文** |

---

## 七、为什么要这么写（决策表）

| 做法 | 原因（不这么写会怎样） |
|---|---|
| 按 (workspace, server) 而不是按 server 共享 | 按 server 共享（gateway 的粒度）会让所有工作区看到彼此的文件——"两个项目共用一个工作台"，改错对方的文件 |
| 引用计数归零才 disconnect | 提前断开：还在用的 Agent 下一次调用直接 Connection reset；不断开：子进程永远残留，一个 server 配几十个僵尸进程 |
| disconnect 放到锁外执行 | 断开可能阻塞（等子进程退出）；攥着全局锁等，其他工作区的 acquire 全被卡住 |
| 会话内调用加 `busy` 锁串行化 | stdio 是单管道：两个线程同时写管道，请求字节流交叉，server 解析出乱码还不报错 |
| release 对不存在的会话直接 return | 幂等：Agent 流程里重试/异常路径重复 release 不会炸 |
| call_tool 的便捷分支 acquire/release 配对 | 临时调用者不配对的话，refs 虚高，最后永远归不了零 |
| 覆盖时记 `prev_owner` | 不留底的话，"B 覆盖了 A 的产出"无据可查；留底后一条 warning 日志直接定位 |
| client_factory 可注入 | 测试时传 FakeClient（不真拉子进程）；生产不传则走 gateway 同款造法，口径不漂移 |

---

## 八、与现有 MCPGateway 的关系

| | MCPGateway（gateway.py:364） | MCPWorkspaceManager |
|---|---|---|
| 复用粒度 | server 名 | (workspace, server) 二元组 |
| 适用场景 | 单 Agent / 普通 Agent 调远程工具 | 多 Agent 协作同一工作区 |
| 状态共享 | 无（各调用独立） | 有状态板（黑板） |
| 引用计数 | 无（进程级生命周期） | 有（refs 归零才断） |
| 两者关系 | 单 Agent 照走 gateway，互不干扰 | 并存，按场景选 |

`_default_factory`（workspace.py:74-85）与 `MCPGateway._new_client` 同一套判定：http → MCPHttpClient，stdio → MCPClient 子进程。抽出来共用，避免两处口径漂移——以后改连接判定逻辑只需改 gateway 一处。

多 Agent 之间的产出共享，除本模块外还有一条既有通道——消息总线（multiagent/harness.py:256-260，委派产出 `send` 广播给队友），那是"语言层共享"（说的）；工作区是"会话层共享"（做的，文件/目录真实可见）；两层互补。

---

## 九、边界情况与排查

### 1. 忘了 release（最常见的坑）

```python
# 症状：进程列表里 MCP server 子进程越来越多
m.session_stats()
# → {'ws-x/fs': 2, 'ws-y/fs': 1}     ← 任务都结束了 refs 还是 2？谁忘了 release 一目了然
```

排查口诀：`session_stats()` 里**任务结束后仍不归零的会话**就是泄漏点，顺 refs 找到没配对 release 的 Agent。

### 2. 同一 Agent 反复 acquire

refs 只是计数，不记"是谁"：同一个 Agent acquire 两次，refs=2，要 release 两次才归零。配对原则：**一次 acquire 对一次 release**，最好用 try/finally 包住。

### 3. 覆盖链条只留一层

`prev_owner/prev_value` 只留最近一次：A→B→C 三次写入后，A 的信息已被顶掉（prev_owner=B）。要完整审计链就得改成 list 追加，代价是内存无限涨——本项目取了"够追溯一步"的折中。

---

## 十、简单总结

1. 隔离三层根源：**stdio 一连接一子进程**（协议）→ 子进程内存互不可见（OS）→ N 个 Agent 各 connect = N 个互不相识的会话
2. 共享的实现一句话：会话归属从 Agent 提升到**工作区**，键是 `(workspace, server)` 元组，同区返回同一个 client 对象
3. 生命周期：refs 0→1 时 connect，1→0 时 **锁内除名、锁外断开**；重复 release 幂等
4. 并发安全两把锁各管一摊：**全局锁管两张表**，**会话锁管单管道串行**（stdio 管道并发写会字节交叉且不报错）
5. 状态板 = 黑板：新值生效、旧 owner/旧值留底一层、跨人覆盖才告警；`snapshot` 供新成员一次同步
6. 排查泄漏：`session_stats()` 里永不归零的 refs 就是没配对 release 的地方
