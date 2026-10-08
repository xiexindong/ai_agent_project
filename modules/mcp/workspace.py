"""
MCP 共享工作区模块
==================
回答"MCP 协议默认每个客户端与服务端是独立会话、状态互相不可见，
为什么多 Agent 框架能让多个 Agent 在同一个 MCP 工作区协作且互不冲突"：

    MCPClient 每连一次 = 独立子进程 = 独立会话（stdio 一进程一会话）。
    让多个 Agent 各自 connect 同一个 server，得到的是【N 个互不相识的会话】，
    谁在文件系统 / 服务端留下的状态对方都看不到。

    MCPWorkspaceManager 的做法：把"会话"的粒度从 Agent 提升到【工作区】——
        同一 workspace 的所有 Agent 共享同一个 client（同一子进程、同一会话）；
        不同 workspace 各拿各的 client（互不可见，天然隔离）。
    再配一块工作区级共享状态板（键值对），让 Agent 之间"看得见彼此的产出"。

    互不冲突的保证：
        ① 连接层：一工作区一 client + 引用计数，最后一个用完才真正 disconnect；
        ② 状态层：共享板每写一个键带持有者标记，覆盖前留旧值来源，冲突可追溯；
        ③ 隔离层：跨 workspace 的键空间物理分开（不同 dict），不存在串工作区。

【与 gateway 的关系】MCPGateway 是"按 server 复用连接"（全进程共享一张连接表），
    粒度是 server；本模块粒度是 (workspace, server)，给多 Agent 协作场景用，
    两者并存不冲突：单 Agent 照走 gateway，多 Agent 共用工作区走这里。
"""
import threading                                                    # 标准库：线程锁（保护共享表 + 串行化会话调用）
from typing import Any, Callable, Dict, Optional, Tuple             # 类型标注工具

from modules.common.logger import get_logger                        # 项目统一日志入口

_logger = get_logger("mcp.workspace")                               # 模块级日志器，名字 "mcp.workspace" 用于日志过滤


# 【什么】一个 (workspace, server) 组合的共享会话条目
# 【作用】持有共享 client、引用计数和这把组合的锁
# 【为什么】三样东西必须绑在一起：拆开就会出"连接活着但锁丢了"的并发事故
class _WorkspaceSession:                                            # 工作区会话条目（内部类，不对外暴露）
    # 【什么】会话条目构造
    # 【作用】登记共享 client 并把引用计数清零、锁就位
    # 【为什么】引用计数从 0 起：acquire 里先建条目再 +1，逻辑统一
    def __init__(self, client: Any):                                # 入参为已连接的 MCP 客户端
        self.client = client          # 共享的 MCP 客户端（同一子进程 = 同一 MCP 会话）
        self.refs = 0                 # 当前有几个 Agent 在用；归零才允许断开
        self.busy = threading.Lock()  # 串行化对同一会话的工具调用（stdio 单管道不容并发写）


# 【什么】MCP 共享工作区管理器
# 【作用】workspace → (server → 共享会话) 的两级映射 + 工作区共享状态板
# 【为什么】多 Agent 协作需要共享会话与共享黑板，且跨工作区必须物理隔离
class MCPWorkspaceManager:                                          # 共享工作区管理器类
    """多 Agent 共享 MCP 工作区：同工作区共享会话，跨工作区完全隔离。

    【输入输出对照】两个 Agent 加入 workspace "project-x" 用 server "fs"：
        agent_a.acquire("project-x", "fs") → 首次：创建 client 并 connect，refs=1
        agent_b.acquire("project-x", "fs") → 复用同一个 client，refs=2（不起第二个子进程）
        agent_a.release("project-x", "fs") → refs=1，连接保持
        agent_b.release("project-x", "fs") → refs=0，真正 disconnect，子进程退出

        另一个工作区 "project-y" 同时 acquire("project-y", "fs")：
        → 拿到的是完全独立的另一个 client/子进程，看不到 project-x 的任何状态。

    【不这么写会怎样】每个 Agent 各自 connect：N 个 Agent = N 个子进程 = N 份内存，
        且 Agent A 用文件工具建的目录，Agent B 的会话里根本不存在——"协作"名存实亡。
    """

    # 【什么】管理器构造
    # 【作用】备好会话表、状态板与全局锁；客户端工厂可注入，便于测试替换
    # 【为什么】工厂模式：造客户端的逻辑（怎么连）与用法（何时连）解耦
    def __init__(self, client_factory: Optional[Callable[[str], Any]] = None):  # 入参为可选的客户端工厂函数
        # client_factory(server_name) → 已 connect 的客户端。
        # 不传则用 gateway 的标准造法（http/stdio 二选一），与 MCPGateway._connect 同源。
        self._factory = client_factory or self._default_factory     # 工厂函数：未注入时用默认造法
        self._sessions: Dict[Tuple[str, str], _WorkspaceSession] = {}   # (workspace, server) → 会话
        # 共享状态板：workspace → {键: {"value": 值, "owner": 谁写的}}
        self._state: Dict[str, Dict[str, dict]] = {}                # 工作区级共享状态板（两级字典）
        self._lock = threading.Lock()      # 保护 _sessions / _state 两张表

    # ------------------------------------------------------------------
    # 【默认造客户端】与 MCPGateway._new_client 同一套判定：http → MCPHttpClient，
    #   stdio → MCPClient 子进程。抽出来共用，避免两处口径漂移。
    # 【什么】默认客户端工厂
    # 【作用】按网关配置造出已连接的客户端；找不到配置抛 MCPException
    # 【为什么】复用 gateway 的装配与造法：传输判定只有一份，两边永不走岔
    # ------------------------------------------------------------------
    @staticmethod                                                   # 不依赖实例状态，声明为静态方法
    def _default_factory(server: str) -> Any:                       # 入参 server 名，返回已连接的客户端
        # from_settings() 返回装配好的 MCPGateway 实例，其 _servers 是
        # {server 名: MCPServerConfig}；取到配置后用 gateway 的静态方法造客户端
        from modules.mcp.gateway import MCPGateway                  # 延迟 import：避免与 gateway 形成模块级循环依赖
        gw = MCPGateway.from_settings()                             # 按全局配置装配一个临时网关（只为拿配置表）
        conf = gw._servers.get(server)                              # 查该 server 的接入配置
        if conf is None:                                            # 没登记过这个 server
            from modules.common.exceptions import MCPException      # 延迟 import 异常类型
            raise MCPException(f"工作区工厂找不到 server 配置: {server}")  # 抛错并附 server 名
        client = MCPGateway._new_client(server, conf)               # 复用网关的造法：按传输 new 出客户端
        client.connect()                                            # 建立连接（stdio 起子进程 / http 握手）
        return client                                               # 返回已连接客户端

    # 【什么】加入工作区（取/建该 (workspace, server) 的共享会话）
    # 【作用】引用计数 +1；首次创建时 connect，返回共享 client
    # 【为什么】引用计数是"最后一个走的人关灯"的前提，必须原子地查-改
    def acquire(self, workspace: str, server: str) -> Any:          # 入参：工作区名+server 名，返回共享 client
        key = (workspace, server)                                   # 组合键：会话的粒度单位
        with self._lock:                                            # 进临界区：查表/建表/计数必须原子
            sess = self._sessions.get(key)                          # 查会话表
            if sess is None:                                        # 该组合还没有会话
                client = self._factory(server)       # 工厂负责 connect；失败直接抛 MCPException
                sess = _WorkspaceSession(client)                    # 建新会话条目（refs 从 0 起）
                self._sessions[key] = sess                          # 登记进会话表
                _logger.info(f"MCP 工作区会话创建: workspace={workspace} server={server}")  # 记创建日志
            sess.refs += 1                                          # 引用计数 +1（新会话与复用会话走同一步）
            return sess.client                                      # 返回共享客户端

    # 【什么】退出工作区（引用计数 -1，归零才断开）
    # 【作用】防止"A 刚退出就把 B 还在用的连接拆了"
    # 【为什么】断开动作放到锁外：disconnect 可能阻塞，攥着全局锁会拖死所有工作区
    def release(self, workspace: str, server: str) -> None:         # 入参：工作区名+server 名，无返回
        key = (workspace, server)                                   # 组合键
        with self._lock:                                            # 进临界区：计数与删表必须原子
            sess = self._sessions.get(key)                          # 查会话表
            if sess is None:                                        # 没有这个会话（多调了一次 release）
                return                                              # 静默返回，幂等
            sess.refs -= 1                                          # 引用计数 -1
            if sess.refs <= 0:                                      # 已是最后一个使用者
                del self._sessions[key]                             # 先从表里摘掉（断开期间别人查不到，会自建新会话）
                client = sess.client                                # 记下要断开的客户端，带到锁外
            else:                                                   # 还有别人在用
                return                                              # 只减计数，不动连接
        # 断开放到锁外：disconnect 可能阻塞（等子进程退出），别攥着全局锁等
        try:
            client.disconnect()                                     # 真正断开（杀子进程/关连接）
            _logger.info(f"MCP 工作区会话关闭: workspace={workspace} server={server}")  # 记关闭日志
        except Exception as e:                        # 断不开只记日志：子进程残留好过拖死调用方
            _logger.warning(f"MCP 工作区会话断开失败(忽略): {e}")   # 记警告

    # 【什么】在共享会话上调用工具（自动排队）
    # 【作用】stdio 是单管道，同一会话的两次并发写会交叉出错——用会话锁串行化
    # 【为什么】没加入也能直接调（自动 acquire/release）：便捷用法与显式用法两条路都通
    def call_tool(self, workspace: str, server: str, tool: str, arguments: dict) -> dict:  # 入参：工作区+server+工具名+参数
        key = (workspace, server)                                   # 组合键
        with self._lock:                                            # 进临界区查会话表
            sess = self._sessions.get(key)                          # 查会话
            if sess is None:                                        # 没加入就调用
                # 没加入就调用：自动 acquire，调完自动 release（便捷用法）
                client = self.acquire(workspace, server)            # 自动加入（注意：acquire 内部会重入同一把锁之外的路径，这里先出隐含的查表流程）
                sess = self._sessions[key]                          # 取回刚建好的会话条目
                try:
                    with sess.busy:                                 # 持会话锁排队
                        return client.call_tool(tool, arguments) or {}  # 调用，None 兜成空 dict
                finally:
                    self.release(workspace, server)                 # 调完自动退出（配对 acquire）
        with sess.busy:                                # 已在会话里：排队调用
            return sess.client.call_tool(tool, arguments) or {}     # 调用，None 兜成空 dict

    # ------------------------------------------------------------------
    # 【共享状态板】工作区内 Agent 间的"黑板"：谁写了什么，一看便知
    # ------------------------------------------------------------------
    # 【什么】往工作区状态板写一个键
    # 【作用】带 owner 标记，后写覆盖先写，但保留被覆盖者，冲突可追溯
    # 【为什么】多 Agent 并发写同一键是常态：不留旧值来源，冲突就成了无头案
    def set_state(self, workspace: str, key: str, value: Any, owner: str = "") -> None:  # 入参：工作区+键+值+持有者
        with self._lock:                                            # 进临界区保护状态板
            board = self._state.setdefault(workspace, {})           # 取该工作区的板子，没有就建一块
            old = board.get(key)                                    # 取旧条目（用于留痕）
            board[key] = {                                          # 写入新条目
                "value": value,                                     # 新值
                "owner": owner,                                     # 谁写的
                "prev_owner": (old or {}).get("owner", ""),   # 被覆盖前是谁写的
                "prev_value": (old or {}).get("value"),       # 被覆盖前的旧值（None=首次写入）
            }
            if old is not None and old.get("owner") and old["owner"] != owner:  # 发生跨持有者覆盖
                _logger.warning(                                    # 记覆盖警告
                    f"工作区 {workspace} 键 '{key}' 被覆盖: "
                    f"{old['owner']} → {owner}（旧值已留存）"
                )

    # 【什么】读工作区状态板的一个键
    # 【作用】跨 Agent 看见彼此产出；不存在的键返回默认值
    # 【为什么】只暴露 value 不暴露元数据：读者关心内容，持有者信息留给 snapshot
    def get_state(self, workspace: str, key: str, default: Any = None) -> Any:  # 入参：工作区+键+默认值
        with self._lock:                                            # 进临界区保护状态板
            board = self._state.get(workspace, {})                  # 取该工作区的板子，没有按空板处理
            entry = board.get(key)                                  # 取条目
            return entry["value"] if entry else default             # 有条目返回值，否则返回默认值

    # 【什么】整块读工作区状态板
    # 【作用】新加入的 Agent 一次性看到全部共享上下文
    # 【为什么】返回的是重组的新 dict（只含 value/owner）：调用方改不动内部板子
    def snapshot(self, workspace: str) -> Dict[str, Any]:           # 入参：工作区名，返回键→值/持有者 的快照
        with self._lock:                                            # 进临界区保护状态板
            board = self._state.get(workspace, {})                  # 取该工作区的板子
            return {k: {"value": v["value"], "owner": v["owner"]} for k, v in board.items()}  # 字典推导：逐键重组快照（丢弃 prev_* 留痕字段）

    # 【什么】当前各会话的引用情况（观测用）
    # 【作用】排查"谁忘了 release"：refs 永不归零的会话就是泄漏点
    # 【为什么】引用计数是内部状态，观测接口是发现泄漏的唯一窗口
    def session_stats(self) -> Dict[str, int]:                      # 返回 "workspace/server" → 引用计数 的字典
        with self._lock:                                            # 进临界区保护会话表
            return {f"{ws}/{srv}": s.refs for (ws, srv), s in self._sessions.items()}  # 字典推导：拼键并取各会话引用数
