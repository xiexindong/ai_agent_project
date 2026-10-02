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
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from modules.common.logger import get_logger

_logger = get_logger("mcp.workspace")


# 【什么】一个 (workspace, server) 组合的共享会话条目
# 【作用】持有共享 client、引用计数和这把组合的锁
class _WorkspaceSession:
    def __init__(self, client: Any):
        self.client = client          # 共享的 MCP 客户端（同一子进程 = 同一 MCP 会话）
        self.refs = 0                 # 当前有几个 Agent 在用；归零才允许断开
        self.busy = threading.Lock()  # 串行化对同一会话的工具调用（stdio 单管道不容并发写）


# 【什么】MCP 共享工作区管理器
# 【作用】workspace → (server → 共享会话) 的两级映射 + 工作区共享状态板
class MCPWorkspaceManager:
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

    def __init__(self, client_factory: Optional[Callable[[str], Any]] = None):
        # client_factory(server_name) → 已 connect 的客户端。
        # 不传则用 gateway 的标准造法（http/stdio 二选一），与 MCPGateway._connect 同源。
        self._factory = client_factory or self._default_factory
        self._sessions: Dict[Tuple[str, str], _WorkspaceSession] = {}   # (workspace, server) → 会话
        # 共享状态板：workspace → {键: {"value": 值, "owner": 谁写的}}
        self._state: Dict[str, Dict[str, dict]] = {}
        self._lock = threading.Lock()      # 保护 _sessions / _state 两张表

    # ------------------------------------------------------------------
    # 【默认造客户端】与 MCPGateway._new_client 同一套判定：http → MCPHttpClient，
    #   stdio → MCPClient 子进程。抽出来共用，避免两处口径漂移。
    # ------------------------------------------------------------------
    @staticmethod
    def _default_factory(server: str) -> Any:
        # from_settings() 返回装配好的 MCPGateway 实例，其 _servers 是
        # {server 名: MCPServerConfig}；取到配置后用 gateway 的静态方法造客户端
        from modules.mcp.gateway import MCPGateway
        gw = MCPGateway.from_settings()
        conf = gw._servers.get(server)
        if conf is None:
            from modules.common.exceptions import MCPException
            raise MCPException(f"工作区工厂找不到 server 配置: {server}")
        client = MCPGateway._new_client(server, conf)
        client.connect()
        return client

    # 【什么】加入工作区（取/建该 (workspace, server) 的共享会话）
    # 【作用】引用计数 +1；首次创建时 connect，返回共享 client
    def acquire(self, workspace: str, server: str) -> Any:
        key = (workspace, server)
        with self._lock:
            sess = self._sessions.get(key)
            if sess is None:
                client = self._factory(server)       # 工厂负责 connect；失败直接抛 MCPException
                sess = _WorkspaceSession(client)
                self._sessions[key] = sess
                _logger.info(f"MCP 工作区会话创建: workspace={workspace} server={server}")
            sess.refs += 1
            return sess.client

    # 【什么】退出工作区（引用计数 -1，归零才断开）
    # 【作用】防止"A 刚退出就把 B 还在用的连接拆了"
    def release(self, workspace: str, server: str) -> None:
        key = (workspace, server)
        with self._lock:
            sess = self._sessions.get(key)
            if sess is None:
                return
            sess.refs -= 1
            if sess.refs <= 0:
                del self._sessions[key]
                client = sess.client
            else:
                return
        # 断开放到锁外：disconnect 可能阻塞（等子进程退出），别攥着全局锁等
        try:
            client.disconnect()
            _logger.info(f"MCP 工作区会话关闭: workspace={workspace} server={server}")
        except Exception as e:                        # 断不开只记日志：子进程残留好过拖死调用方
            _logger.warning(f"MCP 工作区会话断开失败(忽略): {e}")

    # 【什么】在共享会话上调用工具（自动排队）
    # 【作用】stdio 是单管道，同一会话的两次并发写会交叉出错——用会话锁串行化
    def call_tool(self, workspace: str, server: str, tool: str, arguments: dict) -> dict:
        key = (workspace, server)
        with self._lock:
            sess = self._sessions.get(key)
            if sess is None:
                # 没加入就调用：自动 acquire，调完自动 release（便捷用法）
                client = self.acquire(workspace, server)
                sess = self._sessions[key]
                try:
                    with sess.busy:
                        return client.call_tool(tool, arguments) or {}
                finally:
                    self.release(workspace, server)
        with sess.busy:                                # 已在会话里：排队调用
            return sess.client.call_tool(tool, arguments) or {}

    # ------------------------------------------------------------------
    # 【共享状态板】工作区内 Agent 间的"黑板"：谁写了什么，一看便知
    # ------------------------------------------------------------------
    # 【什么】往工作区状态板写一个键
    # 【作用】带 owner 标记，后写覆盖先写，但保留被覆盖者，冲突可追溯
    def set_state(self, workspace: str, key: str, value: Any, owner: str = "") -> None:
        with self._lock:
            board = self._state.setdefault(workspace, {})
            old = board.get(key)
            board[key] = {
                "value": value,
                "owner": owner,
                "prev_owner": (old or {}).get("owner", ""),   # 被覆盖前是谁写的
                "prev_value": (old or {}).get("value"),       # 被覆盖前的旧值（None=首次写入）
            }
            if old is not None and old.get("owner") and old["owner"] != owner:
                _logger.warning(
                    f"工作区 {workspace} 键 '{key}' 被覆盖: "
                    f"{old['owner']} → {owner}（旧值已留存）"
                )

    # 【什么】读工作区状态板的一个键
    # 【作用】跨 Agent 看见彼此产出；不存在的键返回默认值
    def get_state(self, workspace: str, key: str, default: Any = None) -> Any:
        with self._lock:
            board = self._state.get(workspace, {})
            entry = board.get(key)
            return entry["value"] if entry else default

    # 【什么】整块读工作区状态板
    # 【作用】新加入的 Agent 一次性看到全部共享上下文
    def snapshot(self, workspace: str) -> Dict[str, Any]:
        with self._lock:
            board = self._state.get(workspace, {})
            return {k: {"value": v["value"], "owner": v["owner"]} for k, v in board.items()}

    # 【什么】当前各会话的引用情况（观测用）
    # 【作用】排查"谁忘了 release"：refs 永不归零的会话就是泄漏点
    def session_stats(self) -> Dict[str, int]:
        with self._lock:
            return {f"{ws}/{srv}": s.refs for (ws, srv), s in self._sessions.items()}
