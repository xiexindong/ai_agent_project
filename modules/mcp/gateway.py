"""
MCP 技能网关模块
================
把「多个 MCP Server」收敛成一个统一入口，对外只暴露三件事：
    1. 工具目录 catalog()          —— 有哪些远程技能可用（名字 + 描述 + 参数表）
    2. 按需注册 ensure_registered() —— 只有被选中要用的工具才装进 ToolRegistry
    3. 统一调用 call_tool()         —— 调用方不需要知道工具在哪一个子进程里

【核心思路：MCP 远程技能不直连 LLM】
    改造前的链路：启动时 discover_tools() 全量注册 → 每轮 ReAct 把全部工具 schema 发给模型。
    工具从 5 个涨到 100 个，模型每轮就要在 100 份 schema 里做选择，
    prompt 长度与误选概率同步膨胀（这就是「技能膨胀」）。

    现在的链路：
        MCP Server ──► 网关（纳管 + 建目录）──► Tool-RAG 索引（只存语义，不是 schema）
                                                    │
                                        用户提问 ──► 语义召回 top-k
                                                    │
                                            按需注册 + 权限过滤 ──► LLM（只看到 k 个工具）
    模型看到的永远是「本轮用得上的那几个」，远程技能数量增长不再影响单轮 prompt。

【三个机制的职责边界】
    网关     管「在哪、怎么连、能不能用」   —— 多 server 生命周期、目录缓存、熔断重试
    Tool-RAG 管「本轮该召回谁」             —— 语义相似度排序，见 modules/tools/tool_rag.py
    路由层   管「最终名单与回退」           —— 技能绑定优先、超限时截断，见 modules/agent/router.py
"""
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.common.logger import get_logger
from modules.common.exceptions import MCPException

_logger = get_logger("mcp_gateway")


# 【什么】两种传输方式
# 【作用】stdio = 本机子进程（原有方式）；http = 远程机器上的常驻 MCP Server
TRANSPORT_STDIO = "stdio"
TRANSPORT_HTTP = "http"


# 【什么】一个 MCP Server 的接入配置
# 【作用】登记"叫什么、怎么连、启不启用"，网关据此拉子进程或连远程端点
@dataclass
class MCPServerConfig:
    """一个 MCP Server 的接入配置。

    Attributes:
        name:      逻辑名，用于日志与熔断隔离（不同 server 各自计数，互不连累）。
        command:   启动命令（transport=stdio），如 ["python", "-m", "modules.mcp.mcp_server"]；
                   换成 ["node", "server.js"] 就能接 Node 写的 server。
        enabled:   False 时登记但不连接，用于临时摘掉一个故障 server。
        transport: stdio（本机子进程）/ http（远程 MCP Server）。
        endpoint:  transport=http 时的 /mcp 地址，如 http://10.0.0.7:8800/mcp。
        token:     transport=http 时的 Bearer 令牌，可空。
        timeout_seconds: 单次调用超时；0 = 用全局 TOOL_TIMEOUT_SECONDS。

    【两种传输怎么选】
        stdio  工具跑在本机子进程里 —— 要的是进程隔离，Server 仍在本机。
        http   工具跑在远程机器上 —— 要的是远程执行，也是 MCP 的本意（跨机器、跨语言）。
    """
    name: str                                  # 逻辑名；日志与熔断都按它隔离
    command: List[str] = field(default_factory=list)   # stdio：启动命令
    enabled: bool = True                       # False 时登记但不连接，用于临时摘掉故障 server
    transport: str = TRANSPORT_STDIO           # stdio / http
    endpoint: str = ""                         # http：远程 /mcp 地址
    token: str = ""                            # http：Bearer 令牌
    timeout_seconds: float = 0.0               # 0 = 用全局 TOOL_TIMEOUT_SECONDS

    # 【什么】从配置字典构造配置对象
    # 【作用】兼容 MCP_SERVERS_JSON 里字段缺失的情况；
    #         没写 transport 时按"给了 endpoint 就是 http"推断，老配置一行不用改
    @classmethod
    def from_dict(cls, d: dict) -> "MCPServerConfig":
        """从配置字典构造。

        【输入输出对照】
            输入  {"name": "enterprise", "command": ["python", "-m", "mod"], "enabled": true}
            输出  MCPServerConfig(name="enterprise", command=["python","-m","mod"], enabled=True)
            输入  {"name": "remote", "endpoint": "http://10.0.0.7:8800/mcp"}
            输出  MCPServerConfig(name="remote", transport="http",
                                 endpoint="http://10.0.0.7:8800/mcp")

        【为什么要求 name】name 是熔断与日志的 key；缺失时无法定位是哪个 server 出问题。
        """
        transport = str(d.get("transport", "") or "").strip().lower()
        if not transport:                       # 老配置只有 command；新配置只给 endpoint 也能推断
            transport = TRANSPORT_HTTP if d.get("endpoint") else TRANSPORT_STDIO
        return cls(
            name=str(d.get("name", "")),
            command=list(d.get("command", [])),
            enabled=bool(d.get("enabled", True)),
            transport=transport,
            endpoint=str(d.get("endpoint", "") or ""),
            token=str(d.get("token", "") or ""),
            timeout_seconds=float(d.get("timeout_seconds", 0) or 0),
        )


# GatewayStats 就是网关的「水表电表」——一个只有 6 个计数器的空壳数据类，
# 本身不做任何事，只负责把「目录里有 30 个工具」和「实际只起了 2 次进程、
# 装了 1 个工具」记下来，让 health() 能报出来，用来证明懒加载到底省了多少。
# 
# 
# 【什么】网关的运行计数器
# 【作用】记录"目录有多大 vs 实际起了几次进程"，用来观测懒加载省了多少
@dataclass
class GatewayStats:
    """网关运行计数，用于观测「按需加载到底省了多少」。

    Attributes:
        catalog_tools:  目录中登记的远程工具总数（不是本轮用掉的）
        registered:     已被选中并真正注册进 ToolRegistry 的工具数
        connects:       实际建立子进程连接的次数（懒连接下应远小于 catalog_tools）
        calls:          真实工具调用次数
        failures:       调用失败次数（记入熔断）
    """
    catalog_tools: int = 0   # 目录里登记的远程工具总数
    registered: int = 0      # 已真正注册进 ToolRegistry 的工具数
    connects: int = 0        # 实际建立子进程连接的次数
    calls: int = 0           # 真实工具调用次数
    failures: int = 0        # 调用失败次数
    servers: Dict[str, str] = field(default_factory=dict)   # server 名 → 熔断状态


# 【什么】MCP 技能网关：多 server 纳管 + 懒连接 + 按需注册 + 熔断重试
# 【作用】对外只暴露 catalog / ensure_registered / call_tool，调用方不必知道工具在哪
class MCPGateway:
    """MCP 技能网关：多 server 纳管 + 懒连接 + 按需注册 + 熔断重试。

    【内部三张表】
        _servers     server 名 → MCPServerConfig
        _clients     server 名 → MCPClient（已连接的实例，懒连接下平时是空的）
        _tool_index  工具名   → {"server": 名, "info": 工具元数据}
        _registered  已被注册进 ToolRegistry 的工具名集合（避免重复注册刷日志）

    【懒连接的收益与代价】
        收益：100 个远程工具常驻 = 100 个子进程；现在只有被召回的工具才拉起对应 server。
        代价：首次调用某个 server 要付一次握手时间（实测见 examples/mcp_skill_gateway_demo.py）。
    """

    def __init__(
        self,
        servers: Optional[List[MCPServerConfig]] = None,   # 初始登记的 server 配置列表
        lazy_connect: bool = True,                         # True=拉完目录就断开，调用时再连
        catalog_cache: str = "",                           # 目录缓存文件路径；留空表示不落盘
        connect_retry: int = 2,                            # 单个 server 的连接重试次数（默认 2）
        breaker=None,                                      # 熔断器实例；None 表示不做熔断
    ):
        # 【什么】初始化内部四张表与运行开关
        # 【作用】构造期不建任何子进程连接，全部推迟到 catalog() 或真正调用时
        self._servers: Dict[str, MCPServerConfig] = {}   # server 名 → 接入配置
        # 值是 MCPClient，延迟 import 避免本模块与 mcp_client 的加载顺序耦合，
        # 所以这里标 Any（类型检查交给调用点的显式用法）
        self._clients: Dict[str, Any] = {}
        self._tool_index: Dict[str, dict] = {}           # 工具名 → {"server": 名, "info": 工具元数据}
        self._registered: set = set()                     # 已注册进 ToolRegistry 的工具名，防重复注册
        self._lazy = lazy_connect                         # 懒连接开关
        self._cache_path = Path(catalog_cache) if catalog_cache else None   # 目录缓存文件；留空为 None
        self._retry = max(1, int(connect_retry))          # 重试次数，最小 1 次
        self._breaker = breaker                           # 熔断器；None 时不判熔断
        self._stats = GatewayStats()                      # 运行计数器
        self._catalog_loaded = False                      # 目录是否已建好
        self._lock = threading.Lock()                     # 保护目录表与连接表的线程锁
        for cfg in servers or []:
            self.add_server(cfg)

    # ------------------------------------------------------------------
    # 【登记 server】只记配置，不建连接
    #   连接推迟到 catalog(refresh=True) 或第一次真正要用它的时候。
    # 【什么】登记一个 server 配置
    # 【作用】只写进 _servers 表；缺 name 或 command 直接抛 MCPException
    # ------------------------------------------------------------------
    def add_server(self, cfg: MCPServerConfig) -> None:
        if not cfg.name:
            raise MCPException("MCP Server 配置缺少 name", str(cfg))
        # 【按传输校验必填】stdio 要启动命令，http 要远端地址；两样都没有就不知道怎么连
        if cfg.transport == TRANSPORT_HTTP:
            if not cfg.endpoint:
                raise MCPException("MCP Server 配置缺少 endpoint", f"name={cfg.name}(http)")
        elif not cfg.command:
            raise MCPException("MCP Server 配置缺少 command", f"name={cfg.name}(stdio)")
        if cfg.name in self._servers:
            _logger.warning(f"MCP Server 重复登记，覆盖: {cfg.name}")
        self._servers[cfg.name] = cfg
        _logger.info(f"MCP Server 已登记: {cfg.name} | transport={cfg.transport} | "
                     f"enabled={cfg.enabled}")

    # 【什么】已登记的 server 名列表
    # 【作用】启动日志用它报"纳管了哪几个 server"，不反映是否已连接
    @property
    def server_names(self) -> List[str]:
        return list(self._servers.keys())

    # 【什么】各 server 用的传输方式
    # 【作用】启动日志据此报"本机子进程 / 远程 HTTP"，一眼看出 MCP 工具到底跑在哪
    #   输出形如 {"enterprise-mcp": "stdio", "remote-mcp": "http"}
    @property
    def transports(self) -> Dict[str, str]:
        return {name: cfg.transport for name, cfg in self._servers.items()}

    # ------------------------------------------------------------------
    # 【工具目录】召回的原料：只要名字/描述/参数表，不生成 function schema
    #
    #   【为什么用缓存】建 Tool-RAG 索引需要全部工具的描述文本。若每次启动都
    #   把所有 server 拉起来问一遍，启动时间与子进程数都随工具规模线性增长。
    #   落盘缓存之后：冷启动连一次，之后从缓存建索引，server 一个都不用起。
    #
    #   【缓存不新鲜怎么办】server 侧增删工具后，传 refresh=True 重拉一次即可；
    #   不传的话沿用缓存（工具描述变化很罕见，换来的是启动快）。
    #
    #   【输入输出对照】
    #     输入  一个提供了 weather / translate 两个工具的 server
    #     输出  [{"name":"weather","description":"查天气","inputSchema":{...},"server":"geo"},
    #            {"name":"translate","description":"翻译","inputSchema":{...},"server":"geo"}]
    # 【什么】取全部远程工具的元数据目录
    # 【作用】Tool-RAG 建索引的原料；只含名字/描述/参数表，不含连接的子进程
    # ------------------------------------------------------------------
    def catalog(self, refresh: bool = False) -> List[dict]:
        """返回全部远程工具的元数据（不含任何连接状态假设）。"""
        # ------------------------------------------------------------------
        # 【为什么整段加锁】_tool_index（目录表）与 _clients（连接表）会被并发读写
        #   （API 多线程请求 + 后台刷新），不加锁可能读到"建了一半的目录"。
        # ------------------------------------------------------------------
        with self._lock:
            # ------------------------------------------------------------------
            # 【分支 1：目录已建好】最快路径：直接按内存里的表返回，不起任何连接。
            #   返回的是拷贝（dict(...) 新建）：调用方改动不会污染网关内部表。
            #   server 字段是后加的，用来在日志/路由里回溯"这个工具来自哪个 server"。
            # ------------------------------------------------------------------
            if self._catalog_loaded and not refresh:
                return [dict(e["info"], server=e["server"]) for e in self._tool_index.values()]

            # ------------------------------------------------------------------
            # 【分支 2：读落盘缓存】冷启动只会走到这里一次，之后都是分支 1。
            #   refresh=True 时强制跳过缓存（server 侧增删工具后用它重拉）。
            # ------------------------------------------------------------------
            cached = [] if refresh else self._read_cache()
            if cached:
                for entry in cached:
                    self._tool_index[entry["name"]] = {
                        "server": entry.get("server", ""),                          # 老缓存可能没有 server 字段，兜空串
                        "info": {k: v for k, v in entry.items() if k != "server"},  # 其余字段原样当工具元数据
                    }
                self._catalog_loaded = True
                self._stats.catalog_tools = len(self._tool_index)
                _logger.info(f"MCP 工具目录来自缓存: {self._cache_path} | 工具数={len(self._tool_index)}")
                return [dict(e["info"], server=e["server"]) for e in self._tool_index.values()]

            # ------------------------------------------------------------------
            # 【分支 3：现连现拉】没缓存或要求刷新 —— 这一步才真正拉起子进程。
            #   单个 server 出问题（连不上 / 发现失败）只跳过它，不影响其他 server，
            #   也不抛异常：目录少几个工具远好过整个网关不可用。
            # ------------------------------------------------------------------
            for name, cfg in self._servers.items():
                if not cfg.enabled:                             # 停用状态：登记着但不参与，用于临时摘故障 server
                    continue
                client = self._connect(name)                    # 懒连接下，首次建目录才会真正起进程
                if client is None:
                    continue
                try:
                    tools = client.discover_tools()             # 只要元数据：name/description/inputSchema
                except Exception as e:
                    _logger.warning(f"MCP 工具发现失败，跳过该 server: {name} | {e}")
                    continue
                for t in tools:
                    self._tool_index[t["name"]] = {"server": name, "info": t}   # 同名工具：后连的 server 覆盖先连的
                # ------------------------------------------------------------------
                # 【拉完目录就断开】目录是元数据，不需要子进程常驻；
                #   真正调用时 _connect 会重新拉起（同一份配置，行为一致）。
                #   lazy=False 时保留连接，用于「调用极频繁、不想反复握手」的场景。
                #   http 传输没有子进程要养，断开只是白扔一次握手，所以不断。
                # ------------------------------------------------------------------
                if self._lazy and cfg.transport != TRANSPORT_HTTP:
                    client.disconnect()
                    self._clients.pop(name, None)                # 连 _clients 里的记录一起清，下次 _connect 重新拉起

            self._catalog_loaded = True
            self._stats.catalog_tools = len(self._tool_index)
            self._write_cache()                                  # 落盘：下次启动走分支 2，一个 server 都不用起
            _logger.info(f"MCP 工具目录已建立: 工具数={len(self._tool_index)} | servers={self.server_names}")
            return [dict(e["info"], server=e["server"]) for e in self._tool_index.values()]

    # ------------------------------------------------------------------
    # 【按需加载】召回名单里属于远程工具的部分，此刻才真正连 server + 注册
    #
    #   【为什么不在启动时全注册】注册 = 进 ToolRegistry = 后续每轮都可能被
    #   全量塞进 prompt。注册这一步必须发生在「已经决定本轮要用它」之后。
    #
    #   【输入输出对照】
    #     输入  registry（已含本地 calculator/data_query），names=["weather","calculator"]
    #     输出  返回 ["weather"]（真正新注册的）；"calculator" 已在 registry 中，跳过
    # 【什么】把召回名单里的远程工具真正装进 ToolRegistry
    # 【作用】只装"本轮已决定要用"的工具，返回本次新注册的名字
    # ------------------------------------------------------------------
    def ensure_registered(self, registry, names: List[str]) -> List[str]:
        if registry is None or not names:
            return []
        from modules.mcp.mcp_client import MCPToolAdapter
        newly: List[str] = []
        for name in names:
            if name in self._registered:
                continue
            entry = self._tool_index.get(name)
            if not entry:
                continue                      # 本地工具或目录中不存在的名字，交给 registry 自己处理
            server = entry["server"]
            client = self._connect(server)
            if client is None:
                continue
            try:
                registry.register(MCPToolAdapter(client, entry["info"]))
                self._registered.add(name)
                newly.append(name)
            except Exception as e:
                _logger.warning(f"MCP 工具注册失败: {name} | {e}")
        if newly:
            self._stats.registered = len(self._registered)
            _logger.info(f"MCP 工具按需加载: {newly}")
        return newly

    # 【什么】统一调用入口：只给工具名，网关自己找它在哪个 server
    # 【作用】调用成功后给熔断记一次 success；抛异常时按异常类型决定是否记 failure
    def call_tool(self, name: str, arguments: dict) -> dict:
        """统一调用入口：调用方只给工具名，网关负责找到它属于哪个 server。"""
        entry = self._tool_index.get(name)
        if not entry:
            raise MCPException(f"网关未登记该工具: {name}", f"已登记 {len(self._tool_index)} 个")
        server = entry["server"]
        client = self._connect(server)
        if client is None:
            raise MCPException("MCP Server 不可用", f"server={server}, tool={name}")
        self._stats.calls += 1
        try:
            # call_tool 上的 handle_exception 装饰器失败时返回 None，
            # 这里兜成空 dict，调用方不必到处判空（空内容 = 工具没返回文本）
            result = client.call_tool(name, arguments) or {}
            if self._breaker:
                self._breaker.record_success(self._cb_key(server))
            return result
        except Exception as e:
            self._stats.failures += 1
            if self._breaker:
                # 【只记瞬时故障】工具业务性报错（参数非法）不算 server 不可用，
                #   否则一次参数错误就能把整个 server 熔断掉。
                if isinstance(e, MCPException) and "参数" not in str(e):
                    self._breaker.record_failure(self._cb_key(server))
            raise

    # ------------------------------------------------------------------
    # 【连接管理】带熔断 + 重试，失败返回 None 而不是抛出
    #
    #   【为什么这里返回 None 而不是抛异常】目录构建阶段一个 server 挂掉，
    #   不应该让整个网关不可用（其余 server 的工具照样能召回）—— 局部故障局部降级。
    #
    #   【重试次数】默认 2 次（配置 MCP_CONNECT_RETRY）：治的是子进程冷启动偶发抖动；
    #   持续故障交给熔断，避免每次请求都白等一轮超时。
    # 【什么】取（必要时新建）某个 server 的客户端连接
    # 【作用】带缓存复用、熔断拦截与重试；连不上返回 None 而不是抛异常
    # ------------------------------------------------------------------
    def _connect(self, server: str):
        # ------------------------------------------------------------------
        # 【复用已有连接】_clients 里还活着就直接返回，不再起一次子进程。
        #   is_connected 用 getattr 取：http 客户端也有同名字段，两种传输共用这条路径。
        # ------------------------------------------------------------------
        existing = self._clients.get(server)
        if existing is not None and getattr(existing, "is_connected", False):
            return existing

        # 【查配置】没登记或已停用 → 返回 None（调用方据此跳过，不抛异常）
        cfg = self._servers.get(server)
        if cfg is None or not cfg.enabled:
            return None

        # 【熔断前置检查】这个 server 刚连续失败过就直接不连，避免每次请求都去撞一次超时；
        #   熔断按 "mcp:{server}" 计数（_cb_key），不同 server 互不连累。
        if self._breaker and not self._breaker.allow(self._cb_key(server)):
            _logger.warning(f"MCP Server 熔断中，跳过连接: {server}")
            return None

        # ------------------------------------------------------------------
        # 【带重试的连接】最多 self._retry 次（构造时已 max(1,...) 兜底，至少 1 次）。
        #   每次都新建 client：上一次失败的连接对象可能处于半连状态，复用不安全。
        #   成功后记录 connects + 熔断成功；全部失败才记一次熔断失败（不是每次尝试都记，
        #   否则重试反而会加速熔断）。
        # ------------------------------------------------------------------
        last_err = None
        for attempt in range(1, self._retry + 1):
            client = self._new_client(server, cfg)          # http → MCPHttpClient；stdio → MCPClient（起子进程）
            try:
                client.connect()
                self._clients[server] = client              # 放进连接表：后续调用与 catalog 断开逻辑都认这张表
                self._stats.connects += 1                   # 观测用：懒加载省下的就是它
                if self._breaker:
                    self._breaker.record_success(self._cb_key(server))   # 连上了就复位该 server 的失败计数
                _logger.info(f"MCP Server 已连接: {server} | 第 {attempt} 次尝试")
                return client
            except Exception as e:
                last_err = e                                # 只留最后一次错误，最终日志用它
                try:
                    client.disconnect()                     # 清理半连状态的子进程/套接字
                except Exception:
                    pass                                    # 断开本身失败无所谓，不能盖掉真正的连接错误
                _logger.warning(f"MCP Server 连接失败({attempt}/{self._retry}): {server} | {e}")

        # 【全部重试失败】记熔断 + error 日志，返回 None 让调用方降级（跳过该 server）
        if self._breaker:
            self._breaker.record_failure(self._cb_key(server))
        _logger.error(f"MCP Server 连接最终失败: {server} | {last_err}")
        return None

    # 【什么】按传输方式 new 一个客户端
    # 【作用】stdio → 本机子进程 MCPClient；http → 远程 MCPHttpClient；
    #         两个类方法名一致（connect/discover_tools/call_tool/disconnect），
    #         所以下面的连接流程与上层调用一行都不用改
    @staticmethod
    def _new_client(server: str, cfg: "MCPServerConfig"):
        if cfg.transport == TRANSPORT_HTTP:
            from modules.mcp.mcp_http_client import MCPHttpClient
            return MCPHttpClient(endpoint=cfg.endpoint, name=server,
                                 token=cfg.token, timeout_seconds=cfg.timeout_seconds)
        from modules.mcp.mcp_client import MCPClient
        return MCPClient(server_command=list(cfg.command), name=server)

    # 【什么】熔断键：给 server 名加前缀
    # 【作用】熔断按 "mcp:{server}" 计数，不同 server 各自统计、互不连累
    @staticmethod
    def _cb_key(server: str) -> str:
        return f"mcp:{server}"

    # ------------------------------------------------------------------
    # 【目录缓存落盘】JSON 结构：{"version":1,"tools":[{name,description,inputSchema,server}]}
    #   写失败只记 warning —— 缓存是加速手段，不是正确性依赖，写不进去就下次重连。
    # 【什么】读落盘的目录缓存
    # 【作用】命中就直接建索引，一个子进程都不用起；读失败返回 [] 改为实时发现
    # ------------------------------------------------------------------
    def _read_cache(self) -> List[dict]:
        if not self._cache_path or not self._cache_path.exists():
            return []
        try:
            # ------------------------------------------------------------------
            # 【什么】open()：把磁盘上的一个文件"打开"，返回一个可往下读的句柄 f
            #   open(路径, 模式, encoding) 三个入参逐个说：
            #     self._cache_path —— 缓存文件路径（Path 对象，__init__ 第 133 行由 catalog_cache 转来）
            #     "r"              —— 只读模式（read）；文件不存在会抛 FileNotFoundError
            #     encoding="utf-8" —— 按 UTF-8 把字节解码成字符
            #
            # 【with 的作用】with 块一结束（正常走完 / 中途抛异常）都会自动 f.close()
            #   不用 with 就得自己写 try/finally 关文件；漏一次 close 就漏一个文件句柄，
            #   Windows 上还会一直锁着这个文件，导致删不掉、别人也写不进去
            #
            # 【输入输出对照】
            #   输入：data/mcp_catalog.json（config.py:128 的默认值）
            #         {"version": 1, "tools": [{"name": "web_search", "server": "enterprise-mcp"}]}
            #   输出：f —— 文件对象；交给下一行 json.load(f) 读完得到 dict
            #
            # 【依赖警告】这里的 encoding 必须与 _write_cache 里 open(..., "w", encoding="utf-8") 一致，
            #   两边不一致时中文工具描述会解码失败，被下面的 except 接住并降级为「实时发现」
            # ------------------------------------------------------------------
            with open(self._cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # ------------------------------------------------------------------
            # 【tools 是哪来的】就是上面 data 里的 "tools" 字段
            #   链条：_write_cache（第 384 行）把 _tool_index 落成
            #         {"version":1,"tools":[{name,description,inputSchema,server}, ...]}
            #         _tool_index 又来自两条路：本文件缓存命中（第 188 行）或
            #         实时 discover_tools（第 205 行，需真连一次 server）
            #
            # 【为什么用 .get 带兜底】文件被手改坏、只写了半截时可能没有 "tools" 键，
            #   写 data["tools"] 会抛 KeyError；.get(..., []) 缺键就当空目录，
            #   由调用方 catalog() 走实时发现那条路补上
            #
            # 【输入输出对照】
            #   输入：data = {"version":1,"tools":[{"name":"web_search","server":"enterprise-mcp"}, ...]}
            #   输出：tools = [{"name":"web_search","server":"enterprise-mcp"}, ...]
            #   缺键时：data = {"version":1}  ->  tools = []
            # ------------------------------------------------------------------
            tools = data.get("tools", [])
            _logger.info(f"MCP 目录缓存命中: {self._cache_path} | 工具数={len(tools)}")
            return tools
        except Exception as e:
            _logger.warning(f"MCP 目录缓存读取失败，改为实时发现: {e}")
            return []

    # 【什么】把当前目录写进缓存文件
    # 【作用】下次启动可直接复用；没配路径或写失败都只记 warning，不影响本次运行
    def _write_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "tools": [dict(e["info"], server=e["server"]) for e in self._tool_index.values()],
            }
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            _logger.warning(f"MCP 目录缓存写入失败（不影响本次运行）: {e}")

    # ------------------------------------------------------------------
    # 【观测与收尾】
    # 【什么】网关健康快照
    # 【作用】报各 server 的熔断状态与五项计数，用来确认懒加载是否生效
    # ------------------------------------------------------------------
    def health(self) -> dict:
        """网关健康快照：各 server 的熔断状态 + 目录/注册/调用计数。"""
        servers = {}
        for name, cfg in self._servers.items():
            state = "disabled"
            if cfg.enabled:
                state = self._breaker.state(self._cb_key(name)) if self._breaker else "unknown"
            servers[name] = state
        self._stats.servers = servers
        return {
            "servers": servers,
            "catalog_tools": len(self._tool_index),
            "registered": len(self._registered),
            "connects": self._stats.connects,
            "calls": self._stats.calls,
            "failures": self._stats.failures,
        }

    # ------------------------------------------------------------------
    # 【按配置装配】多 server 来自 MCP_SERVERS_JSON；留空时用默认单 server
    #
    #   【输入输出对照】
    #     输入  MCP_SERVERS_JSON='[{"name":"geo","command":["python","-m","geo_server"]}]'
    #     输出  网关登记 1 个 server：geo（stdio，本机子进程）
    #     输入  MCP_SERVERS_JSON='[{"name":"remote","endpoint":"http://10.0.0.7:8800/mcp"}]'
    #     输出  网关登记 1 个 server：remote（http，远程机器上的常驻 Server）
    #     输入  MCP_REMOTE_URL='http://10.0.0.7:8800/mcp'（MCP_SERVERS_JSON 留空）
    #     输出  网关登记 1 个 server：remote-mcp（http）
    #     输入  两者都留空
    #     输出  网关登记默认 server：enterprise-mcp（与改造前同一个）
    #
    #   【配置项解析失败怎么处理】JSON 写错时记 warning 并回退默认 server，
    #   不让一个配置笔误导致 MCP 整体不可用 —— 配置错误属于可降级故障。
    # 【什么】按配置装配网关（最常用的构造方式）
    # 【作用】MCP_SERVERS_JSON 解析失败或留空时，回退到单个默认 server
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings=None, breaker=None, default_command: Optional[List[str]] = None) -> "MCPGateway":
        if settings is None:
            from config import get_settings
            settings = get_settings()
        servers: List[MCPServerConfig] = []
        raw: str = str(getattr(settings, "mcp_servers_json", "") or "").strip()
        if raw:
            try:
                servers = [MCPServerConfig.from_dict(d) for d in json.loads(raw)]
            except Exception as e:
                _logger.warning(f"MCP_SERVERS_JSON 解析失败，回退默认 server: {e}")
                servers = []
        # 【远程 MCP Server】只配一个 MCP_REMOTE_URL 就能接远程机器上的 Server，
        #   不必写完整的 MCP_SERVERS_JSON（与沙箱远程地址的用法保持一致）
        remote_url = str(getattr(settings, "mcp_remote_url", "") or "").strip()
        if not servers and remote_url:
            servers = [MCPServerConfig(
                name="remote-mcp", transport=TRANSPORT_HTTP, endpoint=remote_url,
                token=str(getattr(settings, "mcp_remote_token", "") or ""),
            )]
        if not servers and default_command:
            servers = [MCPServerConfig(name="enterprise-mcp", command=list(default_command))]
        return cls(
            servers=servers,
            lazy_connect=bool(getattr(settings, "mcp_lazy_connect", True)),
            catalog_cache=str(getattr(settings, "mcp_catalog_cache", "") or ""),
            connect_retry=int(getattr(settings, "mcp_connect_retry", 2) or 1),
            breaker=breaker,
        )

    # 【什么】断开全部已连接 server 并清空连接表
    # 【作用】进程退出前调用，避免留下孤儿 MCP 子进程
    def shutdown(self) -> None:
        """断开全部已连接的 server（进程退出前调用，避免残留子进程）。"""
        for name, client in list(self._clients.items()):
            try:
                client.disconnect()
            except Exception as e:
                _logger.warning(f"MCP Server 断开异常: {name} | {e}")
        self._clients.clear()
        _logger.info("MCP 网关已关闭全部连接")
