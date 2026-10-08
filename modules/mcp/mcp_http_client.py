"""
MCP Client（HTTP 传输）：连接「远程机器上」的 MCP Server
========================================================
项目原有的 MCPClient 走 stdio：在本机 `subprocess.Popen` 拉起一个 MCP Server 子进程。
那解决的是「进程隔离」，不是「远程」——Server 仍然跑在本机。

本模块解决真正的远程：远端机器部署一个常驻 MCP Server（HTTP），
Agent 侧作为 MCP Client 用 JSON-RPC 2.0 over HTTP 调它的工具。
好处是**不用自研通信协议**：MCP 自带 initialize / tools/list / tools/call，
工具元数据、参数 schema、错误码都是协议规定好的，跨语言跨机器通用。

【怎么起一个远端 Server（本机模拟）】
    python main.py mcp-http-server --host 127.0.0.1 --port 8800
    # 等价于 uvicorn modules.mcp.mcp_http_server:app --host 0.0.0.0 --port 8800

【怎么用（两种配置二选一）】
    MCP_REMOTE_URL=http://10.0.0.7:8800/mcp                     # 只接一个远程 server
    MCP_SERVERS_JSON='[{"name":"remote","transport":"http",      # 多个 server 混管
                        "endpoint":"http://10.0.0.7:8800/mcp"}]'

【完整数据流（本客户端 ↔ 远程 Server）】
    1. connect()            POST /mcp {"method":"initialize"} → 拿 serverInfo，握手成功
    2. connect()            POST /mcp {"method":"notifications/initialized"}（通知，不等结果）
    3. discover_tools()     POST /mcp {"method":"tools/list"} → 工具名 + 描述 + inputSchema
    4. 网关把清单建成目录 → Tool-RAG 索引 → 每轮语义召回 → ensure_registered 按需注册
    5. call_tool()          POST /mcp {"method":"tools/call","params":{"name","arguments"}}
    6. 拿到 result → MCPToolAdapter 提取 content[].text → ToolResult → 回喂 LLM
    7. 任一步失败（连不上 / 401 / 超时 / isError）→ 抛 MCPException，
       由网关决定"该 server 不可用"还是"工具参数错了"（只前者计熔断）

【四个设计目的】
    1. 远程执行：工具在远端跑，本机只发一个 HTTP 请求 —— 天然隔离，不用再套沙箱
    2. 零协议成本：initialize / tools/list / tools/call 与错误码都是 MCP 规定好的，
       对面换成 Node/Go 写的 Server 也不用改这里
    3. 接口对齐：方法名与 stdio 版 MCPClient 完全一致（connect/discover_tools/
       call_tool/disconnect），所以 MCPToolAdapter 与 MCPGateway 一行都不用改
    4. 可降级：连不上时由网关按"该 server 不可用"处理，本地工具不受影响

【输入输出对照】
    client = MCPHttpClient(endpoint="http://10.0.0.7:8800/mcp", token="t")
    client.connect()         → True（握手拿到 serverInfo）
    client.discover_tools()  → [{"name":"web_search","inputSchema":{...}}, ...]
    client.call_tool("get_current_time", {"timezone": "Asia/Shanghai"})
                             → {"content":[{"type":"text","text":"2026-..."}],"isError":False}
    client.connect() 远端没起 → 抛 MCPException（网关接住后降级，不是崩）
"""
from __future__ import annotations                                # 启用注解前向引用，类型标注里的类名可延迟解析

import urllib.error                                               # 标准库：HTTP 错误类型（HTTPError/URLError）
import urllib.request                                             # 标准库：HTTP 客户端（urllib 足够，不挂 requests/httpx）
from typing import Any, Dict, List, Optional                      # 类型标注工具

from config import get_settings                                   # 全局配置入口，取工具超时
from modules.common.exceptions import MCPException, handle_exception  # MCP 异常类型与统一异常处理装饰器
from modules.common.logger import get_logger                      # 项目统一日志入口
from modules.mcp.protocol import create_request, parse_message    # MCP 协议工具：构造请求帧 / 解析响应帧

_logger = get_logger("mcp_http_client")                           # 模块级日志器，名字 "mcp_http_client" 用于日志过滤


# 【什么】MCP 客户端（HTTP 传输版）
# 【作用】把"发现远程工具、调用远程工具"封装成本地方法，Agent 不必关心协议与网络细节
# 【为什么】与 stdio 版同接口：传输可以换，上层适配器与网关不用改
class MCPHttpClient:                                              # MCP 客户端类（HTTP 传输）
    """MCP 客户端（HTTP 传输）：与 MCPClient 同接口，只是把 stdio 换成 HTTP。

    【一问一答：一次 HTTP = 一个 JSON-RPC 帧】
        POST /mcp
          请求 {"jsonrpc":"2.0","method":"tools/call","params":{"name":..,"arguments":{..}},"id":3}
          响应 {"jsonrpc":"2.0","result":{"content":[{"type":"text","text":".."}]},"id":3}
        通知类（notifications/*）按 JSON-RPC 规范不该有响应体，服务端回 202，这里按成功处理。

    【为什么不用 requests/httpx】
        与沙箱的远程执行器同一原则：基础组件不额外挂网络依赖，标准库 urllib 足够。
    """

    # 【什么】客户端构造：只记配置，不建连接
    # 【作用】真正建连推迟到 connect()，失败可以重试而不必重建对象
    # 【为什么】与 stdio 版保持同一份生命周期约定，网关对两种传输一视同仁
    def __init__(self, endpoint: str, name: str = "mcp-http", token: str = "",
                 timeout_seconds: float = 0.0):                   # 入参：远端地址 + 逻辑名 + 令牌 + 超时
        self._endpoint = str(endpoint or "").strip()      # 远端 /mcp 地址（None 归一化为空串并去空白）
        self._name = name                                 # 逻辑名，只用于日志与熔断隔离
        self._token = str(token or "").strip()            # Bearer 令牌；空则不带头
        self._timeout = float(timeout_seconds or 0.0)     # 0 = 用全局 TOOL_TIMEOUT_SECONDS
        self._req_id = 0                                  # 请求号计数器
        self._connected = False                           # 握手是否成功
        self._server_info: Dict[str, Any] = {}            # 握手拿到的 serverInfo，仅用于日志

    # 【什么】远端地址
    # 【作用】日志与测试用；也方便网关判断"这个工具是不是远程的"
    # 【为什么】用 property 暴露只读视图，内部字段不被外部改写
    @property                                                       # 声明为只读属性
    def endpoint(self) -> str:                                      # 返回远端地址
        return self._endpoint                                       # 返回内部字段

    # 【什么】握手时拿到对端自报的服务信息
    # 【作用】日志与排查用：能看到对端叫什么、什么版本
    # 【为什么】返回拷贝，避免外部改动污染内部状态
    @property                                                       # 声明为只读属性
    def server_info(self) -> Dict[str, Any]:                        # 返回 serverInfo 字典
        return dict(self._server_info)                              # 拷贝后返回

    # 【什么】取下一个请求号
    # 【作用】每次发请求都递增，响应靠这个号与请求配对
    # 【为什么】JSON-RPC 的 id 配对机制要求单调唯一
    def _next_id(self) -> int:                                      # 返回自增后的请求号
        self._req_id += 1                                           # 计数器加一
        return self._req_id                                         # 返回新请求号

    # 【什么】超时取值优先级
    # 【作用】单次调用传的 > 客户端配的 > 全局 TOOL_TIMEOUT_SECONDS；三者都是 0 时兜 30 秒
    # 【为什么】就近原则：调用方最清楚这次调用能等多久，逐级回退保证总有可用值
    def _timeout_of(self, timeout: Optional[float] = None) -> float:  # 入参可选超时，返回最终生效值
        return float(timeout or self._timeout or get_settings().tool_timeout_seconds)  # 按优先级取第一个非 0 值

    # 【什么】连接入口：握手 initialize + 发 initialized 通知
    # 【作用】握手成功置 _connected=True；任何失败都包成 MCPException 往上抛（reraise=True）
    # 【为什么】HTTP 虽无状态，但协议仍要求握手：确认对端是 MCP Server 且版本兼容
    @handle_exception(default=False, log_msg="MCP(HTTP)连接异常", reraise=True)  # 装饰器：记日志并转 MCPException 重抛
    def connect(self, timeout: Optional[float] = None) -> bool:     # 可选超时；返回 True 表示握手成功
        """与远程 MCP Server 握手（initialize）。"""
        if not self._endpoint:                                      # 没配远端地址
            raise MCPException("MCP Server 未配置 endpoint", f"name={self._name}")  # 直接抛错

        # 【为什么握手要带上 clientInfo】协议规定双方自报家门，远端可据此做版本兼容；
        #   不带多数 Server 也能通，但带上是标准做法，排查问题时日志里有据可查
        resp = self._send_request("initialize", {                   # 发 initialize 握手请求
            "protocolVersion": "2024-11-05",                        # 声明支持的协议版本
            "capabilities": {},                                     # 客户端能力集（无额外能力）
            "clientInfo": {"name": "ai-agent-mcp-client", "version": "1.0.0"},  # 客户端自我介绍
        }, timeout=timeout)
        if not resp or "result" not in resp:                        # 没有正常结果帧
            raise MCPException("MCP(HTTP)握手失败", str(resp)[:200])  # 抛握手失败（截断响应避免刷屏）

        self._connected = True                                      # 标记握手成功
        self._server_info = (resp.get("result", {}) or {}).get("serverInfo", {}) or {}  # 记下对端 serverInfo（逐层兜底）
        _logger.info(f"MCP Server 已连接(HTTP): {self._name} | "
                     f"server={self._server_info} | {self._endpoint}")  # 记连接成功日志
        self._send_notification("notifications/initialized", {})    # 按协议回发 initialized 通知
        return True                                                 # 连接成功

    # 【什么】发一条 JSON-RPC 请求并同步等响应
    # 【作用】协议收发核心：POST 一帧 → 解析响应 → 有 error 就转成 MCPException
    # 【为什么】HTTP 错误（401/500）与协议错误（error 帧）分两路处理：前者是传输故障，后者是业务/协议语义
    def _send_request(self, method: str, params: dict,
                      timeout: Optional[float] = None) -> dict:     # 入参：方法名+参数+可选超时
        timeout = self._timeout_of(timeout)                         # 归一化超时
        req_id = self._next_id()                                    # 取本请求的请求号

        # 【请求怎么组出来的】create_request 生成一行 JSON-RPC 文本 → encode 成字节 → POST 出去
        #   Content-Type 必须声明 utf-8：不声明时对方可能按别的编码解，中文参数会变成乱码
        body = create_request(method, params, req_id=req_id).encode("utf-8")  # 构造请求帧并编码成字节
        req = urllib.request.Request(                               # 组装 HTTP 请求对象
            self._endpoint, data=body, method="POST",               # POST 到远端 /mcp 端点
            headers={"Content-Type": "application/json; charset=utf-8"},  # 声明 JSON 与编码
        )
        if self._token:                                             # 配了 Bearer 令牌
            req.add_header("Authorization", f"Bearer {self._token}")  # 加鉴权头

        try:
            # 【为什么用 with】urlopen 返回的是响应对象，with 块结束必定关闭连接；
            #   不关的话并发调用会攒下一堆半开连接，最后表现为"偶发超时"
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # 发请求并同步读响应
                raw = resp.read().decode("utf-8", errors="replace")     # 读响应体并按 UTF-8 解码（坏字节替换而非报错）
        except urllib.error.HTTPError as e:                # 401 / 404 / 500 ...
            # 【为什么要把响应体带上】光看状态码只知道"401"，带上 detail 才知道
            #   是"令牌错"还是" Server 压根没这个端点"，排查省一轮
            detail = ""                                             # 响应体摘录收集器
            try:
                detail = (e.read() or b"").decode("utf-8", errors="replace")[:200]  # 读错误响应体（截断 200 字）
            except Exception:                                       # 响应体也读不出来
                pass                                                # detail 留空即可
            raise MCPException(f"MCP Server 返回 HTTP {e.code}", detail) from e  # 转成 MCP 异常（链式保留原始异常）
        except urllib.error.URLError as e:                 # 连不上 / DNS 失败 / 被拒
            raise MCPException("MCP Server 连接失败", f"{self._endpoint} | {e.reason}") from e  # 转成 MCP 异常并附地址与原因

        if not raw.strip():                                # 通知类：服务端按规范回空体（202）
            return {}                                               # 回空 dict 表示成功无内容
        try:
            resp_msg: Dict[str, Any] = parse_message(raw)           # 解析响应帧为 dict
        except ValueError as e:                                     # 响应不是合法 JSON-RPC
            raise MCPException("MCP响应解析失败", f"{str(e)[:200]}") from e  # 转成 MCP 异常
        if "error" in resp_msg:                                     # 是 error 帧
            err = resp_msg.get("error", {}) or {}                   # 取错误对象
            # 【为什么转成异常而不是返回 None】调用方（网关）靠异常类型区分
            #   "server 不可用"与"工具参数错"，返回 None 会把两种情况混成一团
            raise MCPException(f"MCP调用错误: {err.get('message', '')}", f"code={err.get('code')}")  # 转成 MCP 异常
        return resp_msg                                             # 返回正常响应帧

    # 【什么】发一条通知（POST 出去就走，不关心响应体）
    # 【作用】用于 notifications/initialized；失败只记 debug ——
    #         通知丢了不影响后续调用，不该让一次握手的小抖动变成连接失败
    # 【为什么】通知语义就是 fire-and-forget，可靠性要求低于请求
    def _send_notification(self, method: str, params: dict):        # 入参：通知方法名+参数
        try:
            self._send_request(method, params)                      # 复用请求路径发出去
        except Exception as e:                                      # 发送失败
            _logger.debug(f"MCP通知发送失败（不影响调用）: {method} | {e}")  # 只记 debug，不上抛

    # 【什么】调 MCP 的 tools/list 拉远程工具清单
    # 【作用】网关建目录、客户端注册适配器都靠它；异常时返回空 list 而不是中断
    # 【为什么】发现失败是局部故障：装饰器兜成空列表，网关据此跳过该 server
    @handle_exception(default=[], log_msg="MCP(HTTP)工具发现异常")  # 装饰器：异常记日志并返回空列表
    def discover_tools(self) -> List[dict]:                         # 返回工具元数据列表
        """调用远程 tools/list 发现工具。"""
        resp = self._send_request("tools/list", {})                 # 发 tools/list 请求
        # 【为什么两级兜底】result 可能整个缺失（协议错误），tools 也可能为 null
        tools = (resp.get("result", {}) or {}).get("tools", []) or []  # 逐层取 result.tools，缺层兜空列表
        _logger.info(f"MCP(HTTP)工具发现: {self._name} | 工具数={len(tools)}")  # 记发现结果摘要
        for t in tools:                                             # 逐个工具打一行明细日志
            _logger.info(f"  - {t.get('name')}: {t.get('description', '')[:60]}")  # 名字 + 截断到 60 字的描述
        return tools                                                # 返回工具列表

    # 【什么】调 MCP 的 tools/call 执行一个远程工具
    # 【作用】Server 返回 isError 时转成 MCPException；异常被装饰器兜成 None
    # 【为什么】isError 是业务失败信号，转成异常才能进网关的"不记熔断"分支
    @handle_exception(default=None, log_msg="MCP(HTTP)工具调用异常")  # 装饰器：异常记日志并返回 None
    def call_tool(self, tool_name: str, arguments: dict,
                  timeout: Optional[float] = None) -> Optional[dict]:  # 入参：工具名+参数+可选超时
        """调用远程工具（tools/call）。"""
        resp = self._send_request("tools/call", {                   # 发 tools/call 请求
            "name": tool_name,                                      # 要调用的工具名
            "arguments": arguments,                                 # 工具入参
        }, timeout=timeout)
        result = resp.get("result", {}) or {}                       # 取 result，缺层兜空 dict
        if result.get("isError"):                                   # Server 报告业务执行失败
            # 【isError 是什么意思】工具被受理了但执行失败（如参数不合法），
            #   属于业务失败，不是服务故障 —— 网关据此决定不记熔断
            error_text = "".join(                                   # 拼接所有文本内容块作为错误描述
                item.get("text", "") for item in result.get("content", [])
                if item.get("type") == "text"
            )
            raise MCPException(f"MCP工具执行失败: {tool_name}", error_text)  # 转成异常上抛
        _logger.info(f"MCP(HTTP)工具调用成功: {tool_name}")         # 记调用成功日志
        return result                                               # 返回工具结果

    # 【什么】断开连接
    # 【作用】HTTP 是无状态的，这里没有子进程要杀（stdio 版要 terminate），
    #         只把握手标记置回 False，让 is_connected 归位
    # 【为什么】与 stdio 版保持同名方法，网关的断开逻辑两种传输通用
    def disconnect(self):                                           # 无参数无返回
        if self._connected:                                         # 此前握手成功过
            _logger.info(f"MCP Server 已断开(HTTP): {self._name}")  # 记断开日志
        self._connected = False                                     # 复位握手标记

    # 【什么】判定当前是否已握手
    # 【作用】网关的连接复用判断走它（与 stdio 版的 is_connected 同名同语义）
    # 【为什么】HTTP 无连接可查，只能信握手标记；远端真挂了会在下次调用时暴露
    @property                                                       # 声明为只读属性
    def is_connected(self) -> bool:                                 # 返回布尔：是否已握手
        return self._connected                                      # 返回握手标记
