"""
MCP Client 模块
===============
MCP 标准化工具协议客户端：连接独立 MCP Server，发现工具、调用远程工具。
基于 JSON-RPC 2.0 + stdio 传输，实现 Agent 与工具服务解耦。
兼容【本地原生工具】+【MCP远程工具】双模式调度。
"""
import json
import subprocess
import threading
from typing import Any, Optional, List, Dict
from queue import Queue, Empty
from modules.common.logger import get_logger
from modules.common.exceptions import MCPException, handle_exception
from modules.tools.base_tool import BaseTool, ToolResult
from modules.mcp.protocol import (
    create_request, parse_message,
    create_error, INTERNAL_ERROR, METHOD_NOT_FOUND,
)
from config import get_settings

_logger = get_logger("mcp_client")


# 【什么】MCP 客户端：通过 stdio 子进程跟独立 MCP Server 对话
# 【作用】把"发现工具、调用远程工具"封装成本地方法，Agent 不必关心协议细节
class MCPClient:
    """
    MCP 客户端：通过 stdio 子进程与独立 MCP Server 通信。
    生命周期：connect → discover_tools → call_tool → disconnect
    """

    # 【什么】客户端构造：只记配置，不拉起子进程
    # 【作用】真正建连推迟到 connect()，失败可以重试而不必重建对象
    def __init__(self, server_command: list, name: str = "mcp-server"):
        self._server_command = server_command                  # 启动命令，如 [python, "-m", "modules.mcp.mcp_server"]
        self._name = name                                      # 逻辑名，只用于日志与错误定位，默认 "mcp-server"
        self._process: Optional[subprocess.Popen] = None       # MCP Server 子进程；未连接时是 None
        self._stderr_queue: Queue = Queue()                    # 子进程 stderr 的暂存队列，由后台线程写入
        self._req_id = 0                                       # 请求号计数器，发出的第一个 id 是 1
        self._connected = False                                # 握手是否成功；disconnect 后置回 False

    # 【什么】取下一个请求号
    # 【作用】每次发请求都递增，响应靠这个号与请求配对
    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    # 【什么】连接入口：拉起子进程并完成 initialize 握手
    # 【作用】握手成功置 _connected=True；任何失败都包成 MCPException 往上抛（reraise=True）
    @handle_exception(default=None, log_msg="MCP连接异常", reraise=True)
    def connect(self, timeout: Optional[float] = None) -> bool:
        """启动 MCP Server 子进程并完成握手初始化。"""
        # 未显式传 timeout 时用全局工具超时（TOOL_TIMEOUT_SECONDS 秒）
        timeout = timeout or get_settings().tool_timeout_seconds
        try:
            # ------------------------------------------------------------------
            # 【这行在干什么】拉起 MCP Server 子进程，并把它三条标准流都接成管道。
            #   MCP 的 stdio 传输就跑在这三条管道上，一条都不能少：
            #     stdin  ← 我们写一行 JSON-RPC 请求（_send_request：write + flush）
            #     stdout ← 我们读一行 JSON-RPC 响应（_send_request：readline）
            #     stderr ← 子进程自己打的日志，不参与协议，交给下面那个线程抽走
            #
            # 【处理前 → 处理后】
            #   处理前 self._server_command = [python.exe, "-m", "modules.mcp.mcp_server"]
            #   处理后 self._process 是一个活着的子进程，三条流各有管道可用：
            #     self._process.stdin   → 可写
            #     self._process.stdout  → 可读
            #     self._process.stderr  → 可读（必须有人持续读，见下一块注释）
            #
            # 【三个参数各自的硬性理由 / 不这么写会怎样】
            #   text=True + encoding="utf-8"
            #       少了 text=True → readline 拿到的是 bytes，parse_message 解 JSON 直接抛；
            #       少了 encoding → Windows 上按系统默认编码（GBK）解码，中文工具名与
            #       中文结果变乱码，且是静默乱码，不报错。
            #   bufsize=1
            #       行缓冲。配合 _send_request 里显式 flush()，一行请求立刻送达对端；
            #       不设则是大块缓冲，对端迟迟收不到请求 —— 表现成"卡住直到超时"。
            #   stderr=subprocess.PIPE
            #       接了管道就必须有人一直读。管道缓冲区（Linux 默认 64 KiB）写满后，
            #       子进程再 print 一行日志就会阻塞，请求也就再也等不到响应，
            #       而主进程这边只看到"超时"，看不出是日志把管道堵死了。
            #
            # 【为什么用子进程而不是直接 import】命令换成 ["node", "server.js"] 就能接
            #   Node 写的工具，Python 包版本冲突也一并隔离；Server 崩了只崩它自己。
            # ------------------------------------------------------------------
            self._process = subprocess.Popen(
                self._server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding="utf-8",
            )
            # ------------------------------------------------------------------
            # 【这行在干什么】起一个后台线程专职抽 stderr，主线程不被日志 IO 拖住。
            #   _read_stderr 里是 `for line in self._process.stderr` —— 阻塞式读，
            #   没有下一行就一直等着，所以只能放线程里跑，不能放进主线程。
            #
            # 【处理前 → 处理后】
            #   处理前 子进程往 stderr 写一行：
            #     "[enterprise-mcp] 工具 get_current_time 被调用\n"
            #   处理后 这一行进 _stderr_queue，并打一条 debug 日志：
            #     [enterprise-mcp] stderr: 工具 get_current_time 被调用
            #
            # 【daemon=True 的理由 / 不这么写会怎样】
            #   不加 → 主线程退出时解释器要等这个非守护线程结束，而它正阻塞在读管道、
            #   永远等不到 EOF，主进程就退不掉，只能靠 kill。
            #
            # 【边界情况】这个线程没有停止开关，靠子进程退出时 stderr 关闭、
            #   for 循环自然结束。所以 disconnect() 里 terminate + wait(5) 之后，
            #   队列里不会再有新行进来，残留内容可直接丢弃。
            # ------------------------------------------------------------------
            threading.Thread(
                target=self._read_stderr, daemon=True
            ).start()

            resp = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ai-agent-mcp-client", "version": "1.0.0"},
            }, timeout=timeout)

            if resp and "result" in resp:
                self._connected = True
                _logger.info(
                    f"MCP Server 连接成功: {self._name} | "
                    f"server={resp['result'].get('serverInfo', {})})"
                )
                self._send_notification("notifications/initialized", {})
                return True
            else:
                raise MCPException("MCP握手失败", str(resp))
        except FileNotFoundError:
            raise MCPException(
                "MCP Server 启动失败",
                f"命令不存在: {self._server_command}",
            )
        except Exception as e:
            raise MCPException("MCP连接异常", str(e))

    # 【什么】后台线程体：持续读子进程 stderr 并塞进队列
    # 【作用】把子进程的日志抽走，避免管道缓冲区写满后子进程被卡死
    def _read_stderr(self):
        """后台读取 Server stderr 输出。"""
        if not self._process or not self._process.stderr:
            return
        for line in self._process.stderr:
            self._stderr_queue.put(line.rstrip())
            _logger.debug(f"[{self._name}] stderr: {line.rstrip()}")

    # 【什么】发一条 JSON-RPC 请求并同步等响应
    # 【作用】协议收发的核心：写一帧请求 → 读一行响应 → 有 error 就转成 MCPException
    def _send_request(self, method: str, params: dict, timeout: Optional[float] = None) -> dict:
        """发送 JSON-RPC 请求并等待响应。"""
        timeout = timeout or get_settings().tool_timeout_seconds   # 未传则用全局工具超时
        if not self._process or not self._process.stdin:
            raise MCPException("MCP Server 未连接", "")
        req_id = self._next_id()
        msg = create_request(method, params, req_id=req_id)
        self._process.stdin.write(msg + "\n")
        self._process.stdin.flush()

        try:
            line = self._process.stdout.readline()
            if not line:
                raise MCPException("MCP Server 无响应", "")
            resp: dict[Any, Any] = parse_message(raw=line)
            if "error" in resp:
                err = resp["error"]
                raise MCPException(
                    f"MCP调用错误: {err.get('message', '')}",
                    f"code={err.get('code')}",
                )
            return resp
        except Empty:
            raise MCPException("MCP响应超时", f"method={method}, timeout={timeout}s")

    # 【什么】发一条通知（写了就走，不等响应）
    # 【作用】用于 notifications/initialized 这类单向消息；未连接时静默返回
    def _send_notification(self, method: str, params: dict):
        """发送通知（无需响应）。"""
        if not self._process or not self._process.stdin:
            return
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        self._process.stdin.write(msg + "\n")
        self._process.stdin.flush()

    # 【什么】调 MCP 的 tools/list 拉远程工具清单
    # 【作用】网关建目录、客户端注册适配器都靠它；异常时返回空 list 而不是中断
    @handle_exception(default=[], log_msg="MCP工具发现异常")
    def discover_tools(self) -> list:
        """调用 MCP tools/list 发现远程工具。"""
        resp = self._send_request("tools/list", {})
        tools = resp.get("result", {}).get("tools", [])
        _logger.info(f"MCP工具发现: {self._name} | 工具数={len(tools)}")
        for t in tools:
            _logger.info(f"  - {t.get('name')}: {t.get('description', '')[:60]}")
        return tools

    # 【什么】调 MCP 的 tools/call 执行一个远程工具
    # 【作用】Server 返回 isError 时转成 MCPException；异常被装饰器兜成 None
    @handle_exception(default=None, log_msg="MCP工具调用异常")
    def call_tool(self, tool_name: str, arguments: dict, timeout: Optional[float] = None) -> dict:
        """调用 MCP 远程工具（tools/call）。"""
        timeout = timeout or get_settings().tool_timeout_seconds   # 未传则用全局工具超时
        resp = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        }, timeout=timeout)
        result = resp.get("result", {})
        if result.get("isError"):
            error_text = ""
            for item in result.get("content", []):
                if item.get("type") == "text":
                    error_text += item.get("text", "")
            raise MCPException(f"MCP工具执行失败: {tool_name}", error_text)
        _logger.info(f"MCP工具调用成功: {tool_name}")
        return result

    # 【什么】断开连接：关 stdin → terminate → 等 5 秒
    # 【作用】进程退出前调用，避免留下孤儿子进程；等待超时就 kill
    def disconnect(self):
        """关闭 MCP Server 连接。"""
        if self._process:
            try:
                self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            finally:
                self._process = None
                self._connected = False
                _logger.info(f"MCP Server 已断开: {self._name}")

    # 【什么】判定当前是否真的连着
    # 【作用】三个条件同时成立才算：握手过、有子进程、子进程还没退出
    @property
    def is_connected(self) -> bool:
        return self._connected and self._process is not None and self._process.poll() is None


# 【什么】MCP 远程工具适配器：把远程工具包装成 BaseTool
# 【作用】让 ToolRegistry 能用同一套接口管理本地工具与 MCP 远程工具
class MCPToolAdapter(BaseTool):
    """
    MCP 远程工具适配器：将 MCP 远程工具适配为 BaseTool 接口。
    使 ToolRegistry 可统一管理本地工具与 MCP 工具。

    【为什么继承 BaseTool】ToolRegistry.register 只收 BaseTool，
    此前靠 duck typing 碰巧兼容（有 name/run 等同名成员），
    现在正式继承，类型检查器可验证接口完整性（抽象方法 execute 必须实现）。
    注意：不走 BaseTool.__init__（它要求 parameters dict），
    这里直接从 MCP 的 inputSchema 组装同名属性。
    """

    # 【能力声明】远程工具默认 medium：它跑在外部 server 里，可能有副作用，
    #   比本地只读工具高一级（仍可直接执行，不自动触发人工审批）。
    #   真正的高危远程工具用 TOOL_RISK_OVERRIDES 单独提级为 high/critical。
    capability = {
        "risk": "medium",
        "side_effect": True,
        "sandbox": "none",          # 远程工具本身已在 MCP server 子进程中执行
        "description": "MCP 远程工具，执行发生在外部 server",
    }
    # 【什么】适配器构造：从 MCP 工具元数据组装成本地工具属性
    # 【作用】不走 BaseTool.__init__，直接用 inputSchema 拼同名属性；缺 inputSchema 时兜空 schema
    # 【参数类型为什么是 Any】适配器只要求客户端有 call_tool 方法：
    #   stdio 的 MCPClient 与远程的 MCPHttpClient 都能传，换传输不用改适配器
    def __init__(self, mcp_client: Any, tool_info: dict):
        self._client = mcp_client                 # 复用的 MCP 客户端；真正调用走它
        self._tool_info = tool_info               # Server 返回的原始工具元数据
        self.name = tool_info["name"]  
                   # 工具名，必填；缺失会直接 KeyError
        self.description = tool_info.get("description", "")   # 描述；缺失兜空串
        self.parameters_schema = tool_info.get("inputSchema", {   # 参数 JSON Schema
            "type": "object", "properties": {}, "required": []    # 兜底：空对象、无必填
        })

    # 【什么】标出这是远程 MCP 工具
    # 【作用】权限过滤时本地工具认 tool:{name}，MCP 工具额外认 tool:mcp:{name}
    @property
    def tool_type(self) -> str:
        return "mcp"

    # 【什么】把工具转成 OpenAI function schema
    # 【作用】模型看到的工具表就是它；description 空缺会让模型不容易选中该工具
    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    # 【什么】按 schema 的 required 校验必填参数
    # 【作用】缺参数就地抛 MCPException，把错误拦在发请求之前
    def validate_params(self, params: dict) -> dict:
        required = self.parameters_schema.get("required", [])   # 必填参数名列表，无 schema 时是 []
        missing = [r for r in required if r not in (params or {})]
        if missing:
            raise MCPException(f"参数缺失: {missing}", f"tool={self.name}")
        return params or {}

    # 【什么】BaseTool 抽象方法的实现：真正调远程工具
    # 【作用】只负责"调远程 + 提取文本"，计时/异常兜底/封装交给父类 run
    def execute(self, **kwargs) -> ToolResult:
        """真正调用 MCP 远程工具（BaseTool 抽象方法的实现）。

        BaseTool.run 的统一链路是：validate_params → execute → 封装 ToolResult，
        这里只负责"调远程 + 提取文本"，参数校验/计时/异常兜底交给父类 run。
        """
        result = self._client.call_tool(self.name, kwargs)
        text_parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return ToolResult(
            success=True,
            result="\n".join(text_parts) if text_parts else result,
        )
