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
from typing import Optional, List, Dict
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


class MCPClient:
    """
    MCP 客户端：通过 stdio 子进程与独立 MCP Server 通信。
    生命周期：connect → discover_tools → call_tool → disconnect
    """

    def __init__(self, server_command: list, name: str = "mcp-server"):
        self._server_command = server_command
        self._name = name
        self._process: Optional[subprocess.Popen] = None 
        self._stderr_queue: Queue = Queue()
        self._req_id = 0
        self._connected = False

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    @handle_exception(default=None, log_msg="MCP连接异常", reraise=True)
    def connect(self, timeout: Optional[float] = None) -> bool:
        """启动 MCP Server 子进程并完成握手初始化。"""
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

    def _read_stderr(self):
        """后台读取 Server stderr 输出。"""
        if not self._process or not self._process.stderr:
            return
        for line in self._process.stderr:
            self._stderr_queue.put(line.rstrip())
            _logger.debug(f"[{self._name}] stderr: {line.rstrip()}")

    def _send_request(self, method: str, params: dict, timeout: Optional[float] = None) -> dict:
        """发送 JSON-RPC 请求并等待响应。"""
        timeout = timeout or get_settings().tool_timeout_seconds
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
            resp = parse_message(raw=line)
            if "error" in resp:
                err = resp["error"]
                raise MCPException(
                    f"MCP调用错误: {err.get('message', '')}",
                    f"code={err.get('code')}",
                )
            return resp
        except Empty:
            raise MCPException("MCP响应超时", f"method={method}, timeout={timeout}s")

    def _send_notification(self, method: str, params: dict):
        """发送通知（无需响应）。"""
        if not self._process or not self._process.stdin:
            return
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        self._process.stdin.write(msg + "\n")
        self._process.stdin.flush()

    @handle_exception(default=[], log_msg="MCP工具发现异常")
    def discover_tools(self) -> list:
        """调用 MCP tools/list 发现远程工具。"""
        resp = self._send_request("tools/list", {})
        tools = resp.get("result", {}).get("tools", [])
        _logger.info(f"MCP工具发现: {self._name} | 工具数={len(tools)}")
        for t in tools:
            _logger.info(f"  - {t.get('name')}: {t.get('description', '')[:60]}")
        return tools

    @handle_exception(default=None, log_msg="MCP工具调用异常")
    def call_tool(self, tool_name: str, arguments: dict, timeout: Optional[float] = None) -> dict:
        """调用 MCP 远程工具（tools/call）。"""
        timeout = timeout or get_settings().tool_timeout_seconds
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

    @property
    def is_connected(self) -> bool:
        return self._connected and self._process is not None and self._process.poll() is None


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
    def __init__(self, mcp_client: MCPClient, tool_info: dict):
        self._client = mcp_client
        self._tool_info = tool_info
        self.name = tool_info["name"]
        self.description = tool_info.get("description", "")
        self.parameters_schema = tool_info.get("inputSchema", {
            "type": "object", "properties": {}, "required": []
        })

    @property
    def tool_type(self) -> str:
        return "mcp"

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def validate_params(self, params: dict) -> dict:
        required = self.parameters_schema.get("required", [])
        missing = [r for r in required if r not in (params or {})]
        if missing:
            raise MCPException(f"参数缺失: {missing}", f"tool={self.name}")
        return params or {}

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
