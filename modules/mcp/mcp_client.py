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
    def connect(self, timeout: float = 15.0) -> bool:
        """启动 MCP Server 子进程并完成握手初始化。"""
        try:
            # ── 干什么：在本机新起一个子进程跑 MCP Server ──
            # 注意它不是网络服务（没有 IP/端口），就是操作系统层面的"父子进程"，
            # 双方靠 3 根管道说 JSON-RPC 2.0，一次一行 JSON（NDJSON 行协议）。
            #
            # 通俗比喻：你（主进程）和一位关在小黑屋里的专家（子进程）合作——
            #   塞进门缝的「问题纸条」= stdin
            #   专家递出来的「回答纸条」= stdout
            #   专家自言自语的「碎碎念（日志）」= stderr
            #
            # 命令来自 main.py:107 → [sys.executable, "-m", "modules.mcp.mcp_server"]
            # 等价于在终端敲：python -m modules.mcp.mcp_server
            # 子进程那头跑的是 mcp_server.py:204 的 `for line in sys.stdin:` 事件循环
            self._process = subprocess.Popen(
                self._server_command,
                # 父 → 子：写请求（initialize / tools/list / tools/call）
                # 专业点说：管道是单向的，所以"说"和"听"必须是两根不同的管子
                stdin=subprocess.PIPE,
                # 子 → 父：读响应（_send_request 里 stdout.readline() 一次读一行）
                # ⚠ 这是"协议专线"：子进程任何 print / 日志都不许打到这里，
                #   否则 parse_message 拿到非 JSON 的行，整个协议就崩了
                stdout=subprocess.PIPE,
                # 子 → 父：子进程的 loguru 日志，不参与协议，只给人看
                # ⚠ 必须有人持续读（下面 _read_stderr 线程就在干这事）：
                #   管道是内核里固定大小的缓冲区（Linux 默认 64KB），写满后
                #   子进程再 write 就会卡住等 → 双方互相等 = 死锁
                stderr=subprocess.PIPE,
                # 文本模式：读写直接拿 str，省掉 encode / decode
                text=True,
                # 行缓冲（text 模式下 bufsize=1 即 line-buffered）：
                # 写一行立刻发出去，否则会攒在缓冲区里，对面半天收不到
                # （所以 _send_request 写完还跟了一行 flush）
                bufsize=1,
                # 显式 UTF-8：Windows 上不写会走系统默认 GBK，中文工具参数必乱码
                encoding="utf-8",
                # ⚠ 未传 cwd：子进程继承父进程工作目录
                #   → 必须在项目根目录运行，否则 `-m modules.mcp.mcp_server` 找不到包，
                #     启动失败被上层降级（main.py:114），表现为"少了几个工具"而不是报错
            )
            # 后台读取 stderr（日志/调试）
            threading.Thread(
                target=self._read_stderr, daemon=True
            ).start()

            # MCP initialize 握手
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
                # 发送 initialized 通知（MCP 协议要求）
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

    def _send_request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        """发送 JSON-RPC 请求并等待响应。"""
        if not self._process or not self._process.stdin:
            raise MCPException("MCP Server 未连接", "")
        req_id = self._next_id()
        msg = create_request(method, params, req_id=req_id)
        self._process.stdin.write(msg + "\n")
        self._process.stdin.flush()

        # 读取响应行
        try:
            line = self._process.stdout.readline()
            if not line:
                raise MCPException("MCP Server 无响应", "")
            resp = parse_message(line)
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
    def call_tool(self, tool_name: str, arguments: dict, timeout: float = 60.0) -> dict:
        """调用 MCP 远程工具（tools/call）。"""
        resp = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        }, timeout=timeout)
        result = resp.get("result", {})
        # MCP 工具结果格式: {content: [{type, text}], isError}
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
        # MCP 工具的参数 schema
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
        # 提取文本内容（MCP 返回 content 数组，取其中 type=text 的部分拼接）
        text_parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return ToolResult(
            success=True,
            result="\n".join(text_parts) if text_parts else result,
        )
        # 注：不再自带 duration_ms / 异常兜底 —— 统一由继承来的 BaseTool.run 负责
