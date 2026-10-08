"""
MCP Client 模块
===============
MCP 标准化工具协议客户端：连接独立 MCP Server，发现工具、调用远程工具。
基于 JSON-RPC 2.0 + stdio 传输，实现 Agent 与工具服务解耦。
兼容【本地原生工具】+【MCP远程工具】双模式调度。
"""
import json                                                          # 标准库：JSON 编解码，用于构造通知消息
import subprocess                                                    # 标准库：拉起 MCP Server 子进程（stdio 传输载体）
import threading                                                     # 标准库：后台线程专职抽子进程 stderr
from typing import Any, Optional, List, Dict                         # 类型标注工具
from queue import Queue, Empty                                       # 线程安全队列与空队列异常，暂存 stderr 行
from modules.common.logger import get_logger                         # 项目统一日志入口
from modules.common.exceptions import MCPException, handle_exception  # MCP 异常类型与统一异常处理装饰器
from modules.tools.base_tool import BaseTool, ToolResult             # 工具基类与统一返回结构（适配器要继承）
from modules.mcp.protocol import (                                   # MCP 协议工具函数与错误码
    create_request, parse_message,                                  # 构造请求帧 / 解析响应帧
    create_error, INTERNAL_ERROR, METHOD_NOT_FOUND,                 # 构造错误帧 / 内部错误码 / 方法不存在错误码
)
from config import get_settings                                     # 全局配置入口，取工具超时等参数

_logger = get_logger("mcp_client")                                  # 模块级日志器，名字 "mcp_client" 用于日志过滤


# 【什么】MCP 客户端：通过 stdio 子进程跟独立 MCP Server 对话
# 【作用】把"发现工具、调用远程工具"封装成本地方法，Agent 不必关心协议细节
# 【为什么】协议收发、子进程生命周期都收敛在这一个类里，上层只面对普通方法调用
class MCPClient:                                                    # MCP 客户端类（stdio 传输）
    """
    MCP 客户端：通过 stdio 子进程与独立 MCP Server 通信。
    生命周期：connect → discover_tools → call_tool → disconnect
    """

    # 【什么】客户端构造：只记配置，不拉起子进程
    # 【作用】真正建连推迟到 connect()，失败可以重试而不必重建对象
    # 【为什么】构造与连接分离：网关可以先把配置表建好，连接按需发生
    def __init__(self, server_command: list, name: str = "mcp-server"):  # 入参：启动命令列表 + 逻辑名
        self._server_command = server_command                  # 启动命令，如 [python, "-m", "modules.mcp.mcp_server"]
        self._name = name                                      # 逻辑名，只用于日志与错误定位，默认 "mcp-server"
        self._process: Optional[subprocess.Popen] = None       # MCP Server 子进程；未连接时是 None
        self._stderr_queue: Queue = Queue()                    # 子进程 stderr 的暂存队列，由后台线程写入
        self._req_id = 0                                       # 请求号计数器，发出的第一个 id 是 1
        self._connected = False                                # 握手是否成功；disconnect 后置回 False

    # 【什么】取下一个请求号
    # 【作用】每次发请求都递增，响应靠这个号与请求配对
    # 【为什么】JSON-RPC 靠 id 配对请求与响应，单调递增保证一一对应
    def _next_id(self) -> int:                                 # 返回自增后的请求号
        self._req_id += 1                                      # 计数器加一
        return self._req_id                                    # 返回新请求号

    # 【什么】连接入口：拉起子进程并完成 initialize 握手
    # 【作用】握手成功置 _connected=True；任何失败都包成 MCPException 往上抛（reraise=True）
    # 【为什么】握手是协议强制步骤：不 initialize 就发 tools/list，Server 会直接报错
    @handle_exception(default=None, log_msg="MCP连接异常", reraise=True)  # 装饰器：统一记日志并转成 MCPException 重抛
    def connect(self, timeout: Optional[float] = None) -> bool:     # 可选超时；返回 True 表示握手成功
        """启动 MCP Server 子进程并完成握手初始化。"""
        # 未显式传 timeout 时用全局工具超时（TOOL_TIMEOUT_SECONDS 秒）
        timeout = timeout or get_settings().tool_timeout_seconds   # 归一化超时：None/0 时取全局配置
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
            self._process = subprocess.Popen(                  # 拉起 MCP Server 子进程
                self._server_command,                          # 启动命令列表
                stdin=subprocess.PIPE,                         # 子进程标准输入接管道（我们写请求）
                stdout=subprocess.PIPE,                        # 子进程标准输出接管道（我们读响应）
                stderr=subprocess.PIPE,                        # 子进程标准错误接管道（后台线程抽走）
                text=True,                                     # 文本模式：readline 返回 str 而非 bytes
                bufsize=1,                                     # 行缓冲：一行请求立即送达对端
                encoding="utf-8",                              # 按 UTF-8 编解码，避免 Windows 中文乱码
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
            threading.Thread(                                  # 创建后台线程
                target=self._read_stderr, daemon=True          # 线程体是 _read_stderr；守护线程随主进程退出
            ).start()                                          # 立即启动

            resp = self._send_request("initialize", {          # 发 initialize 握手请求并同步等响应
                "protocolVersion": "2024-11-05",               # 声明客户端支持的 MCP 协议版本
                "capabilities": {},                            # 客户端能力集（本客户端无额外能力）
                "clientInfo": {"name": "ai-agent-mcp-client", "version": "1.0.0"},  # 客户端自我介绍
            }, timeout=timeout)

            if resp and "result" in resp:                      # 收到正常结果帧（非 error 帧）
                self._connected = True                         # 标记握手成功
                _logger.info(                                  # 记连接成功日志
                    f"MCP Server 连接成功: {self._name} | "
                    f"server={resp['result'].get('serverInfo', {})})"  # 附对端自报的 serverInfo
                )
                self._send_notification("notifications/initialized", {})  # 按协议回发 initialized 通知，握手闭环
                return True                                    # 连接成功
            else:                                              # 没有 result 字段
                raise MCPException("MCP握手失败", str(resp))   # 抛握手失败，附原始响应便于排查
        except FileNotFoundError:                              # 启动命令不存在（如 python 不在 PATH）
            raise MCPException(                                # 转成 MCP 异常
                "MCP Server 启动失败",
                f"命令不存在: {self._server_command}",         # 附原始命令便于定位
            )
        except Exception as e:                                 # 其余任何异常（管道失败、握手超时等）
            raise MCPException("MCP连接异常", str(e))          # 统一包成 MCPException 上抛

    # 【什么】后台线程体：持续读子进程 stderr 并塞进队列
    # 【作用】把子进程的日志抽走，避免管道缓冲区写满后子进程被卡死
    # 【为什么】stderr 不参与协议但必须有人读；阻塞读只能放线程里
    def _read_stderr(self):                                    # 后台线程入口，无参数无返回
        """后台读取 Server stderr 输出。"""
        if not self._process or not self._process.stderr:      # 子进程不存在或 stderr 没接管道
            return                                             # 无事可做，直接退出线程
        for line in self._process.stderr:                      # 阻塞式逐行读；子进程退出关闭管道后循环自然结束
            self._stderr_queue.put(line.rstrip())              # 去掉行尾换行后塞进暂存队列
            _logger.debug(f"[{self._name}] stderr: {line.rstrip()}")  # 同步打一条 debug 日志，便于排查 server 侧问题

    # 【什么】发一条 JSON-RPC 请求并同步等响应
    # 【作用】协议收发的核心：写一帧请求 → 读一行响应 → 有 error 就转成 MCPException
    # 【为什么】写读配对必须原子完成，否则并发请求会把响应读串（本客户端假设单线程调用）
    def _send_request(self, method: str, params: dict, timeout: Optional[float] = None) -> dict:  # 入参：方法名+参数+可选超时
        """发送 JSON-RPC 请求并等待响应。"""
        timeout = timeout or get_settings().tool_timeout_seconds   # 未传则用全局工具超时
        if not self._process or not self._process.stdin:       # 未连接（子进程或 stdin 管道不存在）
            raise MCPException("MCP Server 未连接", "")        # 直接抛错，不写半成品请求
        req_id = self._next_id()                               # 取本请求的请求号
        msg = create_request(method, params, req_id=req_id)    # 用协议模块构造 JSON-RPC 请求帧（JSON 字符串）
        self._process.stdin.write(msg + "\n")                  # 写入子进程 stdin，行尾换行即一帧
        self._process.stdin.flush()                            # 立即冲刷，配合行缓冲确保送达

        try:
            line = self._process.stdout.readline()             # 阻塞读一行响应（按行分帧）
            if not line:                                       # 读到空串：子进程已退出并关闭 stdout
                raise MCPException("MCP Server 无响应", "")    # 抛无响应错误
            resp: dict[Any, Any] = parse_message(raw=line)     # 用协议模块解析响应帧为 dict
            if "error" in resp:                                # 是 error 帧
                err = resp["error"]                            # 取错误对象
                raise MCPException(                            # 转成 MCP 异常上抛
                    f"MCP调用错误: {err.get('message', '')}",  # 错误消息
                    f"code={err.get('code')}",                 # 错误码
                )
            return resp                                        # 返回正常响应帧
        except Empty:                                          # 队列空异常（预留给异步读取路径）
            raise MCPException("MCP响应超时", f"method={method}, timeout={timeout}s")  # 转成超时错误

    # 【什么】发一条通知（写了就走，不等响应）
    # 【作用】用于 notifications/initialized 这类单向消息；未连接时静默返回
    # 【为什么】JSON-RPC 通知没有 id、对端不回包，所以不需要 readline
    def _send_notification(self, method: str, params: dict):   # 入参：通知方法名+参数
        """发送通知（无需响应）。"""
        if not self._process or not self._process.stdin:       # 未连接
            return                                             # 静默返回：通知不关键，丢了不影响主流程
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})  # 手工拼通知帧（无 id 字段）
        self._process.stdin.write(msg + "\n")                  # 写入子进程 stdin
        self._process.stdin.flush()                            # 立即冲刷送达

    # 【什么】调 MCP 的 tools/list 拉远程工具清单
    # 【作用】网关建目录、客户端注册适配器都靠它；异常时返回空 list 而不是中断
    # 【为什么】发现失败是局部故障：装饰器兜成空列表，网关据此跳过该 server
    @handle_exception(default=[], log_msg="MCP工具发现异常")   # 装饰器：异常记日志并返回空列表
    def discover_tools(self) -> list:                          # 返回工具元数据列表
        """调用 MCP tools/list 发现远程工具。"""
        resp = self._send_request("tools/list", {})            # 发 tools/list 请求
        tools = resp.get("result", {}).get("tools", [])        # 逐层取 result.tools，缺层兜空列表
        _logger.info(f"MCP工具发现: {self._name} | 工具数={len(tools)}")  # 记发现结果摘要
        for t in tools:                                        # 逐个工具打一行明细日志
            _logger.info(f"  - {t.get('name')}: {t.get('description', '')[:60]}")  # 名字 + 截断到 60 字的描述
        return tools                                           # 返回工具列表

    # 【什么】调 MCP 的 tools/call 执行一个远程工具
    # 【作用】Server 返回 isError 时转成 MCPException；异常被装饰器兜成 None
    # 【为什么】isError 是协议层的业务失败信号，必须转成异常才能进熔断统计
    @handle_exception(default=None, log_msg="MCP工具调用异常")  # 装饰器：异常记日志并返回 None
    def call_tool(self, tool_name: str, arguments: dict, timeout: Optional[float] = None) -> dict:  # 入参：工具名+参数+可选超时
        """调用 MCP 远程工具（tools/call）。"""
        timeout = timeout or get_settings().tool_timeout_seconds   # 未传则用全局工具超时
        resp = self._send_request("tools/call", {              # 发 tools/call 请求
            "name": tool_name,                                 # 要调用的工具名
            "arguments": arguments,                            # 工具入参
        }, timeout=timeout)
        result = resp.get("result", {})                        # 取 result，缺层兜空 dict
        if result.get("isError"):                              # Server 报告业务执行失败
            error_text = ""                                    # 错误文本收集器
            for item in result.get("content", []):             # 遍历内容块
                if item.get("type") == "text":                 # 只收文本块
                    error_text += item.get("text", "")         # 拼接错误文本
            raise MCPException(f"MCP工具执行失败: {tool_name}", error_text)  # 转成异常上抛（带工具名与错误文本）
        _logger.info(f"MCP工具调用成功: {tool_name}")          # 记调用成功日志
        return result                                          # 返回工具结果

    # 【什么】断开连接：关 stdin → terminate → 等 5 秒
    # 【作用】进程退出前调用，避免留下孤儿子进程；等待超时就 kill
    # 【为什么】先礼后兵：terminate 给子进程收尾机会，5 秒不走再 kill 强杀
    def disconnect(self):                                      # 无参数无返回；副作用是关闭子进程
        """关闭 MCP Server 连接。"""
        if self._process:                                      # 当前持有子进程
            try:
                self._process.stdin.close()                    # 先关 stdin：告诉对端不会再有请求
                self._process.terminate()                      # 发 SIGTERM，请子进程自行退出
                self._process.wait(timeout=5)                  # 最多等 5 秒让它走完收尾
            except Exception:                                  # 关流/终止/等待任何一步失败（含超时）
                self._process.kill()                           # 强杀子进程（SIGKILL，不给收尾机会）
            finally:                                           # 无论成败都做状态清理
                self._process = None                           # 释放子进程句柄
                self._connected = False                        # 标记已断开
                _logger.info(f"MCP Server 已断开: {self._name}")  # 记断开日志

    # 【什么】判定当前是否真的连着
    # 【作用】三个条件同时成立才算：握手过、有子进程、子进程还没退出
    # 【为什么】poll() is None 能抓出"握手成功但子进程后来崩了"的僵尸连接
    @property                                                  # 声明为只读属性
    def is_connected(self) -> bool:                            # 返回布尔：是否处于可用连接状态
        return self._connected and self._process is not None and self._process.poll() is None  # 握手标记 ∧ 进程存在 ∧ 进程未退出


# 【什么】MCP 远程工具适配器：把远程工具包装成 BaseTool
# 【作用】让 ToolRegistry 能用同一套接口管理本地工具与 MCP 远程工具
# 【为什么】注册表只认 BaseTool，适配器把协议差异挡在自己内部
class MCPToolAdapter(BaseTool):                                # 适配器类，继承工具基类
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
    capability = {                                             # 类级能力声明（权限/审批路由读取）
        "risk": "medium",                                      # 风险等级：中
        "side_effect": True,                                   # 有副作用（远程执行不可预知）
        "sandbox": "none",          # 远程工具本身已在 MCP server 子进程中执行
        "description": "MCP 远程工具，执行发生在外部 server",   # 能力说明文案
    }
    # 【什么】适配器构造：从 MCP 工具元数据组装成本地工具属性
    # 【作用】不走 BaseTool.__init__，直接用 inputSchema 拼同名属性；缺 inputSchema 时兜空 schema
    # 【为什么】MCP 的 inputSchema 与 BaseTool 的 parameters 同构，直接搬即可，无需转换层
    # 【参数类型为什么是 Any】适配器只要求客户端有 call_tool 方法：
    #   stdio 的 MCPClient 与远程的 MCPHttpClient 都能传，换传输不用改适配器
    def __init__(self, mcp_client: Any, tool_info: dict):      # 入参：MCP 客户端 + 工具元数据
        self._client = mcp_client                 # 复用的 MCP 客户端；真正调用走它
        self._tool_info = tool_info               # Server 返回的原始工具元数据
        self.name = tool_info["name"]             # 工具名，必填；缺失会直接 KeyError
        self.description = tool_info.get("description", "")   # 描述；缺失兜空串
        self.parameters_schema = tool_info.get("inputSchema", {   # 参数 JSON Schema
            "type": "object", "properties": {}, "required": []    # 兜底：空对象、无必填
        })

    # 【什么】标出这是远程 MCP 工具
    # 【作用】权限过滤时本地工具认 tool:{name}，MCP 工具额外认 tool:mcp:{name}
    # 【为什么】类型标记是权限系统区分本地/远程工具的依据
    @property                                                  # 声明为只读属性
    def tool_type(self) -> str:                                # 返回工具类型标识
        return "mcp"                                           # 固定返回 "mcp"

    # 【什么】把工具转成 OpenAI function schema
    # 【作用】模型看到的工具表就是它；description 空缺会让模型不容易选中该工具
    # 【为什么】LLM 只认 OpenAI function calling 格式，协议元数据必须做一次转译
    def to_openai_format(self) -> dict:                        # 返回 OpenAI 函数描述结构
        return {                                               # 组装标准 function schema
            "type": "function",                                # 顶层固定为 function
            "function": {                                      # 函数体
                "name": self.name,                             # 工具名
                "description": self.description,               # 描述（模型选工具的主要依据）
                "parameters": self.parameters_schema,          # 参数 JSON Schema（直接沿用 MCP inputSchema）
            },
        }

    # 【什么】按 schema 的 required 校验必填参数
    # 【作用】缺参数就地抛 MCPException，把错误拦在发请求之前
    # 【为什么】本地拦一道比让请求跑到远程再报错便宜，且错误信息更贴近调用方
    def validate_params(self, params: dict) -> dict:           # 入参：待校验参数；返回原参数（校验通过）
        required = self.parameters_schema.get("required", [])   # 必填参数名列表，无 schema 时是 []
        missing = [r for r in required if r not in (params or {})]  # 列表推导：收集所有缺失的必填参数名
        if missing:                                            # 有缺失
            raise MCPException(f"参数缺失: {missing}", f"tool={self.name}")  # 抛错，附缺失名单与工具名
        return params or {}                                    # 校验通过；None 归一化为空 dict

    # 【什么】BaseTool 抽象方法的实现：真正调远程工具
    # 【作用】只负责"调远程 + 提取文本"，计时/异常兜底/封装交给父类 run
    # 【为什么】父类 run 定死了统一链路，子类只需填执行细节，行为才能全局一致
    def execute(self, **kwargs) -> ToolResult:                 # 入参为工具关键字参数；返回统一结果结构
        """真正调用 MCP 远程工具（BaseTool 抽象方法的实现）。

        BaseTool.run 的统一链路是：validate_params → execute → 封装 ToolResult，
        这里只负责"调远程 + 提取文本"，参数校验/计时/异常兜底交给父类 run。
        """
        result = self._client.call_tool(self.name, kwargs)     # 转发给 MCP 客户端执行远程调用
        text_parts = []                                        # 文本片段收集器
        for item in result.get("content", []):                 # 遍历结果内容块
            if item.get("type") == "text":                     # 只收文本块
                text_parts.append(item.get("text", ""))        # 收进收集器
        return ToolResult(                                     # 封装统一结果结构
            success=True,                                      # 走到这里说明没抛异常，标记成功
            result="\n".join(text_parts) if text_parts else result,  # 有文本块就按行拼接，否则原样返回 result
        )
