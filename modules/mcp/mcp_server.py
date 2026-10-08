"""
MCP Server 模块（示例独立工具服务）
====================================
独立部署的 MCP Server，提供外部工具服务。
演示：Tavily 网络搜索工具 + 当前时间查询工具。
Agent 通过 MCP Client 标准化协议对接，实现工具解耦、可复用、可热更新。

运行方式（独立进程）：
    python -m modules.mcp.mcp_server

MCP 协议交互流程（JSON-RPC 2.0 + stdio）：
    1. Client → Server: initialize（握手）
    2. Client → Server: notifications/initialized（完成通知）
    3. Client → Server: tools/list（发现工具）
    4. Client → Server: tools/call（调用工具）
"""
import sys                                                          # 标准库：访问 stdin（协议帧从这里一行行读进来）
import json                                                         # 标准库：JSON 序列化（knowledge_stats 返回结构化文本）
from datetime import datetime                                       # 标准库：取当前时间（get_current_time 工具用）
from typing import Dict, Optional                                   # 类型标注工具
from modules.common.logger import get_logger                        # 项目统一日志入口
from modules.mcp.protocol import (                                  # MCP 协议工具函数与错误码
    parse_message, create_response, create_error,                   # 解析请求帧 / 构造正常响应帧 / 构造错误帧
    INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR,  # 协议错误码常量
)
from config import get_settings                                     # 全局配置入口，取 Tavily API Key

_logger = get_logger("mcp_server")                                  # 模块级日志器，名字 "mcp_server" 用于日志过滤


# 【什么】独立进程里的 MCP 工具服务端
# 【作用】收一行 JSON-RPC 请求、回一行响应；对外提供 3 个工具
# 【为什么】工具跑在独立进程里：崩溃只崩自己，升级工具不用动 Agent 主进程
class MCPServer:                                                    # MCP 工具服务端类
    """独立 MCP 工具服务（JSON-RPC 2.0 + stdio）。"""

    # 【什么】服务端构造：读配置 + 备好工具清单
    # 【作用】工具清单在启动时定好，之后 tools/list 一直返回同一份
    # 【为什么】清单静态化：发现阶段不用每次现算，响应快且内容稳定可缓存
    def __init__(self):                                             # 无参数构造
        s = get_settings()                                          # 取全局配置单例
        self._tavily_key = s.tavily_api_key    # Tavily API Key；留空时搜索工具返回提示文本
        self._tools = self._define_tools()     # 本 Server 的工具清单（3 个）
        self._initialized = False              # 握手标记：收到 initialize 后置 True

    # 【什么】本 Server 提供的工具清单（硬编码 3 个）
    # 【作用】tools/list 直接返回它；改动这里就是改 Server 对外暴露的能力
    # 【为什么】演示用硬编码；生产版可以从数据库/配置中心动态加载
    def _define_tools(self) -> list:                                # 返回工具元数据列表
        """定义 Server 提供的工具列表。"""
        return [                                                    # 工具清单（每项含 name/description/inputSchema）
            {
                "name": "web_search",          # 网络搜索：必填 query，可选 max_results
                "description": "网络搜索工具，获取互联网最新信息。输入搜索关键词，返回搜索结果摘要。",  # 工具描述（模型选工具的依据）
                "inputSchema": {                                    # 参数 JSON Schema
                    "type": "object",                               # 参数整体是一个对象
                    "properties": {                                 # 逐个参数声明
                        "query": {                                  # 搜索关键词参数
                            "type": "string",                       # 字符串类型
                            "description": "搜索关键词",            # 参数说明
                        },
                        "max_results": {                            # 最大返回条数参数
                            "type": "integer",                      # 整数类型
                            "description": "最大返回结果数，默认3",  # 参数说明
                        },
                    },
                    "required": ["query"],                          # 必填参数：只有 query
                },
            },
            {
                "name": "get_current_time",    # 当前时间：可选 timezone，无必填参数
                "description": "获取当前时间，支持指定时区。返回格式化时间字符串。",  # 工具描述
                "inputSchema": {                                    # 参数 JSON Schema
                    "type": "object",                               # 参数整体是一个对象
                    "properties": {                                 # 参数声明
                        "timezone": {                               # 时区参数
                            "type": "string",                       # 字符串类型
                            "description": "时区，如 'Asia/Shanghai'，默认本机时区",  # 参数说明
                        },
                    },
                    "required": [],                                 # 无必填参数
                },
            },
            {
                "name": "knowledge_stats",     # 知识库统计（演示企业内部工具）：无参数
                "description": "获取知识库统计信息（演示企业内部MCP工具）。",  # 工具描述
                "inputSchema": {                                    # 参数 JSON Schema
                    "type": "object",                               # 参数整体是一个对象
                    "properties": {},                               # 无任何参数
                    "required": [],                                 # 无必填参数
                },
            },
        ]


    # 【什么】Tavily 网络搜索的实现
    # 【作用】给 web_search 工具提供能力；没有 API Key 时返回提示文本，不抛异常
    # 【为什么】失败返回文本而非异常：LLM 能读到失败原因并自行换策略，比协议层报错更有用
    def _web_search(self, query: str, max_results: int = 3) -> str:   # 默认最多返回 3 条
        """Tavily 网络搜索。"""
        if not self._tavily_key:                                    # 没配 API Key
            return "搜索工具未配置 Tavily API Key，无法执行网络搜索。"  # 返回提示文本，让模型换路子
        try:
            import requests                                         # 延迟 import：不搜就不加载 requests
            resp = requests.post(                                   # 调 Tavily 搜索接口
                "https://api.tavily.com/search",                    # 接口地址
                json={                                              # 请求体
                    "api_key": self._tavily_key,                    # 鉴权 Key
                    "query": query,                                 # 搜索关键词
                    "max_results": max_results,                     # 最大返回条数
                    "search_depth": "basic",                        # 基础搜索深度（比 advanced 快且省额度）
                },
                timeout=15,        # 请求 Tavily 的超时（秒）
            )
            data = resp.json()                                      # 解析响应 JSON
            results = []                                            # 结果片段收集器
            # 只取前 max_results 条（默认 3），每条正文再截断到 200 字，避免把整页塞进上下文
            for item in data.get("results", [])[:max_results]:      # 遍历截取后的结果列表
                results.append(                                     # 拼一条"标题+内容+链接"的文本块
                    f"标题: {item.get('title', '')}\n"              # 标题行
                    f"内容: {item.get('content', '')[:200]}\n"      # 内容行（截断 200 字）
                    f"链接: {item.get('url', '')}"                  # 链接行
                )
            return "\n---\n".join(results) if results else "未找到相关结果"  # 多条之间用分隔线连接；空结果给提示
        except Exception as e:                                      # 网络/解析等任何异常
            return f"搜索失败: {e}"                                 # 返回失败文本，不抛异常（理由见方法头注释）

    # 【什么】取当前时间的实现
    # 【作用】给 get_current_time 工具提供能力；时区非法时退回本机时间，不抛异常
    # 【为什么】时区字符串来自 LLM，可能编造；兜底本机时间保证工具永远有输出
    def _get_current_time(self, timezone: str = "") -> str:   # 时区留空表示用本机时区
        """获取当前时间。"""
        from datetime import timezone as tz                         # 延迟 import（当前实现未直接用，保留占位）
        import pytz                                                 # 时区库：按 IANA 名解析时区
        try:
            if timezone:                                            # 调用方给了时区名
                tz_obj = pytz.timezone(timezone)                    # 解析时区对象（非法名抛 UnknownTimeZoneError）
                now = datetime.now(tz_obj)                          # 取该时区的当前时间
            else:                                                   # 没给时区
                now = datetime.now()                                # 用本机时区时间
            return now.strftime("%Y-%m-%d %H:%M:%S %Z")             # 格式化输出（含时区缩写）
        except Exception:                                           # 时区非法等任何异常
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")     # 兜底：本机时间，不带时区后缀

    # 【什么】知识库统计的实现（演示用，返回固定内容）
    # 【作用】给 knowledge_stats 工具提供能力，展示企业内部 MCP 工具长什么样
    # 【为什么】演示工具返回固定结构，真实版可替换为查 Milvus 等真实存储
    def _knowledge_stats(self) -> str:                              # 返回 JSON 字符串
        """知识库统计（演示企业内部工具）。"""
        return json.dumps({                                         # 序列化统计信息
            "knowledge_base": "Milvus knowledge_base",              # 知识库标识
            "status": "运行中",                                     # 运行状态
            "timestamp": datetime.now().isoformat(),                # 统计时刻（ISO 格式）
        }, ensure_ascii=False)                                      # 不转义中文，输出可读


    # 【什么】请求分发：按 method 分到不同分支
    # 【作用】四种 method 各回一条响应；未知方法回 -32601，内部异常回 -32603
    # 【为什么】分发是协议路由层：method 决定语义，req_id 保证响应与请求配对
    # 【返回值为什么是 Optional[str]】notifications/* 这类"通知"按 JSON-RPC 规范不该响应，
    #   走的是 return None 分支；其余 method 都返回响应 JSON 字符串。
    def handle_request(self, msg: dict) -> Optional[str]:           # 入参为已解析的请求 dict，返回响应 JSON 或 None
        """处理 JSON-RPC 请求，返回响应 JSON；通知类消息返回 None（不发响应）。"""
        req_id = msg.get("id", 0)                                   # 取请求号（响应要原样带回），缺省兜 0
        method = msg.get("method", "")                              # 取方法名，缺省兜空串（走未知方法分支）

        try:
            if method == "initialize":                              # 握手请求
                self._initialized = True                            # 置握手标记
                return create_response(req_id, {                    # 回握手响应
                    "protocolVersion": "2024-11-05",                # 本 Server 支持的协议版本
                    "capabilities": {"tools": {}},                  # 服务能力：提供 tools 相关方法
                    "serverInfo": {                                 # 服务端自我介绍
                        "name": "ai-agent-mcp-server",              # 服务名
                        "version": "1.0.0",                         # 版本号
                    },
                })

            elif method == "notifications/initialized":             # 客户端初始化完成通知
                _logger.info("MCP Client 已完成初始化")             # 只记日志
                return None                                         # 通知按规范不回响应

            elif method == "tools/list":                            # 工具发现请求
                return create_response(req_id, {"tools": self._tools})  # 回启动时备好的工具清单

            elif method == "tools/call":                            # 工具调用请求
                return self._handle_tool_call(msg, req_id)          # 转给专用处理方法

            else:                                                   # 不认识的方法名
                return create_error(                                # 回方法不存在错误
                    req_id, METHOD_NOT_FOUND,                       # 错误码 -32601
                    f"未知方法: {method}",                          # 附方法名便于排查
                )
        except Exception as e:                                      # 处理过程任何内部异常
            _logger.error(f"MCP请求处理异常: {method} | {e}")       # 记错误日志
            return create_error(req_id, INTERNAL_ERROR, str(e))     # 回内部错误 -32603，保证请求必有响应

    # 【什么】tools/call 的具体执行：按工具名分派到本地方法
    # 【作用】结果统一包成 content:[{type:text}] 回给 Client；工具名不认识回 -32601
    # 【为什么】分派表用 if/elif 硬编码：演示 Server 工具少，直白优先；工具多了再换注册表
    def _handle_tool_call(self, msg: dict, req_id: int) -> str:     # 入参：原始请求与请求号，返回响应 JSON
        """处理 tools/call 请求。"""
        params = msg.get("params", {})                              # 取 params 段，缺省兜空 dict
        tool_name = params.get("name", "")                          # 取要调用的工具名
        arguments = params.get("arguments", {})                     # 取工具入参，缺省兜空 dict

        if tool_name == "web_search":                               # 网络搜索工具
            result = self._web_search(**arguments)                  # 关键字参数展开调用（query/max_results）
        elif tool_name == "get_current_time":                       # 时间查询工具
            result = self._get_current_time(**arguments)            # 关键字参数展开调用（timezone）
        elif tool_name == "knowledge_stats":                        # 知识库统计工具
            result = self._knowledge_stats()                        # 无参调用
        else:                                                       # 不认识的工具名
            return create_error(                                    # 回方法不存在错误
                req_id, METHOD_NOT_FOUND,                           # 错误码 -32601
                f"工具不存在: {tool_name}",                         # 附工具名便于排查
            )

        return create_response(req_id, {                            # 工具执行成功，回正常响应
            "content": [{"type": "text", "text": str(result)}],     # 结果统一包成文本内容块
            "isError": False,                                       # 标记非业务错误
        })

    # 【什么】stdio 事件循环：一行请求 → 一行响应
    # 【作用】Server 的主循环；空行跳过，解析失败回 -32600，其他异常只记日志不回包
    # 【为什么】按行分帧是 MCP stdio 的约定；flush=True 保证响应立即送达，不等缓冲区满
    def run(self):                                                  # 无参数无返回；阻塞运行直到 stdin 关闭
        """启动 stdio 事件循环，监听 Client 请求。"""
        _logger.info("MCP Server 启动，等待连接...")                # 记启动日志（走 stderr，不污染 stdout 协议流）
        for line in sys.stdin:                                      # 阻塞式逐行读 stdin；Client 断开关闭管道后循环结束
            line = line.strip()                                     # 去掉首尾空白与换行
            if not line:                                            # 空行
                continue                                            # 跳过，不响应
            try:
                msg = parse_message(line)                           # 解析 JSON-RPC 请求帧（非法 JSON 抛 ValueError）
                response = self.handle_request(msg)                 # 分发处理，得到响应 JSON 或 None
                if response:                                        # 有响应（通知返回 None，跳过）
                    print(response, flush=True)                     # 写一行响应到 stdout 并立即冲刷
            except ValueError as e:                                 # 请求帧解析失败（非法 JSON/结构）
                error_resp = create_error(0, INVALID_REQUEST, str(e))  # 构造无效请求错误 -32600（无 id 可配，用 0）
                print(error_resp, flush=True)                       # 回错误帧并立即冲刷
            except Exception as e:                                  # 其他未预期异常
                _logger.error(f"Server内部异常: {e}")               # 只记日志不回包：此时连 req_id 都未必拿得到


if __name__ == "__main__":                                          # 作为脚本直接运行时（python -m modules.mcp.mcp_server）
    MCPServer().run()                                               # 构造服务端并进入 stdio 事件循环
