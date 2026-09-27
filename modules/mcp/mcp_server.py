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
import sys
import json
from datetime import datetime
from typing import Dict, Optional
from modules.common.logger import get_logger
from modules.mcp.protocol import (
    parse_message, create_response, create_error,
    INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR,
)
from config import get_settings

_logger = get_logger("mcp_server")


# 【什么】独立进程里的 MCP 工具服务端
# 【作用】收一行 JSON-RPC 请求、回一行响应；对外提供 3 个工具
class MCPServer:
    """独立 MCP 工具服务（JSON-RPC 2.0 + stdio）。"""

    # 【什么】服务端构造：读配置 + 备好工具清单
    # 【作用】工具清单在启动时定好，之后 tools/list 一直返回同一份
    def __init__(self):
        s = get_settings()
        self._tavily_key = s.tavily_api_key    # Tavily API Key；留空时搜索工具返回提示文本
        self._tools = self._define_tools()     # 本 Server 的工具清单（3 个）
        self._initialized = False              # 握手标记：收到 initialize 后置 True

    # 【什么】本 Server 提供的工具清单（硬编码 3 个）
    # 【作用】tools/list 直接返回它；改动这里就是改 Server 对外暴露的能力
        """定义 Server 提供的工具列表。"""
    def _define_tools(self) -> list:
        return [
            {
                "name": "web_search",          # 网络搜索：必填 query，可选 max_results
                "description": "网络搜索工具，获取互联网最新信息。输入搜索关键词，返回搜索结果摘要。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "最大返回结果数，默认3",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_current_time",    # 当前时间：可选 timezone，无必填参数
                "description": "获取当前时间，支持指定时区。返回格式化时间字符串。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "时区，如 'Asia/Shanghai'，默认本机时区",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "knowledge_stats",     # 知识库统计（演示企业内部工具）：无参数
                "description": "获取知识库统计信息（演示企业内部MCP工具）。",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]


    # 【什么】Tavily 网络搜索的实现
    # 【作用】给 web_search 工具提供能力；没有 API Key 时返回提示文本，不抛异常
    def _web_search(self, query: str, max_results: int = 3) -> str:   # 默认最多返回 3 条
        """Tavily 网络搜索。"""
        if not self._tavily_key:
            return "搜索工具未配置 Tavily API Key，无法执行网络搜索。"
        try:
            import requests
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._tavily_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=15,        # 请求 Tavily 的超时（秒）
            )
            data = resp.json()
            results = []
            # 只取前 max_results 条（默认 3），每条正文再截断到 200 字，避免把整页塞进上下文
            for item in data.get("results", [])[:max_results]:
                results.append(
                    f"标题: {item.get('title', '')}\n"
                    f"内容: {item.get('content', '')[:200]}\n"
                    f"链接: {item.get('url', '')}"
                )
            return "\n---\n".join(results) if results else "未找到相关结果"
        except Exception as e:
            return f"搜索失败: {e}"

    # 【什么】取当前时间的实现
    # 【作用】给 get_current_time 工具提供能力；时区非法时退回本机时间，不抛异常
    def _get_current_time(self, timezone: str = "") -> str:   # 时区留空表示用本机时区
        """获取当前时间。"""
        from datetime import timezone as tz
        import pytz
        try:
            if timezone:
                tz_obj = pytz.timezone(timezone)
                now = datetime.now(tz_obj)
            else:
                now = datetime.now()
            return now.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 【什么】知识库统计的实现（演示用，返回固定内容）
    # 【作用】给 knowledge_stats 工具提供能力，展示企业内部 MCP 工具长什么样
    def _knowledge_stats(self) -> str:
        """知识库统计（演示企业内部工具）。"""
        return json.dumps({
            "knowledge_base": "Milvus knowledge_base",
            "status": "运行中",
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False)


    # 【什么】请求分发：按 method 分到不同分支
    # 【作用】四种 method 各回一条响应；未知方法回 -32601，内部异常回 -32603
    # 【返回值为什么是 Optional[str]】notifications/* 这类"通知"按 JSON-RPC 规范不该响应，
    #   走的是 return None 分支；其余 method 都返回响应 JSON 字符串。
    def handle_request(self, msg: dict) -> Optional[str]:
        """处理 JSON-RPC 请求，返回响应 JSON；通知类消息返回 None（不发响应）。"""
        req_id = msg.get("id", 0)
        method = msg.get("method", "")

        try:
            if method == "initialize":
                self._initialized = True
                return create_response(req_id, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "ai-agent-mcp-server",
                        "version": "1.0.0",
                    },
                })

            elif method == "notifications/initialized":
                _logger.info("MCP Client 已完成初始化")
                return None

            elif method == "tools/list":
                return create_response(req_id, {"tools": self._tools})

            elif method == "tools/call":
                return self._handle_tool_call(msg, req_id)

            else:
                return create_error(
                    req_id, METHOD_NOT_FOUND,
                    f"未知方法: {method}",
                )
        except Exception as e:
            _logger.error(f"MCP请求处理异常: {method} | {e}")
            return create_error(req_id, INTERNAL_ERROR, str(e))

    # 【什么】tools/call 的具体执行：按工具名分派到本地方法
    # 【作用】结果统一包成 content:[{type:text}] 回给 Client；工具名不认识回 -32601
    def _handle_tool_call(self, msg: dict, req_id: int) -> str:
        """处理 tools/call 请求。"""
        params = msg.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "web_search":
            result = self._web_search(**arguments)
        elif tool_name == "get_current_time":
            result = self._get_current_time(**arguments)
        elif tool_name == "knowledge_stats":
            result = self._knowledge_stats()
        else:
            return create_error(
                req_id, METHOD_NOT_FOUND,
                f"工具不存在: {tool_name}",
            )

        return create_response(req_id, {
            "content": [{"type": "text", "text": str(result)}],
            "isError": False,
        })

    # 【什么】stdio 事件循环：一行请求 → 一行响应
    # 【作用】Server 的主循环；空行跳过，解析失败回 -32600，其他异常只记日志不回包
    def run(self):
        """启动 stdio 事件循环，监听 Client 请求。"""
        _logger.info("MCP Server 启动，等待连接...")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = parse_message(line)
                response = self.handle_request(msg)
                if response:
                    print(response, flush=True)
            except ValueError as e:
                error_resp = create_error(0, INVALID_REQUEST, str(e))
                print(error_resp, flush=True)
            except Exception as e:
                _logger.error(f"Server内部异常: {e}")


if __name__ == "__main__":
    MCPServer().run()
