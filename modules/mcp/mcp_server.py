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
from typing import Dict
from modules.common.logger import get_logger
from modules.mcp.protocol import (
    parse_message, create_response, create_error,
    INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR,
)
from config import get_settings

_logger = get_logger("mcp_server")


class MCPServer:
    """独立 MCP 工具服务（JSON-RPC 2.0 + stdio）。"""

    def __init__(self):
        s = get_settings()
        self._tavily_key = s.tavily_api_key
        self._tools = self._define_tools()
        self._initialized = False

        """定义 Server 提供的工具列表。"""
    def _define_tools(self) -> list:
        return [
            {
                "name": "web_search",
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
                "name": "get_current_time",
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
                "name": "knowledge_stats",
                "description": "获取知识库统计信息（演示企业内部MCP工具）。",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    # ==================== 工具实现 ====================

    def _web_search(self, query: str, max_results: int = 3) -> str:
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
                timeout=15,
            )
            data = resp.json()
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(
                    f"标题: {item.get('title', '')}\n"
                    f"内容: {item.get('content', '')[:200]}\n"
                    f"链接: {item.get('url', '')}"
                )
            return "\n---\n".join(results) if results else "未找到相关结果"
        except Exception as e:
            return f"搜索失败: {e}"

    def _get_current_time(self, timezone: str = "") -> str:
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

    def _knowledge_stats(self) -> str:
        """知识库统计（演示企业内部工具）。"""
        return json.dumps({
            "knowledge_base": "Milvus knowledge_base",
            "status": "运行中",
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False)

    # ==================== MCP 协议处理 ====================

    def handle_request(self, msg: dict) -> str:
        """处理 JSON-RPC 请求，返回响应 JSON。"""
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
                return None  # 通知无需响应

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

    def _handle_tool_call(self, msg: dict, req_id: int) -> str:
        """处理 tools/call 请求。"""
        params = msg.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # 路由到具体工具
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

        # MCP 标准返回格式: content + isError
        return create_response(req_id, {
            "content": [{"type": "text", "text": str(result)}],
            "isError": False,
        })

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
