"""
MCP 模块
=========
完整落地 Model Context Protocol 标准化工具通信协议。
- MCP Client（集成在Agent内）：发现工具、调用远程工具
- MCP Server（独立部署）：提供外部工具服务
- 协议解析：JSON-RPC 2.0 消息封装与解析
"""
from modules.mcp.protocol import (
    JSONRPCRequest, JSONRPCResponse, JSONRPCError,
    parse_message, create_request, create_response, create_error,
)
from modules.mcp.mcp_client import MCPClient, MCPToolAdapter
from modules.mcp.mcp_server import MCPServer
from modules.mcp.gateway import MCPGateway, MCPServerConfig

__all__ = [
    "JSONRPCRequest", "JSONRPCResponse", "JSONRPCError",
    "parse_message", "create_request", "create_response", "create_error",
    "MCPClient", "MCPToolAdapter", "MCPServer",
    "MCPGateway", "MCPServerConfig",
]
