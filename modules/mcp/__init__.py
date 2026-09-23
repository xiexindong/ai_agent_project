"""
MCP 模块
=========
完整落地 Model Context Protocol 标准化工具通信协议。
- MCP Client（集成在Agent内）：发现工具、调用远程工具
- MCP Server（独立部署）：提供外部工具服务
- 协议解析：JSON-RPC 2.0 消息封装与解析
"""
# 【什么】协议层：JSON-RPC 2.0 消息的数据类与构造/解析函数
# 【作用】对外统一从包根导入，调用方不必知道它们定义在 modules.mcp.protocol
from modules.mcp.protocol import (
    JSONRPCRequest, JSONRPCResponse, JSONRPCError,
    parse_message, create_request, create_response, create_error,
)
# 【什么】Client 侧：连子进程、发现工具、把远程工具适配成本地 BaseTool
# 【作用】Agent 侧只认 MCPClient / MCPToolAdapter 两个入口
from modules.mcp.mcp_client import MCPClient, MCPToolAdapter
# 【什么】Server 侧：独立进程里提供工具的那一端
# 【作用】python -m modules.mcp.mcp_server 时跑的就是它
from modules.mcp.mcp_server import MCPServer
# 【什么】网关：把多个 MCP Server 收敛成一个统一入口
# 【作用】对外只暴露工具目录、按需注册、统一调用这三件事
from modules.mcp.gateway import MCPGateway, MCPServerConfig

# 【什么】包的公开导出名单
# 【作用】限定 from modules.mcp import * 能拿到什么，内部符号不外泄
__all__ = [
    "JSONRPCRequest", "JSONRPCResponse", "JSONRPCError",
    "parse_message", "create_request", "create_response", "create_error",
    "MCPClient", "MCPToolAdapter", "MCPServer",
    "MCPGateway", "MCPServerConfig",
]
