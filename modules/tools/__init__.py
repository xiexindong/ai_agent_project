"""
工具模块
=========
本地工具集合 + 工具注册 + 参数校验。
兼容【本地原生工具】+【MCP远程工具】双模式调度。
覆盖边界2（参数非法校验）、边界3（工具死循环拦截）。
"""
from modules.tools.base_tool import BaseTool, ToolResult
from modules.tools.calculator import CalculatorTool
from modules.tools.data_query import DataQueryTool
from modules.tools.tool_registry import ToolRegistry
from modules.tools.tool_rag import ToolRAGIndex, ToolHit

__all__ = [
    "BaseTool", "ToolResult", "CalculatorTool", "DataQueryTool", "ToolRegistry",
    "ToolRAGIndex", "ToolHit",
]
