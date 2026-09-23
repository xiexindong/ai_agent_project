"""
工具模块
=========
本地工具集合 + 工具注册 + 参数校验。
兼容【本地原生工具】+【MCP远程工具】双模式调度。
覆盖边界2（参数非法校验）、边界3（工具死循环拦截）。
"""
# 【什么】工具基类与结果对象
# 【作用】定义 run()/execute() 统一接口，本地工具与 MCP 工具共用
from modules.tools.base_tool import BaseTool, ToolResult
# 【什么】两个进程内本地工具
# 【作用】计算器与业务数据查询，进程里直接跑，无子进程开销
from modules.tools.calculator import CalculatorTool
from modules.tools.data_query import DataQueryTool
# 【什么】工具注册器
# 【作用】工具调度唯一入口，内含 5 次上限与同参去重两道闸门
from modules.tools.tool_registry import ToolRegistry
# 【什么】工具的语义检索索引
# 【作用】工具多时只把 top-k 个工具喂给模型，省 prompt token
from modules.tools.tool_rag import ToolRAGIndex, ToolHit

# 【什么】本包对外的符号白名单
# 【作用】限制 import * 能拿到的名字，内部结构不对外承诺
__all__ = [
    "BaseTool", "ToolResult", "CalculatorTool", "DataQueryTool", "ToolRegistry",
    "ToolRAGIndex", "ToolHit",
]
