"""
公共工具模块
============
token统计、异常捕获、格式校验、日志封装等工程化辅助能力。
"""
# 【什么】日志与异常体系的统一出口
# 【作用】上层只写 `from modules.common import ...` 就能拿到 logger 与 6 个异常类，不用记住子模块路径
from modules.common.logger import get_logger
from modules.common.exceptions import (
    AgentException,
    LLMException,
    RAGException,
    ToolException,
    MemoryException,
    MCPException,
    handle_exception,
)
# 【什么】token 计数函数的导出
# 【作用】对外只暴露精确计数 count_tokens 与粗估算 estimate_tokens 两个入口
from modules.common.token_counter import count_tokens, estimate_tokens
# 【什么】校验与清洗函数的导出
# 【作用】对外暴露 JSON 校验/修复、输入清洗、注入检测四个入口
from modules.common.validator import (
    validate_json,
    repair_json,
    sanitize_input,
    check_prompt_injection,
)

# 【什么】导出白名单，共 14 个名字
# 【作用】限定 `from modules.common import *` 的范围，未列出的内部符号不会被带出去
__all__ = [
    "get_logger",
    "AgentException", "LLMException", "RAGException",
    "ToolException", "MemoryException", "MCPException",
    "handle_exception",
    "count_tokens", "estimate_tokens",
    "validate_json", "repair_json", "sanitize_input", "check_prompt_injection",
]
