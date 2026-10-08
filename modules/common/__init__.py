"""
公共工具模块
============
token统计、异常捕获、格式校验、日志封装等工程化辅助能力。
"""
# 【什么】日志与异常体系的统一出口
# 【作用】上层只写 `from modules.common import ...` 就能拿到 logger 与 6 个异常类，不用记住子模块路径
# 【为什么】收敛导入入口：子模块路径日后调整时，只改这一处，上层业务代码零改动
from modules.common.logger import get_logger   # 导出日志工厂：各模块用它取带模块标识的 logger
from modules.common.exceptions import (         # 导出 6 个异常类 + 统一异常处理装饰器
    AgentException,                            # 异常基类：所有自定义异常的父类
    LLMException,                              # 大模型调用异常
    RAGException,                              # RAG 检索异常
    ToolException,                             # 工具调用异常
    MemoryException,                           # 记忆系统异常
    MCPException,                              # MCP 协议异常
    handle_exception,                          # 统一异常处理装饰器
)
# 【什么】token 计数函数的导出
# 【作用】对外只暴露精确计数 count_tokens 与粗估算 estimate_tokens 两个入口
from modules.common.token_counter import count_tokens, estimate_tokens   # 导出 token 精确计数与字符公式粗估算
# 【什么】校验与清洗函数的导出
# 【作用】对外暴露 JSON 校验/修复、输入清洗、注入检测四个入口
from modules.common.validator import (         # 导出校验清洗四件套
    validate_json,                             # JSON 合法性校验（兼容代码块包裹）
    repair_json,                               # 近似 JSON 尽力修复
    sanitize_input,                            # 用户输入清洗（控制字符/截断/去空白）
    check_prompt_injection,                    # Prompt 注入检测
)

# 【什么】导出白名单，共 14 个名字
# 【作用】限定 `from modules.common import *` 的范围，未列出的内部符号不会被带出去
__all__ = [                                    # 星号导入白名单：只放行列出的名字
    "get_logger",                              # 日志工厂
    "AgentException", "LLMException", "RAGException",   # 异常体系：基类 + LLM/RAG
    "ToolException", "MemoryException", "MCPException", # 异常体系：工具/记忆/MCP
    "handle_exception",                        # 统一异常处理装饰器
    "count_tokens", "estimate_tokens",         # token 计数两个入口
    "validate_json", "repair_json", "sanitize_input", "check_prompt_injection",   # 校验清洗四件套
]
