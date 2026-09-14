"""
异常体系模块
============
企业级异常分层定义，覆盖 LLM、RAG、Agent、工具、记忆、MCP 全链路。
提供统一异常处理装饰器，支持重试、兜底、日志记录。
"""
import functools
from typing import Callable
from modules.common.logger import get_logger

_logger = get_logger("exception")


# ---- 基础异常 ----
class AgentException(Exception):
    """Agent 系统基础异常，所有自定义异常的父类。"""
    def __init__(self, message: str, detail: str = ""):
        self.message = message
        self.detail = detail
        super().__init__(message)


class LLMException(AgentException):
    """大模型调用异常：超时、格式错乱、参数非法。"""


class RAGException(AgentException):
    """RAG 检索异常：无结果、低相关、Milvus不可用。"""


class ToolException(AgentException):
    """工具调用异常：参数校验失败、死循环、超时。"""


class MemoryException(AgentException):
    """记忆系统异常：PG读写失败、向量召回异常。"""


class MCPException(AgentException):
    """MCP 协议异常：连接失败、协议解析错误。"""


# ---- 统一异常处理 ----
def handle_exception(
    default=None,
    log_msg: str = "操作异常",
    reraise: bool = False,
):
    """
    装饰器：统一捕获异常，记录日志，返回兜底值。
    
    Args:
        default: 异常时返回的兜底值
        log_msg: 日志描述前缀
        reraise: 是否重新抛出异常（关键路径用）
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AgentException as e:
                _logger.error(
                    f"{log_msg} | {type(e).__name__}: {e.message} | detail={e.detail}"
                )
                if reraise:
                    raise
                return default
            except Exception as e:
                _logger.error(f"{log_msg} | 未知异常: {type(e).__name__}: {str(e)}")
                if reraise:
                    raise AgentException(log_msg, str(e))
                return default
        return wrapper
    return decorator
