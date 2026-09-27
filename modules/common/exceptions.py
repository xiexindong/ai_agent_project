"""
异常体系模块
============
企业级异常分层定义，覆盖 LLM、RAG、Agent、工具、记忆、MCP 全链路。
提供统一异常处理装饰器，支持重试、兜底、日志记录。
"""
import functools
from typing import Any, Callable, Literal, ParamSpec, TypeVar, overload
from modules.common.logger import get_logger

_logger = get_logger("exception")

# 【什么】两个类型变量：P 收参数签名，R 收返回值类型
# 【作用】让 handle_exception 包装后仍保留原函数的参数与返回值类型，不退化成 Any
P = ParamSpec("P")
R = TypeVar("R")


# 【什么】Agent 系统异常基类
# 【作用】统一携带 message 与 detail 两个字段，日志、API 返回、审计都读这两个字段
class AgentException(Exception):
    """Agent 系统基础异常，所有自定义异常的父类。"""
    # 【什么】异常构造：一条给人看的 message + 一段给排查用的 detail
    # 【作用】detail 放原始入参/堆栈等长内容，不混进 message 影响前端展示
    def __init__(self, message: str, detail: str = ""):
        self.message = message        # 简短可读的错误描述，直接展示给用户
        self.detail = detail          # 详细上下文（入参、原始报错等），只进日志
        super().__init__(message)     # 交给 Exception 保管 args[0]，str(e) 才等于 message


# 【什么】大模型调用异常
# 【作用】超时、返回格式错乱、参数非法时抛，便于上层区分是模型问题还是业务问题
class LLMException(AgentException):
    """大模型调用异常：超时、格式错乱、参数非法。"""


# 【什么】RAG 检索异常
# 【作用】召回无结果、相关度过低、Milvus 不可用时抛，上层可据此降级为纯模型回答
class RAGException(AgentException):
    """RAG 检索异常：无结果、低相关、Milvus不可用。"""


# 【什么】工具调用异常
# 【作用】参数校验失败、死循环、执行超时时抛，工具层的错误统一收口到这里
class ToolException(AgentException):
    """工具调用异常：参数校验失败、死循环、超时。"""


# 【什么】记忆系统异常
# 【作用】PG 读写失败、向量召回异常时抛，记忆出问题不该让整轮对话崩掉
class MemoryException(AgentException):
    """记忆系统异常：PG读写失败、向量召回异常。"""


# 【什么】MCP 协议异常
# 【作用】MCP 连接失败、协议解析错误时抛，与本地工具异常区分开
class MCPException(AgentException):
    """MCP 协议异常：连接失败、协议解析错误。"""


# 【什么】类型重载一：reraise=False 分支
# 【作用】告诉类型检查器这条分支返回值仍是 R，调用方不用再判 None
@overload
def handle_exception(
    # 【为什么是 Any 而不是 R】default 一旦参与 R 的推断，写 @handle_exception(default=None)
    #   时 R 会被锁成 None，于是"被装饰函数必须返回 None"，所有带返回值的函数都会报类型不兼容。
    #   改成 Any 后，R 只从被装饰函数本身推断，包装后返回值类型原样保留。
    default: Any = ...,                # 兜底值；不参与返回值类型推断
    log_msg: str = "操作异常",        # 日志前缀，标明是哪个调用点出错
    reraise: Literal[False] = False,  # 固定 False：吞掉异常，返回 default
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

# 【什么】类型重载二：reraise=True 分支
# 【作用】这条分支异常会继续向上抛，返回值类型标注为 R（不会返回 default）
@overload
def handle_exception(
    default: Any = None,              # 兜底值；reraise=True 时实际不会被返回
    log_msg: str = "操作异常",          # 日志前缀
    reraise: Literal[True] = True,    # 固定 True：记完日志继续抛给上层
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def handle_exception(
    default: Any = None,          # 异常时返回的兜底值
    log_msg: str = "操作异常",      # 日志前缀，便于定位是哪个调用点出错
    reraise: bool = False,        # True = 记完日志继续向上抛，关键路径用
) -> Callable[[Callable[P, R]], Callable[P, R | None]]:
    """
    装饰器：统一捕获异常，记录日志，返回兜底值。

    Args:
        default: 异常时返回的兜底值
        log_msg: 日志描述前缀
        reraise: 是否重新抛出异常（关键路径用）
    """
    # 【什么】真正套在函数上的装饰器
    # 【作用】把原函数换成包装体，异常在此收口，调用方拿到返回值而不是裸异常
    def decorator(func: Callable[P, R]) -> Callable[P, R | None]:
        # 【wraps 的作用】保留原函数的 __name__ / __doc__，被包装后元信息不丢
        @functools.wraps(func)
        # 【什么】替身函数，签名与原函数一致（*args / **kwargs 原样转发）
        # 【作用】所有异常在这里被捕获并按 reraise 决定是吞掉还是抛出
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
            try:
                return func(*args, **kwargs)
            except AgentException as e:
                # 【自定义异常】有 message/detail 两个字段，分开打出来便于检索
                _logger.error(
                    f"{log_msg} | {type(e).__name__}: {e.message} | detail={e.detail}"
                )
                if reraise:
                    raise                 # 原样抛出，保留原始异常类型与堆栈
                return default            # 不重抛时返回兜底值，调用方照常往下走
            except Exception as e:
                # 【非自定义异常】没有 message/detail，只能打类型名与 str(e)
                _logger.error(f"{log_msg} | 未知异常: {type(e).__name__}: {str(e)}")
                if reraise:
                    raise AgentException(log_msg, str(e))   # 统一包装成 AgentException 再抛
                return default
        return wrapper
    return decorator
