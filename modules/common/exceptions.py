"""
异常体系模块
============
企业级异常分层定义，覆盖 LLM、RAG、Agent、工具、记忆、MCP 全链路。
提供统一异常处理装饰器，支持重试、兜底、日志记录。
"""
import functools                               # 标准库：functools.wraps 保留被装饰函数的元信息
from typing import Any, Callable, Literal, ParamSpec, TypeVar, overload  # 类型标注工具集：装饰器泛型签名用
from modules.common.logger import get_logger   # 取带模块标识的 logger，本模块统一用 "exception" 标识

_logger = get_logger("exception")              # 模块级 logger：所有异常捕获日志都打到这个标识下

# 【什么】两个类型变量：P 收参数签名，R 收返回值类型
# 【作用】让 handle_exception 包装后仍保留原函数的参数与返回值类型，不退化成 Any
P = ParamSpec("P")                             # P 捕获被装饰函数的参数签名（*args/**kwargs 的形状）
R = TypeVar("R")                               # R 捕获被装饰函数的返回值类型


# 【什么】Agent 系统异常基类
# 【作用】统一携带 message 与 detail 两个字段，日志、API 返回、审计都读这两个字段
# 【为什么】所有自定义异常收口到一棵树上，上层一个 except AgentException 就能兜住全部业务异常
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
# 【为什么】模型侧故障通常可降级/换模型，必须与代码 bug 区分开才能走对处理路径
class LLMException(AgentException):            # 继承基类，自动获得 message/detail 双字段
    """大模型调用异常：超时、格式错乱、参数非法。"""


# 【什么】RAG 检索异常
# 【作用】召回无结果、相关度过低、Milvus 不可用时抛，上层可据此降级为纯模型回答
# 【为什么】检索失败不该让整轮对话崩掉，单独成类便于上层精准降级
class RAGException(AgentException):            # 继承基类
    """RAG 检索异常：无结果、低相关、Milvus不可用。"""


# 【什么】工具调用异常
# 【作用】参数校验失败、死循环、执行超时时抛，工具层的错误统一收口到这里
# 【为什么】工具是外部副作用入口，错误必须独立标记，避免与模型/检索错误混为一谈
class ToolException(AgentException):           # 继承基类
    """工具调用异常：参数校验失败、死循环、超时。"""


# 【什么】记忆系统异常
# 【作用】PG 读写失败、向量召回异常时抛，记忆出问题不该让整轮对话崩掉
# 【为什么】记忆是增强项而非必需项，独立异常便于"记忆挂了照常回答"的容错策略
class MemoryException(AgentException):         # 继承基类
    """记忆系统异常：PG读写失败、向量召回异常。"""


# 【什么】MCP 协议异常
# 【作用】MCP 连接失败、协议解析错误时抛，与本地工具异常区分开
# 【为什么】MCP 走网络协议，故障面与本地工具不同，独立成类便于分别监控与熔断
class MCPException(AgentException):            # 继承基类
    """MCP 协议异常：连接失败、协议解析错误。"""


# 【什么】类型重载一：reraise=False 分支
# 【作用】告诉类型检查器这条分支返回值仍是 R，调用方不用再判 None
# 【为什么】没有重载时，写 reraise=False 的调用方会被迫处理 None，类型体验很差
@overload                                      # 标记这是类型重载声明之一，运行时不执行，仅给类型检查器看
def handle_exception(
    # 【为什么是 Any 而不是 R】default 一旦参与 R 的推断，写 @handle_exception(default=None)
    #   时 R 会被锁成 None，于是"被装饰函数必须返回 None"，所有带返回值的函数都会报类型不兼容。
    #   改成 Any 后，R 只从被装饰函数本身推断，包装后返回值类型原样保留。
    default: Any = ...,                # 兜底值；不参与返回值类型推断
    log_msg: str = "操作异常",        # 日志前缀，标明是哪个调用点出错
    reraise: Literal[False] = False,  # 固定 False：吞掉异常，返回 default
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...   # 返回装饰器：吃进原函数、吐出同签名包装体；... 表示只有声明没有实现

# 【什么】类型重载二：reraise=True 分支
# 【作用】这条分支异常会继续向上抛，返回值类型标注为 R（不会返回 default）
# 【为什么】reraise=True 时 default 实际用不上，类型上仍标 R，调用方按正常返回值处理即可
@overload                                      # 重载声明之二，同样只给类型检查器看
def handle_exception(
    default: Any = None,              # 兜底值；reraise=True 时实际不会被返回
    log_msg: str = "操作异常",          # 日志前缀
    reraise: Literal[True] = True,    # 固定 True：记完日志继续抛给上层
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...   # 声明签名同上；... 表示只有声明没有实现


# 【什么】异常处理装饰器的真正实现（上面两份 overload 只是给类型检查器看的声明）
# 【作用】包住任意函数：正常时透传返回值，异常时记日志并按 reraise 决定吞掉还是上抛
# 【为什么】统一收口异常处理，业务代码不用到处写 try/except，日志格式与兜底策略全项目一致
def handle_exception(
    default: Any = None,          # 异常时返回的兜底值
    log_msg: str = "操作异常",      # 日志前缀，便于定位是哪个调用点出错
    reraise: bool = False,        # True = 记完日志继续向上抛，关键路径用
) -> Callable[[Callable[P, R]], Callable[P, R | None]]:   # 实现签名：返回值可能是 R 也可能是 None（兜底值）
    """
    装饰器：统一捕获异常，记录日志，返回兜底值。

    Args:
        default: 异常时返回的兜底值
        log_msg: 日志描述前缀
        reraise: 是否重新抛出异常（关键路径用）
    """
    # 【什么】真正套在函数上的装饰器
    # 【作用】把原函数换成包装体，异常在此收口，调用方拿到返回值而不是裸异常
    def decorator(func: Callable[P, R]) -> Callable[P, R | None]:   # 吃进原函数，返回同签名的包装体
        # 【wraps 的作用】保留原函数的 __name__ / __doc__，被包装后元信息不丢
        @functools.wraps(func)                                        # 把原函数元信息拷到 wrapper 上
        # 【什么】替身函数，签名与原函数一致（*args / **kwargs 原样转发）
        # 【作用】所有异常在这里被捕获并按 reraise 决定是吞掉还是抛出
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:   # 位置/关键字参数原样转发
            try:
                return func(*args, **kwargs)                          # 正常路径：调用原函数并直接返回结果
            except AgentException as e:                               # 分支一：体系内的自定义异常
                # 【自定义异常】有 message/detail 两个字段，分开打出来便于检索
                _logger.error(
                    f"{log_msg} | {type(e).__name__}: {e.message} | detail={e.detail}"   # 前缀 | 异常名: message | detail=详情
                )
                if reraise:                                           # 关键路径要求继续上抛
                    raise                 # 原样抛出，保留原始异常类型与堆栈
                return default            # 不重抛时返回兜底值，调用方照常往下走
            except Exception as e:                                    # 分支二：体系外的一切未知异常
                # 【非自定义异常】没有 message/detail，只能打类型名与 str(e)
                _logger.error(f"{log_msg} | 未知异常: {type(e).__name__}: {str(e)}")   # 前缀 | 未知异常: 类型: 描述
                if reraise:                                           # 关键路径：包装后继续上抛
                    raise AgentException(log_msg, str(e))   # 统一包装成 AgentException 再抛
                return default                                        # 非关键路径：吞掉异常返回兜底值
        return wrapper                                                # 返回包装体，替换原函数
    return decorator                                                  # 返回装饰器本身，完成三层闭包
