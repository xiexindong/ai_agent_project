"""
工具执行重试模块
================
回答"工具调用失败了怎么办"里的"自动重试"一环：
    只对【瞬时故障】重试（超时 / 连接断开 / 服务暂不可用）；
    【确定性失败】（参数缺失、权限不足、越界）重试一百次结果也一样，直接把
    失败原因回喂模型，让模型改参数或换工具。

    BaseTool.run() 已经把一切异常兜成 ToolResult(success=False, error=...)，
    所以本模块不碰异常，只看 error 文本归类。

【为什么不无限重试】每次重试都是真实的下游压力（DB / MCP 子进程），
    无限重试 = 把"模型选错工具"升级成"打挂下游服务"。默认 2 次、指数退避。
"""
import re  # 正则模块：识别 error 文本是否命中瞬时故障模式
import time  # 时间模块：重试间隔用 time.sleep
from typing import Callable  # 类型注解：可调用对象

from modules.common.logger import get_logger  # 统一日志工厂
from config import get_settings  # 全局配置：取 tool_retry_attempts / tool_retry_backoff

_logger = get_logger("tool.retry")  # 模块级 logger：本文件日志统一带 "tool.retry" 前缀

# 【什么】瞬时故障的识别清单：error 命中任一模式才算"值得再试一次"
# 【作用】宁可漏判（漏判=不多试一次，失败照常回喂模型），不可误判
#   （误判=对确定性失败白打两次下游，还拖长整轮 ReAct 的墙钟时间）
_TRANSIENT_PATTERNS = [  # 瞬时故障正则模式清单
    r"timeout", r"timed?\s*out", r"超时",             # 网络超时 / 工具自身超时
    r"connection", r"connect", r"连接",                # 连接断开、连不上
    r"temporarily", r"unavailable", r"不可用",          # 服务端临时不可用
    r"reset", r"broken", r"断开", r"reset\s*by\s*peer",  # 连接被重置
    r"502|503|504",                                    # 网关类 HTTP 状态
]
_TRANSIENT_RE = re.compile("|".join(_TRANSIENT_PATTERNS), re.IGNORECASE)  # 把模式清单拼成一个大正则并编译；IGNORECASE 让英文大小写都能命中


# 【什么】工具执行重试策略（指数退避 + 瞬时故障判定）
# 【作用】包住一次工具执行：失败且属于瞬时故障 → 等一下再试；其余直接返回原结果
class ToolRetryPolicy:  # 重试策略类：指数退避 + 瞬时故障判定
    """指数退避重试：attempt 1 失败 → 等 backoff → attempt 2 → 等 backoff*2 → ...

    【输入输出对照】max_retries=2, backoff=0.5，调用一个每次都超时的工具：
        第 1 次执行 → 失败 error="Connection timeout"
        等 0.5s
        第 2 次执行 → 失败 error="Connection timeout"
        等 1.0s（0.5 * 2）
        第 3 次执行 → 失败 error="Connection timeout"
        返回第 3 次的结果（共执行 3 次 = 首次 + 最多重试 2 次）

        换成参数错误 error="工具参数缺失: calculator 缺少必填参数 ['expression']"：
        第 1 次执行 → 失败
        判定：非瞬时故障 → 立刻返回，不重试（重试结果也一样，白等白打）

    【不这么写会怎样】不区分故障类型、无脑重试 2 次：模型传错参数时，
        每次失败调用都变成 3 次下游压力 + 1.5 秒等待，ReAct 一轮 5 次调用
        就是 15 次下游请求，整个对话肉眼可见地变卡。
    """

    def __init__(self, max_retries: int = None, backoff: float = None):  # 构造：None 时从全局配置读默认值
        s = get_settings()  # 拿全局配置对象
        # max_retries 语义：首次执行之外【额外】再试几次（0 = 不重试）
        self._max_retries = max(0, int(max_retries if max_retries is not None  # 显式传了就用传入值
                                        else getattr(s, "tool_retry_attempts", 2)))  # 否则读配置，兜底默认 2；max(0,...) 防负数
        self._backoff = max(0.0, float(backoff if backoff is not None  # 显式传了就用传入值
                                       else getattr(s, "tool_retry_backoff", 0.5)))  # 否则读配置，兜底默认 0.5 秒；max(0.0,...) 防负数

    # 【什么】判断一次失败是否值得重试
    # 【作用】error 文本命中瞬时故障清单 → True；参数/权限/不存在类错误 → False
    @staticmethod  # 静态方法：纯文本判定，不依赖实例状态
    def is_transient(error: str) -> bool:  # 输入 error 文本，输出是否瞬时故障
        """瞬时故障判定：只认超时 / 连接 / 暂不可用类错误。

        【判定例子】
            "Connection timeout"                    → True  （重试有意义）
            "MCP Server 不可用"                     → True
            "工具参数缺失: calculator 缺少必填参数"  → False （重试结果相同）
            "当前身份无权调用工具 data_query"        → False （权限问题，等模型换路）
            "工具不存在: calc"                      → False （幻觉工具名，等模型改名）
        """
        if not error:  # 空串 / None：没有错误信息
            return False  # 不算瞬时故障
        return bool(_TRANSIENT_RE.search(error))  # 用预编译大正则在 error 文本里找命中

    # 【什么】带重试地执行一次工具调用
    # 【作用】fn 返回 ToolResult；失败且瞬时 → 退避后重试；最多 1+max_retries 次
    def run(self, tool_name: str, fn: Callable):  # 包装执行：tool_name 只用于日志，fn 是真正干活的调用
        """执行 fn（一次工具调用），瞬时失败自动重试。

        Args:
            tool_name: 工具名，只用于日志定位。
            fn:        无参调用，返回 ToolResult（BaseTool.run 已保证不抛异常）。

        Returns:
            最后一次执行的 ToolResult（成功或最终失败），不抛异常。
        """
        result = fn()  # 先跑一次（首次执行）
        total = self._max_retries + 1  # 总尝试次数 = 首次 1 次 + 重试 max_retries 次
        for attempt in range(1, total + 1):  # attempt 从 1 到 total：每轮检查当前 result
            if result.success:  # 成功了
                return result                      # 成功：不需要任何后续动作
            error = str(getattr(result, "error", "") or "")  # 取 error 文本，兜底空串
            if not self.is_transient(error):  # 不是瞬时故障（参数错/权限错/不存在）
                _logger.info(f"工具 {tool_name} 失败且非瞬时故障，不重试: {error[:80]}")  # 记 info：说明不重试的原因
                return result                      # 确定性失败：立刻回喂模型
            if attempt >= total:  # 已经是最后一次尝试
                break                              # 重试额度用完，跳出循环返回最后结果
            wait = self._backoff * (2 ** (attempt - 1))    # 0.5 → 1.0 → 2.0（指数退避）
            _logger.warning(  # 记 warning：要重试了
                f"工具 {tool_name} 瞬时故障({attempt}/{self._max_retries})，"  # 第几次重试/总配额
                f"{wait:.1f}s 后重试: {error[:80]}"  # 等多久 + 错误摘要（前 80 字符）
            )
            time.sleep(wait)  # 阻塞等待 backoff 秒
            result = fn()  # 再跑一次，覆盖旧 result
        return result  # 返回最后一次执行的结果（可能成功也可能最终失败）
