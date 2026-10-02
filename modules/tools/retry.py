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
import re
import time
from typing import Callable

from modules.common.logger import get_logger
from config import get_settings

_logger = get_logger("tool.retry")

# 【什么】瞬时故障的识别清单：error 命中任一模式才算"值得再试一次"
# 【作用】宁可漏判（漏判=不多试一次，失败照常回喂模型），不可误判
#   （误判=对确定性失败白打两次下游，还拖长整轮 ReAct 的墙钟时间）
_TRANSIENT_PATTERNS = [
    r"timeout", r"timed?\s*out", r"超时",             # 网络超时 / 工具自身超时
    r"connection", r"connect", r"连接",                # 连接断开、连不上
    r"temporarily", r"unavailable", r"不可用",          # 服务端临时不可用
    r"reset", r"broken", r"断开", r"reset\s*by\s*peer",  # 连接被重置
    r"502|503|504",                                    # 网关类 HTTP 状态
]
_TRANSIENT_RE = re.compile("|".join(_TRANSIENT_PATTERNS), re.IGNORECASE)


# 【什么】工具执行重试策略（指数退避 + 瞬时故障判定）
# 【作用】包住一次工具执行：失败且属于瞬时故障 → 等一下再试；其余直接返回原结果
class ToolRetryPolicy:
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

    def __init__(self, max_retries: int = None, backoff: float = None):
        s = get_settings()
        # max_retries 语义：首次执行之外【额外】再试几次（0 = 不重试）
        self._max_retries = max(0, int(max_retries if max_retries is not None
                                        else getattr(s, "tool_retry_attempts", 2)))
        self._backoff = max(0.0, float(backoff if backoff is not None
                                       else getattr(s, "tool_retry_backoff", 0.5)))

    # 【什么】判断一次失败是否值得重试
    # 【作用】error 文本命中瞬时故障清单 → True；参数/权限/不存在类错误 → False
    @staticmethod
    def is_transient(error: str) -> bool:
        """瞬时故障判定：只认超时 / 连接 / 暂不可用类错误。

        【判定例子】
            "Connection timeout"                    → True  （重试有意义）
            "MCP Server 不可用"                     → True
            "工具参数缺失: calculator 缺少必填参数"  → False （重试结果相同）
            "当前身份无权调用工具 data_query"        → False （权限问题，等模型换路）
            "工具不存在: calc"                      → False （幻觉工具名，等模型改名）
        """
        if not error:
            return False
        return bool(_TRANSIENT_RE.search(error))

    # 【什么】带重试地执行一次工具调用
    # 【作用】fn 返回 ToolResult；失败且瞬时 → 退避后重试；最多 1+max_retries 次
    def run(self, tool_name: str, fn: Callable):
        """执行 fn（一次工具调用），瞬时失败自动重试。

        Args:
            tool_name: 工具名，只用于日志定位。
            fn:        无参调用，返回 ToolResult（BaseTool.run 已保证不抛异常）。

        Returns:
            最后一次执行的 ToolResult（成功或最终失败），不抛异常。
        """
        result = fn()
        total = self._max_retries + 1
        for attempt in range(1, total + 1):
            if result.success:
                return result                      # 成功：不需要任何后续动作
            error = str(getattr(result, "error", "") or "")
            if not self.is_transient(error):
                _logger.info(f"工具 {tool_name} 失败且非瞬时故障，不重试: {error[:80]}")
                return result                      # 确定性失败：立刻回喂模型
            if attempt >= total:
                break                              # 重试额度用完，跳出循环返回最后结果
            wait = self._backoff * (2 ** (attempt - 1))    # 0.5 → 1.0 → 2.0（指数退避）
            _logger.warning(
                f"工具 {tool_name} 瞬时故障({attempt}/{self._max_retries})，"
                f"{wait:.1f}s 后重试: {error[:80]}"
            )
            time.sleep(wait)
            result = fn()
        return result
