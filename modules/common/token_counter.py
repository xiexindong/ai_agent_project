"""
Token 计数模块
==============
基于 tiktoken 实现 token 统计，支持阈值监控、上下文溢出预警。

【token 是什么？】LLM 不按"字符"也不按"单词"计费，按"token"计费：
    1 个中文字 ≈ 1~2 token
    1 个英文单词 ≈ 1.3 token（"hello"=1 token，"fantastic"可能拆成 2~3 token）
    DeepSeek 一次请求最多 ~64K token，超了报错。

【为什么需要这个模块？】
    - 输入超长时撑爆上下文，必须先 count 再决定要不要压缩（边界4）
    - 计费 / 限流 / 溢出预警都依赖准确的 token 数
"""
import tiktoken
from modules.common.logger import get_logger

_logger = get_logger("token")
_enc = None


def _get_encoder():
    """获取 tiktoken 编码器（懒加载，DeepSeek 兼容 cl100k_base）。"""
    global _enc
    if _enc is None:
        try:
            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _logger.warning("tiktoken cl100k_base 不可用，降级为字符估算")
            _enc = "fallback"
    return _enc


def count_tokens(text: str) -> int:
    """
    精确统计文本 token 数（tiktoken 算不准时降级为估算）。

    【输入输出对比】
        "hello"              → 1
        "你好"               → 2
        "3*7 等于多少"        → 7
        ""                   → 0
        10万字长文本          → ~20000+（每次只算一次，不重复加载编码器）

    降级路径：tiktoken 包未装 / 编码失败 → 调 estimate_tokens 粗略估算
    """
    if not text:
        return 0
    enc = _get_encoder()
    if enc == "fallback":
        return estimate_tokens(text)
    try:
        return len(enc.encode(text))
    except Exception:
        return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """
    粗略估算 token 数（不依赖 tiktoken，纯字符数公式）。

    【公式】中文字 * 2 + 其他字符 * 1.3
        "你好"   → 2*2 = 4
        "hello" → 5*1.3 = 6（粗略，实际可能是 1）
        "abc你好" → 3*1.3 + 2*2 = 7.9 → 7

    用途：tiktoken 不可用时兜底，精度差但保证不崩。
    """
    if not text:
        return 0
    chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other = len(text) - chinese
    return int(chinese * 2 + other * 1.3)


def check_token_overflow(text: str, threshold: int) -> bool:
    """
    检查文本 token 是否超过阈值（边界4 触发条件）。

    【典型用法】在 MemoryManager.build_context 里：
        if check_token_overflow(context_text, max_history_tokens=4000):
            # 触发压缩：合并久远摘要减少 token

    【输入输出对比】
        check_token_overflow("你好", 4000)   → False（2 < 4000，不溢出）
        check_token_overflow(长文本, 4000)   → True  （超阈值，触发压缩）
        check_token_overflow("", 4000)      → False（空文本不溢出）
    """
    tok = count_tokens(text)
    overflow = tok > threshold
    if overflow:
        _logger.warning(f"Token溢出: 当前{tok} > 阈值{threshold}")
    return overflow
