"""
文本清洗
========
流水线第 1 步：把用户原始输入规整成「可以直接喂给 LLM 和 embedding」的文本。

【为什么必须第一个做】
    后面的风控正则、token 计数、意图识别都假设输入是干净的。
    如果带着零宽字符去匹配风控规则，"删 除 生 产 库" 就可能绕过 "删除生产库" 的规则 ——
    清洗在前，等于先把输入压成规则能认出的形态。

【清洗什么】
    ① 零宽/不可见字符（\u200b 等）：看不见但会打断正则匹配
    ② 控制字符（\x00-\x1f）：搞坏 prompt 结构
    ③ 全角字母数字 → 半角：让 "ＡＢＣ" 和 "ABC" 走同一条规则
    ④ 连续空白/空行：省 token，也让文本更规整
    ⑤ 重复标点："。。。。" → "。"
    ⑥ 超长截断：防止后面 token 计数被极端输入拖垮
"""
import re
from typing import List

from modules.common.logger import get_logger
from modules.pipeline.models import CleanResult

_logger = get_logger("pipeline.clean")

# ----------------------------------------------------------------------
# 【字符类】零宽与不可见字符：不占视觉宽度，但会让正则匹配失败
#   \u200b 零宽空格  \u200c-\u200f 方向控制  \ufeff BOM  \u202a-\u202e 方向嵌入
# ----------------------------------------------------------------------
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\ufeff\u202a-\u202e\u00ad]")

# 控制字符（保留 \t \n \r，它们是合法排版字符）
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 连续空行（3 个以上换行压成 2 个）
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# 重复标点：。。。。 → 。
_REPEAT_PUNCT_RE = re.compile(r"([。！？!?，,；;]){3,}")

_DEFAULT_MAX_CHARS = 5000


def _to_halfwidth(text: str) -> str:
    """全角字母、数字与全角空格转半角。

    【输入输出对照】
        输入 "ＡＢＣ１２３　xyz，你好"  → 输出 "ABC123 xyz，你好"
            （全角逗号「，」保持原样，中文排版不变）

    【为什么只转字母数字与空格，不转中文标点】
        全角标点（，。！？、）是中文排版的一部分，转成半角会破坏阅读习惯，
        而且它们不参与任何风控规则或工具名匹配 —— 转了没收益、只添乱。
        真正需要统一的是「ＡＢＣ１２３」这类会被当成不同字符的字母数字。
    """
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:                                      # 全角空格
            out.append(" ")
        elif (0xFF10 <= code <= 0xFF19                          # 全角数字 ０-９
              or 0xFF21 <= code <= 0xFF3A                        # 全角大写 Ａ-Ｚ
              or 0xFF41 <= code <= 0xFF5A):                      # 全角小写 ａ-ｚ
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def clean_text(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> CleanResult:
    """清洗用户输入。

    Args:
        text: 用户原始输入（可能为 None / 空串 / 含控制字符）
        max_chars: 最大字符数，默认 5000（约 1 万 token，超过就没什么有效信息了）

    Returns:
        CleanResult：永远非 None；输入为空时返回 text="" 且 changed=False

    【输入输出对照】
        输入 "你\u200b好\x00啊"          → text="你好啊", removed=["零宽字符x1","控制字符x1"]
        输入 "ＡＢＣ　１２３"             → text="ABC 123"
        输入 "注意。。。。。。"           → text="注意。"
        输入 "A" * 6000                  → text="A"*5000, removed 含 "超长截断6000->5000"
    """
    if not text:
        return CleanResult(text="", original_length=0, cleaned_length=0, changed=False)

    original_length = len(text)
    removed: List[str] = []
    out = text

    # ① 零宽字符
    hits = len(_ZERO_WIDTH_RE.findall(out))
    if hits:
        out = _ZERO_WIDTH_RE.sub("", out)
        removed.append(f"零宽字符x{hits}")

    # ② 控制字符
    hits = len(_CONTROL_RE.findall(out))
    if hits:
        out = _CONTROL_RE.sub("", out)
        removed.append(f"控制字符x{hits}")

    # ③ 全角转半角
    half = _to_halfwidth(out)
    if half != out:
        removed.append("全角转半角")
    out = half

    # ④ 空白规整：每行内部多空格压成 1 个，空行最多保留 1 个
    norm = "\n".join(re.sub(r"[ \t\u3000]+", " ", line).strip() for line in out.split("\n"))
    norm = _BLANK_LINES_RE.sub("\n\n", norm).strip()
    if norm != out:
        removed.append("空白规整")
    out = norm

    # ⑤ 重复标点
    dedup = _REPEAT_PUNCT_RE.sub(r"\1", out)
    if dedup != out:
        removed.append("重复标点")
    out = dedup

    # ⑥ 超长截断
    if len(out) > max_chars:
        removed.append(f"超长截断{len(out)}->{max_chars}")
        out = out[:max_chars]

    changed = out != text
    if changed:
        _logger.info(f"文本清洗: {original_length} -> {len(out)} 字符 | {removed}")
    return CleanResult(
        text=out,
        original_length=original_length,
        cleaned_length=len(out),
        changed=changed,
        removed=removed,
    )
