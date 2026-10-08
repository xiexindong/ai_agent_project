"""
文本清洗
========
流水线第 1 步：把用户原始输入规整成「可以直接喂给 LLM 和 embedding」的文本。

【为什么必须第一个做】
    后面的风控正则、token 计数、意图识别都假设输入是干净的。
    如果带着零宽字符去匹配风控规则，"删 除 生 产 库" 就可能绕过 "删除生产库" 的规则 ——
    清洗在前，等于先把输入压成规则能认出的形态。

【清洗什么】
    ① 零宽/不可见字符（​ 等）：看不见但会打断正则匹配
    ② 控制字符（\x00-\x1f）：搞坏 prompt 结构
    ③ 全角字母数字 → 半角：让 "ＡＢＣ" 和 "ABC" 走同一条规则
    ④ 连续空白/空行：省 token，也让文本更规整
    ⑤ 重复标点："。。。。" → "。"
    ⑥ 超长截断：防止后面 token 计数被极端输入拖垮
"""
import re                                                       # 正则：各类字符清洗规则用
from typing import List                                         # 类型注解：removed 是字符串列表

from modules.common.logger import get_logger                    # 统一日志入口
from modules.pipeline.models import CleanResult                 # 本环节的结果模型

_logger = get_logger("pipeline.clean")                          # 本模块日志器，带 pipeline.clean 前缀

# ----------------------------------------------------------------------
# 【字符类】零宽与不可见字符：不占视觉宽度，但会让正则匹配失败
#   ​ 零宽空格  ‌-‏ 方向控制  ﻿ BOM  ‪-‮ 方向嵌入
# ----------------------------------------------------------------------
_ZERO_WIDTH_RE = re.compile(r"[​-‏﻿‪-‮­]")

# 控制字符（保留 \t \n \r，它们是合法排版字符）
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 连续空行（3 个以上换行压成 2 个）
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# 重复标点：。。。。 → 。
_REPEAT_PUNCT_RE = re.compile(r"([。！？!?，,；;]){3,}")

# 【什么】默认截断阈值：5000 字符
# 【作用】约 1 万 token，再长基本没有有效信息，且会拖垮后面的 token 计数
_DEFAULT_MAX_CHARS = 5000


# 【什么】全角字母数字与全角空格转半角
# 【作用】让 "ＡＢＣ" 与 "ABC" 走到同一条规则上，避免同义输入被当成两种字符
# 【为什么】全角字母数字在 Unicode 里是另一段码位，不转的话风控规则与工具名匹配会漏判
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
    out = []                                                # 逐字符转换结果容器
    for ch in text:                                         # 逐字符处理
        code = ord(ch)                                      # 取字符的 Unicode 码位
        if code == 0x3000:                                      # 全角空格
            out.append(" ")                                 # 转成半角空格
        elif (0xFF10 <= code <= 0xFF19                          # 全角数字 ０-９
              or 0xFF21 <= code <= 0xFF3A                        # 全角大写 Ａ-Ｚ
              or 0xFF41 <= code <= 0xFF5A):                      # 全角小写 ａ-ｚ
            out.append(chr(code - 0xFEE0))                  # 全角与半角码位固定相差 0xFEE0，减去即得半角
        else:                                               # 其余字符（中文、半角、中文标点等）
            out.append(ch)                                  # 原样保留
    return "".join(out)                                     # 拼回字符串返回


def clean_text(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> CleanResult:  # max_chars=截断上限字符数
    """清洗用户输入。

    Args:
        text: 用户原始输入（可能为 None / 空串 / 含控制字符）
        max_chars: 最大字符数，默认 5000（约 1 万 token，超过就没什么有效信息了）

    Returns:
        CleanResult：永远非 None；输入为空时返回 text="" 且 changed=False

    【输入输出对照】
        输入 "你​好\x00啊"          → text="你好啊", removed=["零宽字符x1","控制字符x1"]
        输入 "ＡＢＣ　１２３"             → text="ABC 123"
        输入 "注意。。。。。。"           → text="注意。"
        输入 "A" * 6000                  → text="A"*5000, removed 含 "超长截断6000->5000"
    """
    if not text:                                            # None 或空串
        return CleanResult(text="", original_length=0, cleaned_length=0, changed=False)  # 直接返回空结果

    original_length = len(text)                             # 记录清洗前字符数
    removed: List[str] = []                                 # 记录每类被清掉的内容，供日志溯源
    out = text                                              # 工作副本，逐步清洗

    # ① 零宽字符
    hits = len(_ZERO_WIDTH_RE.findall(out))                 # 先数命中个数（用于记录）
    if hits:                                                # 有命中才做替换
        out = _ZERO_WIDTH_RE.sub("", out)                   # 全部替换为空串
        removed.append(f"零宽字符x{hits}")                  # 记录"零宽字符xN"

    # ② 控制字符
    hits = len(_CONTROL_RE.findall(out))                    # 先数命中个数
    if hits:                                                # 有命中才做替换
        out = _CONTROL_RE.sub("", out)                      # 全部替换为空串
        removed.append(f"控制字符x{hits}")                  # 记录"控制字符xN"

    # ③ 全角转半角
    half = _to_halfwidth(out)                               # 全角字母数字与全角空格转半角
    if half != out:                                         # 有实际变化
        removed.append("全角转半角")                        # 记录
    out = half                                              # 更新工作副本

    # ④ 空白规整：每行内部多空格压成 1 个，空行最多保留 1 个
    norm = "\n".join(re.sub(r"[ \t　]+", " ", line).strip() for line in out.split("\n"))  # 逐行压空白并 strip
    norm = _BLANK_LINES_RE.sub("\n\n", norm).strip()        # 3 个以上换行压成 2 个，整体去首尾空白
    if norm != out:                                         # 有实际变化
        removed.append("空白规整")                          # 记录
    out = norm                                              # 更新工作副本

    # ⑤ 重复标点
    dedup = _REPEAT_PUNCT_RE.sub(r"\1", out)                # 3 个以上重复标点压成 1 个（\1 = 第一个标点）
    if dedup != out:                                        # 有实际变化
        removed.append("重复标点")                          # 记录
    out = dedup                                             # 更新工作副本

    # ⑥ 超长截断
    if len(out) > max_chars:                                # 超过字符上限
        removed.append(f"超长截断{len(out)}->{max_chars}")  # 记录截断前后长度
        out = out[:max_chars]                               # 硬切到上限

    changed = out != text                                   # 整个流程是否有改动
    if changed:                                             # 有改动才记日志
        _logger.info(f"文本清洗: {original_length} -> {len(out)} 字符 | {removed}")
    return CleanResult(
        text=out,                                           # 清洗后的文本
        original_length=original_length,                    # 清洗前字符数
        cleaned_length=len(out),                            # 清洗后字符数
        changed=changed,                                    # 是否改动过
        removed=removed,                                    # 被清掉的内容清单
    )
