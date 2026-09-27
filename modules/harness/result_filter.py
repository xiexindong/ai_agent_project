"""
工具结果过滤
============
工具返回什么，不直接等于模型看到什么。

【为什么要过滤】
    1. 防上下文膨胀：一个 data_query 可能返回几千字，全部回灌进下一轮 prompt，
       几轮就把上下文窗口吃满（这就是"上下文膨胀"的主因之一）。
    2. 防越权泄漏：工具按自己的权限取回了完整记录（含手机号、薪资），
       但当前身份的 mask_level 是 2，模型不该看到明文。
    3. 防噪声干扰：超长结果里真正有用的往往只有前几行。

【过滤三步】
    脱敏（按 principal.mask_level） → 裁剪（按 max_result_chars） → 标注（meta 里写清楚被改过）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from modules.common.logger import get_logger
from modules.tools.base_tool import ToolResult

_logger = get_logger("harness.result_filter")

# 裁剪标记：模型看到它就知道"后面还有，但没给我"，不会误以为结果只有这么点
_TRUNCATED_HINT = "…(结果已裁剪，共 {total} 字符，已保留前 {kept} 字符)"


@dataclass
class ResultFilterConfig:
    max_result_chars: int = 2000        # 成功结果最大字符数
    max_error_chars: int = 500          # 错误信息最大字符数
    mask_enabled: bool = True           # 是否按 principal.mask_level 脱敏
    sensitive_words: tuple = ()         # 额外要打码的词


# 【什么】结果过滤器
# 【作用】把工具返回值变成"可以安全回灌上下文"的形态
class ResultFilter:
    """工具结果裁剪 + 脱敏。

    【输入输出对照】
        f = ResultFilter(max_result_chars=20)
        f.filter(ToolResult(True, result="abcdefghijklmnopqrstuvwxyz"))
        → ToolResult(True, result="abcdefghijklmnopqrst…(结果已裁剪，共 26 字符，已保留前 20 字符)")

        f.filter(ToolResult(True, result={"phone": "13812345678"}), principal=<mask_level=2>)
        → result 里 phone 被打码（复用 modules.enterprise.masking.mask_record）
    """

    def __init__(self, config: Optional[ResultFilterConfig] = None):
        self.config = config or ResultFilterConfig()

    # 【什么】过滤单个结果
    # 【作用】失败结果只裁错误信息；成功结果先脱敏再裁剪
    def filter(self, result: ToolResult, principal: Any = None,
               max_chars: int = 0) -> ToolResult:
        if result is None:
            return result
        limit = int(max_chars or self.config.max_result_chars or 0)
        meta = dict(getattr(result, "meta", {}) or {})

        if not result.success:
            error = str(result.error or "")
            if limit and len(error) > self.config.max_error_chars:
                result.error = error[: self.config.max_error_chars] + "…(错误信息已裁剪)"
                meta["error_trimmed"] = True
            result.meta = meta
            return result

        payload = result.result
        masked = False
        if self.config.mask_enabled and principal is not None:
            level = int(getattr(principal, "mask_level", 0) or 0)
            if level > 0 and isinstance(payload, (dict, list)):
                payload = self._mask(payload, level)
                masked = True
                meta["mask_level"] = level
        text = self._stringify(payload)
        trimmed = False
        if limit and len(text) > limit:
            text = text[:limit] + _TRUNCATED_HINT.format(total=len(text), kept=limit)
            trimmed = True
        result.result = text
        meta["filtered"] = True
        meta["trimmed"] = trimmed
        meta["masked"] = masked
        meta["chars"] = len(text)
        result.meta = meta
        return result

    # 【什么】按字段脱敏
    # 【作用】复用企业模块的脱敏规则，不另造一套口径
    @staticmethod
    def _mask(payload: Any, level: int) -> Any:
        try:
            from modules.enterprise.masking import mask_record
            return mask_record(payload, level)
        except Exception as e:
            _logger.warning(f"结果脱敏失败（按不脱敏处理）: {e}")
            return payload

    # 【什么】把任意结果转成字符串
    # 【作用】回灌上下文时统一是文本；dict/list 用 JSON 保留结构
    @staticmethod
    def _stringify(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (dict, list)):
            import json
            try:
                return json.dumps(payload, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(payload)
        return str(payload)
