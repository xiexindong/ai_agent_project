"""
敏感字段脱敏
============
按角色的 mask_level 对业务数据做字段级脱敏。

【为什么脱敏要放在"工具返回后、进入 LLM 上下文前"】
    LLM 上下文是会被摘要进长期记忆、被 RAG 引用、被审计日志记录的——
    一旦明文进了上下文，后面每一层都要单独防泄漏。源头脱敏最省事也最可靠。

【三级强度】
    0 不脱敏（财务经理看金额、HR 看身份证）
    1 部分脱敏：证件号/手机号/银行卡保留头尾
    2 强脱敏：金额类字段整体替换为区间描述，杜绝精确值外泄
"""
from typing import Any

# 字段名 → 脱敏类别
_FIELD_KIND = {
    "id_card": "id",
    "idcard": "id",
    "identity": "id",
    "phone": "phone",
    "mobile": "phone",
    "bank_account": "bank",
    "bankcard": "bank",
    "salary": "money",
    "amount": "money",
    "price": "money",
    "cost": "money",
}


def _mask_id(v: str) -> str:
    """证件号：前 6 后 4，中间星号。"""
    v = str(v)
    return v if len(v) <= 10 else f"{v[:6]}{'*' * (len(v) - 10)}{v[-4:]}"


def _mask_phone(v: str) -> str:
    v = str(v)
    return v if len(v) < 7 else f"{v[:3]}****{v[-4:]}"


def _mask_bank(v: str) -> str:
    v = str(v)
    return v if len(v) <= 4 else f"****{v[-4:]}"


def _mask_money(v: Any, strong: bool) -> str:
    """
    金额脱敏：弱脱敏保留数值但标注"仅限本人/本部门"无意义，故统一做区间化。
    strong=True 时只给区间，False 时保留两位小数。
    """
    try:
        num = float(v)
    except (TypeError, ValueError):
        return "***"
    if strong:
        if num >= 1_000_000:
            return "≥100万"
        if num >= 10_000:
            return f"{int(num // 10000)}万~{int(num // 10000) + 1}万"
        return "0~1万"
    return f"{num:.2f}"


def mask_value(field_name: str, value: Any, mask_level: int) -> Any:
    """按字段名与脱敏强度返回脱敏后的值；无需脱敏则原样返回。"""
    if mask_level <= 0 or value is None:
        return value
    kind = _FIELD_KIND.get(str(field_name).lower())
    if kind is None:
        return value
    if kind == "id":
        return _mask_id(value)
    if kind == "phone":
        return _mask_phone(value)
    if kind == "bank":
        return _mask_bank(value)
    if kind == "money":
        return _mask_money(value, strong=(mask_level >= 2))
    return value


def mask_record(record: Any, mask_level: int) -> Any:
    """递归脱敏 dict / list 中的敏感字段（业务数据常是嵌套结构）。"""
    if mask_level <= 0:
        return record
    if isinstance(record, dict):
        return {k: mask_value(k, mask_record(v, mask_level), mask_level) for k, v in record.items()}
    if isinstance(record, list):
        return [mask_record(item, mask_level) for item in record]
    return record
