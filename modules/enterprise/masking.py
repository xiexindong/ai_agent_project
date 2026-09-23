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

# 【什么】字段名 → 敏感类别的映射表（11 个键，4 类）
# 【作用】按字段名判断该怎么脱敏；查不到的字段一律原样返回，不做猜测
_FIELD_KIND = {
    "id_card": "id",          # 身份证
    "idcard": "id",           # 身份证（简写）
    "identity": "id",         # 证件号
    "phone": "phone",         # 电话
    "mobile": "phone",        # 手机号
    "bank_account": "bank",   # 银行账号
    "bankcard": "bank",       # 银行卡号
    "salary": "money",        # 薪资
    "amount": "money",        # 金额
    "price": "money",         # 单价
    "cost": "money",          # 成本
}


# 【什么】证件号脱敏：留前 6 后 4
# 【作用】中间全部打星，长度 ≤ 10 的原样返回（遮完没有保留价值）
def _mask_id(v: str) -> str:
    """证件号：前 6 后 4，中间星号。"""
    v = str(v)                # 统一转字符串，数据库里可能是数字型
    return v if len(v) <= 10 else f"{v[:6]}{'*' * (len(v) - 10)}{v[-4:]}"


# 【什么】手机号脱敏：留前 3 后 4
# 【作用】中间固定 4 个星号；长度 < 7 的原样返回
def _mask_phone(v: str) -> str:
    v = str(v)
    return v if len(v) < 7 else f"{v[:3]}****{v[-4:]}"


# 【什么】银行卡脱敏：只留后 4 位
# 【作用】前面全星号；长度 ≤ 4 的原样返回（全遮等于没信息）
def _mask_bank(v: str) -> str:
    v = str(v)
    return v if len(v) <= 4 else f"****{v[-4:]}"


# 【什么】金额脱敏：区间化或保留两位小数
# 【作用】strong=True 只给区间杜绝精确值外泄，False 保留两位小数
def _mask_money(v: Any, strong: bool) -> str:
    """
    金额脱敏：弱脱敏保留数值但标注"仅限本人/本部门"无意义，故统一做区间化。
    strong=True 时只给区间，False 时保留两位小数。
    """
    try:
        num = float(v)        # 转 float；金额可能是 Decimal / str / int
    except (TypeError, ValueError):
        return "***"          # 转不了的数字一律三颗星，不抛异常
    if strong:
        if num >= 1_000_000:      # ≥100 万：只说下限，不给上限
            return "≥100万"
        if num >= 10_000:         # 1 万 ~ 100 万：给整万区间
            return f"{int(num // 10000)}万~{int(num // 10000) + 1}万"
        return "0~1万"            # < 1 万：区间 0~1 万
    return f"{num:.2f}"           # 弱脱敏：保留两位小数的精确值


# 【什么】单个字段值的脱敏分派
# 【作用】按字段名查类别表，再交给对应的 _mask_xxx；不在表里的字段原样返回
def mask_value(field_name: str, value: Any, mask_level: int) -> Any:
    """按字段名与脱敏强度返回脱敏后的值；无需脱敏则原样返回。"""
    if mask_level <= 0 or value is None:
        return value                                            # 0 级不脱敏；None 也没得脱
    kind = _FIELD_KIND.get(str(field_name).lower())             # 字段名转小写后查表；None = 非敏感字段
    if kind is None:
        return value                                            # 查不到 → 原样返回
    if kind == "id":
        return _mask_id(value)                                  # 证件号：前 6 后 4
    if kind == "phone":
        return _mask_phone(value)                               # 手机号：前 3 后 4
    if kind == "bank":
        return _mask_bank(value)                                # 银行卡：只留后 4
    if kind == "money":
        return _mask_money(value, strong=(mask_level >= 2))     # 金额：2 级及以上才算强脱敏
    return value


# 【什么】递归脱敏一整条记录
# 【作用】业务数据常是嵌套 dict/list，逐层下钻，让深层字段也按名脱敏
def mask_record(record: Any, mask_level: int) -> Any:
    """递归脱敏 dict / list 中的敏感字段（业务数据常是嵌套结构）。"""
    if mask_level <= 0:
        return record                  # 0 级：整条不脱敏，直接返回
    if isinstance(record, dict):
        # 【dict】先递归脱敏每个 value，再按 key（字段名）脱敏一次
        return {k: mask_value(k, mask_record(v, mask_level), mask_level) for k, v in record.items()}
    if isinstance(record, list):
        return [mask_record(item, mask_level) for item in record]   # 【list】逐元素递归
    return record                      # 标量：字段名不在这里，无需处理
