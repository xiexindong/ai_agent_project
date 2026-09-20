"""
数据查询工具
=============
模拟企业内部数据查询工具（订单、用户、库存等），用于演示 Function Calling 与**行级权限**。

【企业级改造：行级数据域（Row-Level Scope）】
    同一句"查一下订单"，不同人必须看到不同结果：
        scope=self   普通员工 → 只看到 owner 是自己的记录
        scope=dept   部门经理 → 只看到 owner_dept 是本部门的记录
        scope=all    管理员/审计 → 全公司
    scope 由 RBAC 角色决定，通过私有参数注入，**不出现在 Function Calling schema 里**
    （一旦出现在 schema，模型就会尝试自己传参，等于把权限交给模型决定）。

【为什么"记录不存在"要分两种错】
    无权访问  → "无权访问该记录（数据范围: dept）"，让模型立刻停止尝试
    真不存在  → "记录不存在"，模型可以换个 ID 再查
    混为一谈会导致模型反复重试探测，既浪费轮次又给攻击者一个枚举探测通道。
"""
from typing import Dict
from modules.tools.base_tool import BaseTool, ToolResult
from modules.enterprise.masking import mask_record

# 模拟内部数据库（owner / owner_dept 是行级权限的判定依据）
_MOCK_DATA = {
    "users": {
        "U001": {"name": "张三", "dept": "技术部", "dept_code": "D_TECH", "level": "高级",
                 "join_date": "2022-03-15", "salary": 25000, "phone": "13800000001",
                 "owner": "U001", "owner_dept": "D_TECH"},
        "U002": {"name": "李四", "dept": "财务部", "dept_code": "D_FINANCE", "level": "中级",
                 "join_date": "2023-07-01", "salary": 18000, "phone": "13800000002",
                 "owner": "U002", "owner_dept": "D_FINANCE"},
        "U003": {"name": "王五", "dept": "财务部", "dept_code": "D_FINANCE", "level": "初级",
                 "join_date": "2024-01-10", "salary": 12000, "phone": "13800000003",
                 "owner": "U003", "owner_dept": "D_FINANCE"},
        "U004": {"name": "赵六", "dept": "人力资源部", "dept_code": "D_HR", "level": "中级",
                 "join_date": "2021-05-20", "salary": 16000, "phone": "13800000004",
                 "owner": "U004", "owner_dept": "D_HR"},
    },
    "orders": {
        "O1001": {"customer": "张三", "amount": 1280.00, "status": "已完成", "date": "2026-08-01",
                  "owner": "U001", "owner_dept": "D_TECH"},
        "O1002": {"customer": "李四", "amount": 3500.50, "status": "配送中", "date": "2026-09-05",
                  "owner": "U002", "owner_dept": "D_FINANCE"},
        "O1003": {"customer": "王五", "amount": 899.00, "status": "待付款", "date": "2026-09-10",
                  "owner": "U003", "owner_dept": "D_FINANCE"},
    },
    # 库存是全公司共享数据，owner_dept 置空 = 不受部门域限制
    "inventory": {
        "SKU-001": {"name": "笔记本电脑", "stock": 45, "price": 5999.00, "owner": "", "owner_dept": ""},
        "SKU-002": {"name": "无线鼠标", "stock": 230, "price": 89.00, "owner": "", "owner_dept": ""},
        "SKU-003": {"name": "机械键盘", "stock": 0, "price": 299.00, "owner": "", "owner_dept": ""},
    },
}


class DataQueryTool(BaseTool):
    """
    企业内部数据查询工具（Mock 数据，演示用）。

    【三种查询场景的结果对比】
        ✓ 查单条存在的记录（data_type="users", record_id="U001"）：
            ToolResult(success=True, result={"id":"U001", "name":"张三", "dept":"技术部", ...})

        ✓ 查整类型列表（data_type="orders", record_id="" 留空）：
            ToolResult(success=True, result={"type":"orders", "total":3, "records":["O1001",...], "scope":"dept"})
            → 只给 ID 清单，避免数据过多撑爆 LLM 上下文

        ✗ 查不存在的记录（data_type="users", record_id="U999"）：
            ToolResult(success=False, error="记录不存在: users/U999")
            → 模型知道"这个 ID 没数据"，会换 ID 再查或直接答用户

        ✗ 越权访问（普通员工查别人的档案）：
            ToolResult(success=False, error="无权访问该记录: users/U002（数据范围: self）")
            → 与"不存在"区分开，模型不会继续探测
    """

    def __init__(self):
        super().__init__(
            name="data_query",
            description="查询企业内部业务数据（用户信息、订单状态、库存数据）。支持按类型和ID精确查询。只能查询你权限范围内的数据。",
            parameters={
                "properties": {
                    "data_type": {
                        "type": "string",
                        "enum": ["users", "orders", "inventory"],
                        "description": "数据类型: users(用户), orders(订单), inventory(库存)",
                    },
                    "record_id": {
                        "type": "string",
                        "description": "记录ID，如 'U001', 'O1001', 'SKU-001'。留空返回该类型全部记录",
                    },
                },
                "required": ["data_type"],
            },
        )

    # ==================== 行级权限 ====================

    @staticmethod
    def _visible_records(dataset: Dict[str, dict], scope: str, dept_code: str, user_code: str) -> Dict[str, dict]:
        """按数据域过滤可见记录；owner_dept 为空的记录视为全公司共享。"""
        visible = {}
        for rid, rec in dataset.items():
            owner = rec.get("owner", "")
            owner_dept = rec.get("owner_dept", "")
            if scope == "self" and owner and owner != user_code:
                continue
            if scope == "dept" and owner_dept and owner_dept != dept_code:
                continue
            visible[rid] = rec
        return visible

    def execute(self, data_type: str, record_id: str = "", **kwargs) -> ToolResult:
        """
        执行数据查询。

        kwargs 里的私有参数（由 Agent 按登录用户注入，模型不可见、不可传）：
            _data_scope  self / dept / all
            _dept_code   当前用户部门码
            _user_code   当前用户工号
            _mask_level  脱敏强度 0/1/2
        """
        if data_type not in _MOCK_DATA:
            return ToolResult(
                success=False,
                error=f"不支持的数据类型: {data_type}，可选: {list(_MOCK_DATA.keys())}",
            )

        # 数据域按数据类型取（_scopes 由服务端按 RBAC 注入：{"users":"dept", ...}）
        scopes = kwargs.get("_scopes") or {}
        scope = scopes.get(data_type) or kwargs.get("_data_scope", "all") or "all"
        dept_code = kwargs.get("_dept_code", "") or ""
        user_code = kwargs.get("_user_code", "") or ""
        try:
            mask_level = int(kwargs.get("_mask_level", 0) or 0)
        except (TypeError, ValueError):
            mask_level = 0

        dataset = _MOCK_DATA[data_type]
        visible = self._visible_records(dataset, scope, dept_code, user_code)

        if not record_id:
            # 列表模式：只回 ID 清单，避免全量字段撑爆上下文
            return ToolResult(success=True, result={
                "type": data_type,
                "total": len(visible),
                "records": list(visible.keys()),
                "scope": scope,
            })

        if record_id not in visible:
            if record_id in dataset:
                # 存在但越权：明确告知"无权"，阻止模型继续探测
                return ToolResult(
                    success=False,
                    error=f"无权访问该记录: {data_type}/{record_id}（你的数据范围: {scope}）",
                )
            return ToolResult(success=False, error=f"记录不存在: {data_type}/{record_id}")

        record = visible[record_id]
        # 字段级脱敏：明文不进 LLM 上下文（会进摘要、进记忆、进审计）
        safe_record = mask_record(record, mask_level)
        return ToolResult(success=True, result={"id": record_id, **safe_record})
