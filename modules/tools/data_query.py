"""
数据查询工具
=============
模拟企业内部数据查询工具（订单、用户、库存等）。
用于演示 Function Calling 工具调用与权限分级。
"""
from typing import Dict
from datetime import datetime
from modules.tools.base_tool import BaseTool, ToolResult

# 模拟内部数据库
_MOCK_DATA = {
    "users": {
        "U001": {"name": "张三", "dept": "技术部", "level": "高级", "join_date": "2022-03-15"},
        "U002": {"name": "李四", "dept": "市场部", "level": "中级", "join_date": "2023-07-01"},
        "U003": {"name": "王五", "dept": "财务部", "level": "初级", "join_date": "2024-01-10"},
    },
    "orders": {
        "O1001": {"customer": "张三", "amount": 1280.00, "status": "已完成", "date": "2026-08-01"},
        "O1002": {"customer": "李四", "amount": 3500.50, "status": "配送中", "date": "2026-09-05"},
        "O1003": {"customer": "王五", "amount": 899.00, "status": "待付款", "date": "2026-09-10"},
    },
    "inventory": {
        "SKU-001": {"name": "笔记本电脑", "stock": 45, "price": 5999.00},
        "SKU-002": {"name": "无线鼠标", "stock": 230, "price": 89.00},
        "SKU-003": {"name": "机械键盘", "stock": 0, "price": 299.00},
    },
}


class DataQueryTool(BaseTool):
    """
    企业内部数据查询工具（Mock 数据，演示用）。

    【三种查询场景的结果对比】
        ✓ 查单条存在的记录（data_type="users", record_id="U001"）：
            ToolResult(success=True, result={"id":"U001", "name":"张三", "dept":"技术部", ...})

        ✓ 查整类型列表（data_type="orders", record_id="" 留空）：
            ToolResult(success=True, result={"type":"orders", "total":3, "records":["O1001","O1002","O1003"]})
            → 不返回全部字段，只给 ID 清单，避免数据过多撑爆 LLM 上下文

        ✗ 查不存在的记录（data_type="users", record_id="U999"）：
            ToolResult(success=False, error="记录不存在: users/U999")
            → error 回喂给 LLM，模型知道"这个 ID 没数据"，会换 ID 再查或直接答用户

        ✗ 传错数据类型（data_type="products"）：
            ToolResult(success=False, error="不支持的数据类型: products，可选: ['users','orders','inventory']")
    """

    def __init__(self):
        super().__init__(
            name="data_query",
            description="查询企业内部业务数据（用户信息、订单状态、库存数据）。支持按类型和ID精确查询。",
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

    def execute(self, data_type: str, record_id: str = "", **kwargs) -> ToolResult:
        """执行数据查询。"""
        if data_type not in _MOCK_DATA:
            return ToolResult(
                success=False,
                error=f"不支持的数据类型: {data_type}，可选: {list(_MOCK_DATA.keys())}",
            )

        dataset = _MOCK_DATA[data_type]

        if not record_id:
            # 返回该类型全部记录（脱敏处理后）
            return ToolResult(success=True, result={
                "type": data_type,
                "total": len(dataset),
                "records": list(dataset.keys()),
            })

        if record_id not in dataset:
            return ToolResult(success=False, error=f"记录不存在: {data_type}/{record_id}")

        record = dataset[record_id]
        return ToolResult(success=True, result={"id": record_id, **record})
