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
from typing import Dict  # 类型注解：字典
from modules.tools.base_tool import BaseTool, ToolResult  # 工具基类与统一结果对象
from modules.enterprise.masking import mask_record  # 字段级脱敏工具：按 mask_level 给手机号/薪资打码

# 【什么】模拟业务数据集：users 4 条 / orders 3 条 / inventory 3 条
# 【作用】每条记录都带 owner 与 owner_dept，供行级数据域过滤使用
_MOCK_DATA = {  # 模块级 Mock 数据集：三层结构 {类型: {记录ID: 记录字段}}
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
    "inventory": {
        "SKU-001": {"name": "笔记本电脑", "stock": 45, "price": 5999.00, "owner": "", "owner_dept": ""},
        "SKU-002": {"name": "无线鼠标", "stock": 230, "price": 89.00, "owner": "", "owner_dept": ""},
        "SKU-003": {"name": "机械键盘", "stock": 0, "price": 299.00, "owner": "", "owner_dept": ""},
    },
}


# 【什么】企业内部数据查询工具（Mock 数据）
# 【作用】演示 Function Calling + 行级数据域（self/dept/all）过滤
class DataQueryTool(BaseTool):  # 数据查询工具：演示 Function Calling + 行级数据域过滤
    """
    企业内部数据查询工具（Mock 数据，演示用）。

    【能力声明】只读查询，但涉及 users / orders / inventory 三类业务数据，
    因此声明 data_scopes，由 Harness 的 ScopeMiddleware 校验
    参数里的 dept_code / user_code 是否超出当前身份的可见范围。

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

    # 【能力声明】只读查询；data_scopes 让 Harness 校验参数里的
    #   dept_code / user_code 是否越出当前身份的数据域（self / dept / all）
    capability = {  # 类属性：Harness 治理时读这张表
        "risk": "low",  # 风险等级：低（只读查询）
        "side_effect": False,  # 无副作用（不写库不发通知）
        "sandbox": "none",  # 不需要沙箱：纯内存查表
        "data_scopes": ("users", "orders", "inventory"),  # 涉及的三类业务数据域
        "description": "只读业务数据查询，按身份数据域过滤",  # 给人看的描述
    }

    # 【什么】构造函数：声明工具名、说明与参数 schema
    # 【作用】data_type 必填；record_id 选填，留空表示查可见列表
    def __init__(self):  # 构造：注册工具名/描述/参数 schema
        super().__init__(  # 调父类 BaseTool 构造
            name="data_query",   # 工具名；权限码对应 tool:data_query
            description="查询企业内部业务数据（用户信息、订单状态、库存数据）。支持按类型和ID精确查询。只能查询你权限范围内的数据。",  # 给 LLM 看的用途说明
            parameters={  # JSON Schema 形参说明书
                "properties": {  # 形参字段集
                    "data_type": {  # 入参 1：数据类型
                        "type": "string",  # 字符串类型
                        "enum": ["users", "orders", "inventory"],  # 枚举三选一
                        "description": "数据类型: users(用户), orders(订单), inventory(库存)",  # 用法说明
                    },   # 必填，三种取值：users / orders / inventory
                    "record_id": {  # 入参 2：记录ID
                        "type": "string",  # 字符串类型
                        "description": "记录ID，如 'U001', 'O1001', 'SKU-001'。留空返回该类型全部记录",  # 说明留空语义
                    },   # 选填，留空=返回该类型可见记录清单
                },
                "required": ["data_type"],   # 只强校验 data_type；record_id 留空是合法用法
            },
        )


    @staticmethod  # 静态方法：不依赖实例状态
    # 【什么】按数据域筛出可见记录
    # 【作用】self 只看本人、dept 只看本部门；owner_dept 为空的记录视为全公司共享
    def _visible_records(dataset: Dict[str, dict], scope: str, dept_code: str, user_code: str) -> Dict[str, dict]:  # 行级数据域过滤：返回可见记录子集
        """按数据域过滤可见记录；owner_dept 为空的记录视为全公司共享。"""
        visible = {}  # 收集可见记录
        for rid, rec in dataset.items():  # 遍历数据集每条记录（rid=记录ID, rec=记录字段）
            owner = rec.get("owner", "")  # 取记录的归属人工号
            owner_dept = rec.get("owner_dept", "")  # 取记录的归属部门码
            if scope == "self" and owner and owner != user_code:  # self 域：有归属人且不是本人
                continue  # 跳过，本人看不到
            if scope == "dept" and owner_dept and owner_dept != dept_code:  # dept 域：有归属部门且不是本部门
                continue  # 跳过，跨部门看不到
            visible[rid] = rec  # 通过过滤：放进可见集合
        return visible  # 返回可见记录子集

    # 【什么】真正执行查询的三分支：列表 / 单条 / 失败
    # 【作用】私有参数（_data_scope 等）由服务端注入，模型不可见也不可传
    def execute(self, data_type: str, record_id: str = "", **kwargs) -> ToolResult:  # 业务入口；**kwargs 接服务端注入的私有参数
        """
        执行数据查询。

        kwargs 里的私有参数（由 Agent 按登录用户注入，模型不可见、不可传）：
            _data_scope  self / dept / all
            _dept_code   当前用户部门码
            _user_code   当前用户工号
            _mask_level  脱敏强度 0/1/2
        """
        if data_type not in _MOCK_DATA:  # 数据类型不在三类之内
            return ToolResult(  # 回失败结果
                success=False,  # 标记失败
                error=f"不支持的数据类型: {data_type}，可选: {list(_MOCK_DATA.keys())}",  # 报错并列出合法类型
            )

        scopes = kwargs.get("_scopes") or {}                                      # 按数据类型细分的数据域字典（由 Agent 侧注入，见 react_agent._server_params，来源 principal.scope_for），未传则为空 dict
        scope = scopes.get(data_type) or kwargs.get("_data_scope", "all") or "all"  # 取值优先级：_scopes[data_type] > _data_scope > "all"
        dept_code = kwargs.get("_dept_code", "") or ""                            # 当前用户部门码，dept 域过滤用
        user_code = kwargs.get("_user_code", "") or ""                            # 当前用户工号，self 域过滤用
        try:  # mask_level 可能是字符串或非法值，用 try 包住
            mask_level = int(kwargs.get("_mask_level", 0) or 0)   # 脱敏强度 0/1/2，未注入时按 0（不脱敏）
        except (TypeError, ValueError):  # 类型错误或值错误（如 "abc"）
            mask_level = 0                                        # 非法取值兜底为不脱敏，不让取值问题中断查询

        dataset = _MOCK_DATA[data_type]  # 取出该类型的整份数据集
        visible = self._visible_records(dataset, scope, dept_code, user_code)  # 按数据域过滤出可见记录

        if not record_id:  # record_id 留空：查列表分支
            return ToolResult(success=True, result={  # 回成功结果
                "type": data_type,  # 数据类型回显
                "total": len(visible),  # 可见记录数
                "records": list(visible.keys()),  # 只给 ID 清单（不返回完整记录，避免撑爆 LLM 上下文）
                "scope": scope,  # 当前生效的数据域，便于上层观测
            })

        if record_id not in visible:  # 单条查询但不在可见集合里
            if record_id in dataset:  # 记录存在但当前身份看不到
                return ToolResult(  # 回"无权访问"错误（区别于"不存在"）
                    success=False,  # 标记失败
                    error=f"无权访问该记录: {data_type}/{record_id}（你的数据范围: {scope}）",  # 告诉模型是权限问题
                )
            return ToolResult(success=False, error=f"记录不存在: {data_type}/{record_id}")  # 记录真不存在：模型可换 ID 再查

        record = visible[record_id]  # 取出目标记录
        safe_record = mask_record(record, mask_level)  # 按脱敏强度给敏感字段打码
        return ToolResult(success=True, result={"id": record_id, **safe_record})  # 把 ID 和脱敏后的记录合并返回
