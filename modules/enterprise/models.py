"""
权限载体模型
============
Principal（主体）= 一次请求里"这个用户到底是谁、能干啥"的完整快照。

【为什么要有这么个对象，而不是到处传 user_id】
    传 user_id 意味着每个环节都要回头查一次库：agent 查一次工具权限、RAG 查一次
    知识库权限、data_query 查一次数据域——一次对话十几次查询，还得各处处理"查不到"
    的边界。Principal 在入口解析一次，后面全是内存判定，且天然可测试：
    单元测试里构造一个 Principal 就能验证任意权限分支。

【权限码与通配】
    tool:calculator   精确匹配
    tool:mcp:*        前缀通配，命中所有 tool:mcp:xxx
    api:admin:*       同理
"""
from dataclasses import dataclass, field
from typing import Iterable

# 数据域：可见范围由小到大，多角色冲突时取最大
_SCOPE_ORDER = {"self": 0, "dept": 1, "all": 2}


@dataclass
class Principal:
    """一次请求的权限快照（不可变语义，构造后只读）。"""

    user_code: str                      # 工号，等价于旧代码的 user_id
    real_name: str = ""
    tenant_code: str = ""
    dept_code: str = ""
    dept_name: str = ""
    job_title: str = ""
    user_level: str = "normal"          # normal/advanced/admin，兼容旧一维分级
    roles: list = field(default_factory=list)
    permissions: set = field(default_factory=set)
    kb_codes: list = field(default_factory=list)     # 可见知识库 kb_code 列表
    data_scope: str = "self"            # self / dept / all
    mask_level: int = 0                 # 0 不脱敏 1 部分 2 强脱敏
    source: str = "mysql"               # mysql / memory（降级来源，排障用）

    # ==================== 权限判定 ====================

    def can(self, perm_code: str) -> bool:
        """
        判定是否拥有某权限，支持 `tool:mcp:*` 这类前缀通配。

        ✓ permissions={'tool:mcp:*'}      can('tool:mcp:web_search') → True
        ✓ permissions={'tool:calculator'} can('tool:calculator')     → True
        ✗ permissions={'tool:calculator'} can('tool:web_search')     → False
        ✗ permissions={}                  can('anything')            → False（默认拒绝）
        """
        if not perm_code or not self.permissions:
            return False
        if perm_code in self.permissions:
            return True
        for granted in self.permissions:
            if granted.endswith("*"):
                prefix = granted[:-1]                   # 'tool:mcp:*' → 'tool:mcp:'
                if perm_code.startswith(prefix):
                    return True
        return False

    def can_any(self, perm_codes: Iterable[str]) -> bool:
        """任一命中即通过（工具同时登记了普通码与 mcp 码时用）。"""
        return any(self.can(code) for code in perm_codes)

    def can_kb(self, kb_code: str) -> bool:
        """知识库可读性判定；未开启隔离时 kb_codes 为空表示不限制。"""
        if not self.kb_codes:
            return True
        return kb_code in self.kb_codes

    def scope_for(self, data_type: str) -> str:
        """
        确定某类业务数据的可见范围：角色 data_scope 与显式 data:xxx:yyy 权限取最大。

        例：角色 data_scope='dept' + 权限 data:orders:all → orders 视为 all
        """
        scope = self.data_scope or "self"
        for cand in ("all", "dept"):
            if self.can(f"data:{data_type}:{cand}"):
                if _SCOPE_ORDER[cand] > _SCOPE_ORDER.get(scope, 0):
                    scope = cand
        return scope

    # ==================== 展示与日志 ====================

    def to_dict(self) -> dict:
        return {
            "user_code": self.user_code,
            "real_name": self.real_name,
            "tenant_code": self.tenant_code,
            "dept_code": self.dept_code,
            "dept_name": self.dept_name,
            "job_title": self.job_title,
            "user_level": self.user_level,
            "roles": list(self.roles),
            "permissions": sorted(self.permissions),
            "kb_codes": list(self.kb_codes),
            "data_scope": self.data_scope,
            "mask_level": self.mask_level,
            "source": self.source,
        }

    def brief(self) -> str:
        """日志用的一句话摘要：U002/李四/财务部/D_FINANCE[财务专员]"""
        return (
            f"{self.user_code}/{self.real_name}/{self.dept_name}"
            f"/{self.dept_code}[{'+'.join(self.roles) or '无角色'}]"
        )
