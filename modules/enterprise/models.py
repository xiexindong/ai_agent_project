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

# 【什么】数据域优先级表（self 0 < dept 1 < all 2）
# 【作用】给 scope_for 比较大小用，数字越大可见范围越宽
_SCOPE_ORDER = {"self": 0, "dept": 1, "all": 2}


# 【什么】一次请求的权限快照（dataclass）
# 【作用】入口解析一次，之后 Agent 全链路只认这一个对象，不再逐层查库
@dataclass
class Principal:
    """一次请求的权限快照（不可变语义，构造后只读）。"""

    # 【字段说明】前 7 个是身份字段，后 6 个是权限字段；除 user_code 外都有默认值
    user_code: str                                        # 工号，唯一标识，如 U007
    real_name: str = ""                                   # 姓名
    tenant_code: str = ""                                 # 租户码，跨租户隔离依据
    dept_code: str = ""                                   # 部门码，dept 域过滤时用
    dept_name: str = ""                                   # 部门名，日志展示用
    job_title: str = ""                                   # 岗位名
    user_level: str = "normal"                            # 等级：normal/advanced/admin
    roles: list = field(default_factory=list)             # 角色码列表，默认空 list
    permissions: set = field(default_factory=set)         # 权限码集合，默认空 set，支持 * 前缀通配
    kb_codes: list = field(default_factory=list)          # 可见知识库编码；空 list 表示不做库过滤
    data_scope: str = "self"                              # 数据域：self/dept/all，默认最窄 self
    mask_level: int = 0                                   # 脱敏强度：0 不脱敏 / 1 部分 / 2 强
    source: str = "mysql"                                 # 来源：mysql=库里身份，memory=内存兜底


    # 【什么】单个权限码判定
    # 【作用】先看精确匹配，再看 * 前缀通配；权限集为空时一律拒绝（默认拒绝）
    def can(self, perm_code: str) -> bool:
        """
        判定是否拥有某权限，支持 `tool:mcp:*` 这类前缀通配。

        ✓ permissions={'tool:mcp:*'}      can('tool:mcp:web_search') → True
        ✓ permissions={'tool:calculator'} can('tool:calculator')     → True
        ✗ permissions={'tool:calculator'} can('tool:web_search')     → False
        ✗ permissions={}                  can('anything')            → False（默认拒绝）
        """
        if not perm_code or not self.permissions:
            return False                # 空权限码或空权限集 → 默认拒绝
        if perm_code in self.permissions:
            return True                 # 精确命中，如 'tool:calculator'
        for granted in self.permissions:
            if granted.endswith("*"):
                prefix = granted[:-1]   # 去掉末尾的 '*'，得到 'tool:mcp:'
                if perm_code.startswith(prefix):
                    return True         # 前缀通配命中，如 'tool:mcp:web_search'
        return False

    # 【什么】多权限码取"或"
    # 【作用】一个工具同时登记了普通码与 mcp 码时，任一命中就算有权限
    def can_any(self, perm_codes: Iterable[str]) -> bool:
        """任一命中即通过（工具同时登记了普通码与 mcp 码时用）。"""
        return any(self.can(code) for code in perm_codes)

    # 【什么】知识库可读性判定
    # 【作用】kb_codes 为空表示未开启隔离，一律放行；否则必须列在 kb_codes 里
    def can_kb(self, kb_code: str) -> bool:
        """知识库可读性判定；未开启隔离时 kb_codes 为空表示不限制。"""
        if not self.kb_codes:
            return True
        return kb_code in self.kb_codes

    # 【什么】确定某类业务数据的可见范围
    # 【作用】角色自带的 data_scope 与显式 data:xxx:all/dept 权限取最宽的那个
    def scope_for(self, data_type: str) -> str:
        """
        确定某类业务数据的可见范围：角色 data_scope 与显式 data:xxx:yyy 权限取最大。

        例：角色 data_scope='dept' + 权限 data:orders:all → orders 视为 all
        """
        scope = self.data_scope or "self"       # 兜底最窄的 self
        for cand in ("all", "dept"):            # 从宽到窄试：显式权限能放宽就放宽
            if self.can(f"data:{data_type}:{cand}"):
                if _SCOPE_ORDER[cand] > _SCOPE_ORDER.get(scope, 0):
                    scope = cand               # 只有当显式权限比现有范围更宽时才覆盖
        return scope


    # 【什么】把快照转成可 JSON 序列化的 dict
    # 【作用】接口返回与日志落库用；set 转排序 list，避免输出顺序不稳定
    def to_dict(self) -> dict:
        return {
            "user_code": self.user_code,
            "real_name": self.real_name,
            "tenant_code": self.tenant_code,
            "dept_code": self.dept_code,
            "dept_name": self.dept_name,
            "job_title": self.job_title,
            "user_level": self.user_level,
            "roles": list(self.roles),                  # 转 list 副本
            "permissions": sorted(self.permissions),    # set 排序后转 list，输出顺序稳定
            "kb_codes": list(self.kb_codes),            # 转 list 副本
            "data_scope": self.data_scope,
            "mask_level": self.mask_level,
            "source": self.source,
        }

    # 【什么】日志用的一句话摘要
    # 【作用】审计与日志里打印身份用；无角色时角色位显示"无角色"
    def brief(self) -> str:
        """日志用的一句话摘要：U002/李四/财务部/D_FINANCE[财务专员]"""
        return (
            f"{self.user_code}/{self.real_name}/{self.dept_name}"
            f"/{self.dept_code}[{'+'.join(self.roles) or '无角色'}]"
        )
