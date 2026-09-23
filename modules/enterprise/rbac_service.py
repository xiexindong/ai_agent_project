"""
RBAC 权限服务
=============
职责：把一条 API Key 解析成 Principal（身份 + 角色 + 权限 + 数据域 + 知识库可见范围）。

【一次解析查 4 次库，之后全内存】
    ① sys_api_key → sys_user → 身份与在职状态
    ② sys_user_role → sys_role → 数据域与脱敏强度
    ③ sys_role_permission → sys_permission → 权限码集合
    ④ sys_kb / sys_kb_grant → 可见知识库列表
    结果按 api_key 摘要缓存 RBAC_CACHE_TTL 秒（默认 60s），改权限最多延迟 60 秒生效。

【缓存 key 为什么用摘要而不是明文】
    明文 API Key 不进内存快照、不进日志、不进 trace。这里缓存键取 sha256 前 16 位，
    出问题时凭摘要到库里反查是哪把 Key，而不是让明文在进程里到处留痕。

【降级策略（fail-closed）】
    RBAC_ENABLED=False（显式关闭权限）时走内存鉴权，方便本地演示。
    已启用 RBAC 但 MySQL 不可用时，默认拒绝服务（RBAC_FAIL_CLOSED=True）：
    权限库是唯一的跨库鉴权入口，放行进内存里的三把 demo Key 等于权限裸奔。
    确需"权限挂了也要能聊"的场景，显式设 RBAC_FAIL_CLOSED=False 才会降级。

【历史行为（已废弃）】
    曾经 MySQL 不可用一律降级放行进内存 Key，权限取宽松集，仅在
    Principal.source 标记 'memory'——这在双库架构下等于权限裸奔，已改为 fail-closed。
"""
# 【什么】标准库与项目内依赖
# 【作用】time 计 TTL、hashlib 算 Key 摘要、mysql_client 只读连权限库、audit 写审计
import time
import hashlib
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from modules.common.logger import get_logger
from modules.enterprise.models import Principal
from modules.enterprise.mysql_client import get_mysql_ro_client
from modules.enterprise import audit

_logger = get_logger("rbac")

# 【什么】降级 Key 表：明文 Key → (用户码, 姓名, 等级)，共 3 把 demo Key
# 【作用】RBAC 关闭或 MySQL 不可用时做本地鉴权；不在表里的 Key 一律拒绝
_FALLBACK_KEYS = {
    "sk-user-normal-001":   ("U001", "张三", "normal"),
    "sk-user-advanced-002": ("U002", "李四", "advanced"),
    "sk-admin-003":         ("U003", "管理员", "admin"),
}
# 【什么】降级权限表：等级 → 权限码集合
# 【作用】给内存身份配权限；advanced/admin 是 "*" 全放开，normal 只有 3 个码
_FALLBACK_PERMS = {
    "normal":   {"tool:calculator", "kb:public", "data:users:self"},
    "advanced": {"*"},
    "admin":    {"*"},
}


# 【什么】鉴权结果容器：成功带 Principal，失败带 reason
# 【作用】把两种结果打包成统一返回值，调用方只看 ok 分支，不用判空
@dataclass
class AuthResult:
    """鉴权结果：ok=True 时 principal 必不为空。"""
    ok: bool                                # 是否放行
    principal: Optional[Principal] = None   # 身份；ok=True 时必不为空
    reason: str = "ok"                      # 结果码：ok/no_key/invalid/disabled/expired/resigned/mysql_unavailable
    message: str = ""                       # 给人看的说明，失败时才有内容


# 【什么】权限服务主体（全局单例）
# 【作用】对外只用四个方法：resolve 解析身份、allowed_tools 过滤工具、kb_expr 生成过滤、invalidate 清缓存
class RBACService:
    """权限服务（全局单例，见 get_rbac_service）。"""

    def __init__(self):
        # 【什么】把 4 个配置固化成实例属性
        # 【作用】运行期不再读配置，改配置需重启或重建实例才生效
        s = _settings()
        self._enabled = bool(s.rbac_enabled)            # 权限总开关：False 时整条链路都走内存 Key
        self._ttl = int(s.rbac_cache_ttl)               # 缓存有效期（秒），默认 60
        self._kb_acl = bool(s.kb_acl_enabled)           # 知识库隔离开关：关掉则 kb_expr 返回 None
        self._fail_closed = bool(s.rbac_fail_closed)    # 库挂时 True=拒绝请求，False=降级放行
        self._cache: dict = {}                          # {Key 摘要 16 位: (存入时刻, AuthResult)}
        self._available: Optional[bool] = None          # 库连通性探测结果，None=还没探测过


    def available(self) -> bool:
        """MySQL 权限库是否可用（结果缓存，避免每次请求都探测）。"""
        # 【什么】探测并记录权限库连通性
        # 【作用】只 ping 一次就记住结果，避免每个请求都建连探测；RBAC 未启用直接 False
        if not self._enabled:
            return False
        if self._available is None:
            self._available = get_mysql_ro_client().ping()
            _logger.info(f"MySQL 权限库可用性: {self._available}")
        return self._available


    def resolve(self, api_key: str) -> AuthResult:
        """API Key → AuthResult。缓存命中直接返回，未命中走库或降级。"""
        # 【什么】鉴权总入口，外部只调这一个方法
        # 【作用】先查缓存（命中 0 次 SQL），未命中才解析并写回缓存 + 写审计
        if not api_key:
            return AuthResult(False, None, "no_key", "缺少 API Key")
        
        # ------------------------------------------------------------------
        # 【缓存键】明文 Key 不进内存，只留 sha256 前 16 位当字典键
        #   "sk-acme-U007-admin-007" → ck = "da100e388a7b9348"
        #   （与库里鉴权用的完整 64 位 SHA2(%s,256) 无关，两者别混）
        #
        # 【命中判定】_cache 结构 {ck: (存入时刻, AuthResult)}
        #   有记录 且 time.time() - hit[0] < _ttl（默认 60 秒）才算命中，否则重查后覆盖写回
        #
        # 【边界】成功/失败结果都会进缓存，改权限最多延迟 60 秒生效；立即生效调 invalidate()
        # 【依赖警告】同一个 ck 被 _record_auth 用作审计里的 Key 摘要，改算法会对不上日志
        # ------------------------------------------------------------------
        ck = hashlib.sha256(data=api_key.encode()).hexdigest()[:16]
        hit = self._cache.get(ck)
        if hit and time.time() - hit[0] < self._ttl:
            return hit[1]

        result = self._resolve_uncached(api_key)
        self._cache[ck] = (time.time(), result)
        self._record_auth(result, ck)
        return result

    def _resolve_uncached(self, api_key: str) -> AuthResult:
        # 【什么】缓存未命中时的路径选择
        # 【作用】未启用→内存；库可用→查库；库挂→fail_closed 拒绝，否则降级内存
        if not self._enabled:
            return self._from_memory(api_key)

        if self.available():
            try:
                return self._from_db(api_key)
            except Exception as e:
                _logger.error(f"MySQL 鉴权异常: {e}")
                audit.log_audit("system", "auth", "mysql", "error", detail={"error": str(e)})
                if self._fail_closed:
                    return AuthResult(
                        False, None, "mysql_unavailable", "权限库不可用，已拒绝本次请求"
                    )
        elif self._fail_closed:
            return AuthResult(False, None, "mysql_unavailable", "权限库不可用，已拒绝本次请求")

        return self._from_memory(api_key)


    def _from_db(self, api_key: str) -> AuthResult:
        # 【什么】从 MySQL 把 Key 解析成 Principal（4 次查询）
        # 【作用】先校验 Key（无效/吊销/过期/离职）任一不过就拒绝，通过再拼身份与权限
        db = get_mysql_ro_client()

        row = db.query_one(
            "SELECT u.id AS user_id, u.user_code, u.real_name, u.job_title, u.user_level, u.status AS user_status,"
            "       t.id AS tenant_id, t.tenant_code, d.dept_code, d.dept_name,"
            "       k.status AS key_status, k.expires_at"
            "  FROM sys_api_key k"
            "  JOIN sys_user  u ON u.id = k.user_id"
            "  JOIN sys_tenant t ON t.id = u.tenant_id"
            "  JOIN sys_dept  d ON d.id = u.dept_id"
            " WHERE k.key_hash = SHA2(%s, 256)",
            (api_key,),
        )
        if not row:
            return AuthResult(False, None, "invalid", "API Key 无效或已失效")
        if row["key_status"] != 1:
            return AuthResult(False, None, "disabled", "API Key 已被吊销")
        if row["expires_at"] is not None and row["expires_at"] < datetime.now():
            return AuthResult(False, None, "expired", f"API Key 已于 {row['expires_at']} 过期")
        if row["user_status"] != 1:
            return AuthResult(False, None, "resigned", "账号已停用（离职/冻结）")

        roles = db.query_all(
            "SELECT r.role_code, r.data_scope, r.mask_level"
            "  FROM sys_user_role ur JOIN sys_role r ON r.id = ur.role_id"
            " WHERE ur.user_id = %s",
            (row["user_id"],),
        )
        perms = db.query_all(
            "SELECT DISTINCT p.perm_code"
            "  FROM sys_user_role ur"
            "  JOIN sys_role r ON r.id = ur.role_id"
            "  JOIN sys_role_permission rp ON rp.role_id = r.id"
            "  JOIN sys_permission p ON p.id = rp.perm_id"
            " WHERE ur.user_id = %s",
            (row["user_id"],),
        )
        role_codes = [r["role_code"] for r in roles]
        kb_codes = self._load_kb(db, row["tenant_id"], row["dept_code"], role_codes, row["user_code"])

        principal = Principal(
            user_code=row["user_code"],                                  # 工号，如 U007
            real_name=row["real_name"],                                  # 姓名
            tenant_code=row["tenant_code"],                              # 租户码，跨租户隔离依据
            dept_code=row["dept_code"],                                  # 部门码，dept 域过滤时用
            dept_name=row["dept_name"],                                  # 部门名
            job_title=row["job_title"] or "",                            # 岗位；库里为 NULL 时兜底空串
            user_level=row["user_level"],                                # 用户等级：normal/advanced/admin
            roles=role_codes,                                            # 角色码列表，如 ['R_FINANCE']
            permissions={p["perm_code"] for p in perms},                 # 权限码集合，如 {'tool:calculator'}
            kb_codes=kb_codes,                                           # 可见知识库；[] 表示不做库过滤
            data_scope=_max_scope([r["data_scope"] for r in roles]),     # 数据域 self/dept/all，多角色取最宽
            mask_level=max([r["mask_level"] for r in roles], default=0), # 脱敏 0不脱敏/1部分/2强，多角色取最强
            source="mysql",                                              # 来源标记：库里查出的真实身份
        )
        return AuthResult(True, principal, "ok", "")

    def _load_kb(self, db, tenant_id, dept_code, role_codes, user_code) -> list:
        """可见知识库 = public 库 ∪ 按部门/角色/个人授权的私有库。"""
        # 【什么】查该用户可见的知识库编码列表
        # 【作用】结果写进 Principal.kb_codes，供 kb_expr 生成检索过滤；隔离关闭时返回 []
        if not self._kb_acl:
            return []
        placeholders = ",".join(["%s"] * len(role_codes)) or "''"
        rows = db.query_all(
            "SELECT kb_code FROM sys_kb WHERE tenant_id = %s AND visibility = 'public'"
            " UNION "
            "SELECT k.kb_code FROM sys_kb k JOIN sys_kb_grant g ON g.kb_id = k.id AND g.can_read = 1"
            " WHERE k.tenant_id = %s AND ("
            "   (g.grant_type = 'dept' AND g.target_code = %s) OR"
            "   (g.grant_type = 'role' AND g.target_code IN (" + placeholders + ")) OR"
            "   (g.grant_type = 'user' AND g.target_code = %s))",
            (tenant_id, tenant_id, dept_code, *role_codes, user_code),
        )
        return [r["kb_code"] for r in rows]


    def _from_memory(self, api_key: str) -> AuthResult:
        # 【什么】用内存表拼一个降级身份
        # 【作用】source='memory' 标记来源是本地兜底，调用方据此知道这不是真实库结果
        info = _FALLBACK_KEYS.get(api_key)
        if not info:
            return AuthResult(False, None, "invalid", "API Key 无效或已失效")
        code, name, level = info
        principal = Principal(
            user_code=code,                                              # 工号，取自 _FALLBACK_KEYS
            real_name=name,                                              # 姓名
            tenant_code="local",                                         # 固定租户：本地演示
            dept_code="D_LOCAL",                                         # 固定部门码
            dept_name="本地演示",                                            # 固定部门名
            user_level=level,                                            # 等级：normal/advanced/admin
            roles=[f"FALLBACK_{level.upper()}"],                         # 假角色名，如 FALLBACK_ADMIN
            permissions=set(_FALLBACK_PERMS.get(level, set())),          # 按等级取权限；未知等级得空集
            kb_codes=[],                                                 # 空=不做库过滤，kb_expr 返回 None
            data_scope="all" if level in ("advanced", "admin") else "self",  # advanced/admin 全公司，normal 仅本人
            mask_level=0,                                                # 降级身份一律不脱敏
            source="memory",                                             # 来源标记：内存兜底，非库里身份
        )
        return AuthResult(True, principal, "ok", "降级鉴权（MySQL 不可用）")


    def allowed_tools(self, principal: Principal, tools: list) -> list:
        """
        按权限过滤工具：模型看不到 = 调不到（从源头杜绝越权）。

        权限码：本地工具 `tool:{name}`；MCP 工具同时认 `tool:mcp:{name}` 与 `tool:mcp:*`。
        未登录/无 Principal 时返回空列表，绝不返回全部。
        """
        # 【什么】按权限码过滤工具列表
        # 【作用】越权工具根本不进模型的工具表，从源头杜绝调用；无身份返回 []
        if principal is None:
            return []
        allowed = []
        for t in tools:
            codes = [f"tool:{t.name}"]
            if getattr(t, "tool_type", "local") == "mcp":
                codes.append(f"tool:mcp:{t.name}")
            if principal.can_any(codes):
                allowed.append(t)
            else:
                _logger.info(f"工具被权限过滤: {t.name} | user={principal.user_code}")
        return allowed

    def kb_expr(self, principal: Principal) -> Optional[str]:
        """
        生成 Milvus 检索的标量过滤表达式。

        ✓ 财务专员 → 'kb_code in ["public","finance"]'
        ✓ 管理员（全量授权）→ 同样的 in 表达式，不做全表
        ✗ kb_codes 为空 → None（未开启隔离，保持旧行为）
        """
        # 【什么】生成 Milvus 的 kb_code 过滤表达式
        # 【作用】让检索只在可见库里捞；无身份/无可见库/隔离关闭时返回 None 走旧行为
        if principal is None or not principal.kb_codes or not self._kb_acl:
            return None
        codes = ",".join(f'"{c}"' for c in principal.kb_codes)
        return f"kb_code in [{codes}]"

    def invalidate(self):
        """清空权限缓存（改权限后立即生效时调用）。"""
        # 【什么】手动清空缓存
        # 【作用】改权限后立刻生效；不调的话旧结果最多延续 TTL（默认 60 秒）
        self._cache.clear()


    def _record_auth(self, result: AuthResult, cache_key: str):
        """鉴权结果入审计（失败也记，key 只留摘要）。"""
        # 【什么】把本次鉴权结果写审计
        # 【作用】放行与拒绝都留痕，资源名只写 api_key:<16 位摘要>，凭摘要反查是哪把 Key
        if result.ok:
            p = result.principal
            audit.log_audit(p.user_code, "auth", "api_key:" + cache_key, "allow", dept_code=p.dept_code,
                            detail={"source": p.source, "roles": p.roles})
        else:
            audit.log_audit("anonymous", "auth", "api_key:" + cache_key, "deny",
                            detail={"reason": result.reason})



# 【什么】数据域优先级表（self 0 < dept 1 < all 2）
# 【作用】给 _max_scope 排序用，数字越大可见范围越宽
_SCOPE_ORDER = {"self": 0, "dept": 1, "all": 2}


def _max_scope(scopes: list) -> str:
    """多角色数据域冲突时取最大可见范围。"""
    # 【什么】多角色数据域取最宽的那个
    # 【作用】一人多角色时权限不互相挤压，取最大值；空列表默认最窄的 "self"
    result = "self"
    for s in scopes or []:
        if _SCOPE_ORDER.get(s, 0) > _SCOPE_ORDER.get(result, 0):
            result = s
    return result


def _settings():
    # 【什么】函数内延迟导入配置
    # 【作用】避免 import 本模块就触发 config 加载，绕开循环依赖
    from config import get_settings
    return get_settings()


_service: Optional[RBACService] = None


def get_rbac_service() -> RBACService:
    """全局单例权限服务。"""
    # 【什么】懒加载的进程级单例
    # 【作用】全进程共用一个实例，缓存与可用性探测结果才能跨请求复用
    global _service
    if _service is None:
        _service = RBACService()
    return _service
