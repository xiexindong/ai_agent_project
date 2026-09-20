"""
鉴权与限流模块
===============
企业级后端安全：用户鉴权、请求限流、权限分级、会话隔离校验。
覆盖边界9.5：用户身份权限隔离、会话权限管控、工具权限分级、接口鉴权、资源访问限流。
"""
import time
from collections import defaultdict
from typing import Optional
from enum import Enum
from fastapi import Request, HTTPException, Security, Header
from fastapi.security import APIKeyHeader
from modules.common.logger import get_logger
from config import get_settings

_logger = get_logger("auth")


class UserLevel(str, Enum):
    """用户权限等级。"""
    NORMAL = "normal"
    ADVANCED = "advanced"
    ADMIN = "admin"


class AuthManager:
    """
    用户鉴权管理（API Key + 用户映射）。
    生产环境应对接 JWT/OAuth，此处用 API Key 演示完整鉴权链路。
    """

    _API_KEYS = {
        "sk-user-normal-001": {"user_id": "U001", "name": "张三", "level": UserLevel.NORMAL},
        "sk-user-advanced-002": {"user_id": "U002", "name": "李四", "level": UserLevel.ADVANCED},
        "sk-admin-003": {"user_id": "U003", "name": "管理员", "level": UserLevel.ADMIN},
    }

    @classmethod
    def authenticate(cls, api_key: str) -> Optional[dict]:
        """
        验证 API Key，返回用户信息 dict 或 None。

        【鉴权流程】请求头 X-API-Key → 查 _API_KEYS 表 → 命中返回用户信息
            失败不抛异常，返回 None 让上层决定（统一返回 403 避免暴露"哪个 key 错了"）

        【输入输出对比】
            ✓ 输入 "sk-user-normal-001" → {"user_id":"U001","name":"张三","level":UserLevel.NORMAL}
            ✓ 输入 "sk-admin-003"       → {"user_id":"U003","name":"管理员","level":UserLevel.ADMIN}
            ✗ 输入 "sk-fake-999"       → None（无效 key）
            ✗ 输入 ""                  → None（未提供）

        生产环境应替换为 JWT/OAuth + DB 查询，此处用内存字典演示完整鉴权链路。
        """
        user = cls._API_KEYS.get(api_key)
        if not user:
            _logger.warning(f"鉴权失败: api_key={api_key[:12]}...")
            return None
        _logger.info(f"鉴权成功: user={user['user_id']}, level={user['level']}")
        return user

    @classmethod
    def check_session_owner(cls, user_id: str, session_id: str, pg_manager) -> bool:
        """
        校验会话归属：仅会话所属用户可访问（防会话劫持）。

        【为什么要这一步？】用户 A 拿着合法 API Key，但用 session_id=B 的会话查询。
            不校验就会让 A 看到 B 的对话历史——会话劫持漏洞。
            边界9.5 的会话权限管控。
        """
        try:
            return bool(pg_manager.get_session_state(session_id, user_id))
        except Exception:
            return False


class RateLimiter:
    """
    请求限流器：基于滑动窗口算法管控单用户请求频次。

    【为什么要限流？】
        防服务雪崩：一个用户疯狂打接口，把 LLM/PG 资源耗光，其他用户全卡死
        防计费超标：DeepSeek 按 token 计费，恶意用户一晚能花掉几千块
        边界9.5 的资源访问限流。

    【滑动窗口 vs 固定窗口】
        固定窗口：每分钟重置计数 → 用户在 59 秒发 20 个 + 1 秒后再发 20 个 = 40 个/2 秒
        滑动窗口：只看"现在往前 60 秒内"的请求数 → 不存在边界翻倍漏洞

    【限流前后对比】默认配置 max_requests=20 / window_seconds=60
        ✓ 用户 1 分钟内发 1~20 个请求 → check() 返回 True，全放行
        ✗ 用户 1 分钟内发第 21 个请求 → check() 返回 False，触发 429
        ✓ 30 秒后再发请求（窗口滑走 30 个旧的）→ 又能发，最多发到 50 个/分（不超 20 滑动窗口内）
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._records: dict = defaultdict(list)

    def check(self, user_id: str) -> bool:
        """
        检查用户是否在滑动窗口内超限，返回 True=放行 / False=拒绝。

        【一次调用的内部状态变化】max=20, window=60s
            时刻 t=0s   records=[]                      → True（计数0<20，放行，append）
            时刻 t=1s   records=[0]                     → True（计数1<20，放行，append）
            ...
            时刻 t=20s  records=[0,1,...,19]            → True（计数20不<20，触发 False）
            时刻 t=61s  过滤掉 t<1s 的，剩 1~19         → True（窗口滑走 0s 那条）
        """
        now = time.time()
        records = self._records[user_id]
        records[:] = [t for t in records if now - t < self._window]
        if len(records) >= self._max:
            _logger.warning(
                f"限流触发: user={user_id} | "
                f"请求数={len(records)}/{self._max} per {self._window}s"
            )
            return False
        records.append(now)
        return True


_rate_limiter = RateLimiter(max_requests=20, window_seconds=60)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(api_key: str = Security(_api_key_header)) -> dict:
    """
    FastAPI 依赖注入主入口：API Key 鉴权 + 限流检查一锅端。

    【企业级改造点】身份不再来自内存字典，而是走 MySQL 的 RBAC：
        X-API-Key → SHA2 哈希查 sys_api_key → 校验有效期/吊销/在职状态
                  → 装配角色、权限码、数据域、知识库可见范围 → Principal

    【一次请求的四层校验】
        1. 缺 API Key 头        → 401（未登录）
        2. Key 无效/吊销/过期   → 403（鉴权失败，且不区分原因避免信息泄露）
        3. 账号停用（离职）      → 403
        4. 鉴权通过但限流触发    → 429（请求太频繁）

    【返回的 user dict 新增字段】
        user_id    工号（U001），与 PG 会话表的 user_id 对齐
        level      normal/advanced/admin，兼容旧代码
        principal  Principal 对象，Agent 全链路的权限载体（新增，核心）

    Returns:
        鉴权通过的 user dict；失败时 raise HTTPException，本身不会返回 None
    """
    from modules.enterprise.rbac_service import get_rbac_service

    if not api_key:
        raise HTTPException(status_code=401, detail="缺少 API Key，请在请求头携带 X-API-Key")

    result = get_rbac_service().resolve(api_key)
    if not result.ok:
        raise HTTPException(status_code=403, detail=result.message or "API Key 无效或已失效")

    p = result.principal
    if not _rate_limiter.check(p.user_code):
        raise HTTPException(status_code=429, detail="请求频次超限，请稍后重试")

    return {
        "user_id": p.user_code,
        "name": p.real_name,
        "level": p.user_level,
        "principal": p,
        "dept_code": p.dept_code,
        "dept_name": p.dept_name,
        "roles": p.roles,
        "source": p.source,
    }


def RequirePermission(required_level: UserLevel):
    """
    权限分级装饰器工厂：返回一个 FastAPI 依赖，校验用户等级是否够。

    【三级权限对照】level_order: NORMAL=1, ADVANCED=2, ADMIN=3
        ✓ NORMAL 用户访问 /chat（要求 NORMAL）→ 1 >= 1，放行
        ✓ ADMIN 用户访问 /chat（要求 NORMAL）→ 3 >= 1，放行（高级别可访问低级接口）
        ✗ NORMAL 用户访问 /admin/audit（要求 ADMIN）→ 1 < 3，HTTP 403 "权限不足"
        ✗ ADVANCED 用户访问 /admin/audit → 2 < 3，HTTP 403

    【装饰器工厂模式】类比 NestJS 的 @Roles('admin') 装饰器——
        不是装饰器本身，而是返回一个依赖函数的工厂。
        路由用 Depends(RequirePermission(UserLevel.ADMIN)) 挂上。

    Args:
        required_level: 该接口要求的最低权限等级

    Returns:
        dependency: 一个 async 函数，FastAPI 在请求时调用它做权限校验
    """
    level_order = {UserLevel.NORMAL: 1, UserLevel.ADVANCED: 2, UserLevel.ADMIN: 3}

    async def dependency(user: dict = Security(get_current_user)) -> dict:
        user_level = user["level"] if isinstance(user["level"], UserLevel) else UserLevel(user["level"])
        if level_order.get(user_level, 0) < level_order.get(required_level, 0):
            _logger.warning(
                f"权限不足: user={user['user_id']} level={user_level} < required={required_level}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"权限不足，需要 {required_level.value} 等级",
            )
        return user
    return dependency


def RequirePerm(perm_code: str):
    """
    细粒度权限依赖工厂（RBAC 真正在用的那把锁）。

    与 RequirePermission 的区别：
        RequirePermission(UserLevel.ADMIN)  —— 一维分级，只能表达"等级够不够"
        RequirePerm("api:admin:*")          —— 权限码，能表达"能不能做这件事"

    企业级接口应当用后者：等级是给人看的属性，权限码才是给系统判定的契约。

    用法：
        @app.get("/admin/audit")
        async def audit_log(user=Depends(RequirePerm("api:audit:*"))): ...
    """
    async def dependency(user: dict = Security(get_current_user)) -> dict:
        principal = user.get("principal")
        allowed = principal is not None and principal.can(perm_code)
        from modules.enterprise import audit
        audit.log_audit(
            principal.user_code if principal else "anonymous",
            "api",
            perm_code,
            "allow" if allowed else "deny",
            dept_code=principal.dept_code if principal else "",
        )
        if not allowed:
            raise HTTPException(status_code=403, detail="权限不足")
        return user
    return dependency


async def get_principal(user: dict = Security(get_current_user)):
    """直接拿 Principal 的依赖（Agent 链路专用，省一层取值）。"""
    return user["principal"]
