"""
后端 API 服务模块
==================
FastAPI 企业级后端服务：接口鉴权、请求限流、操作审计、多用户隔离。
覆盖9.5节：用户身份权限隔离、会话权限管控、工具权限分级、接口鉴权、操作日志审计、资源访问限流。
"""
# 【什么】鉴权与限流组件：等级枚举、Key 校验器、限流器、依赖工厂
# 【作用】对外统一从包根导入，调用方不必知道它们定义在 modules.api.auth
from modules.api.auth import (
    AuthManager, RateLimiter, get_current_user,
    UserLevel, RequirePermission,
)
# 【什么】FastAPI 应用工厂与服务封装
# 【作用】main.py 只需 create_app() 或 APIServer() 就能起服务
from modules.api.server import create_app, APIServer

# 【什么】包的公开导出名单
# 【作用】限定 from modules.api import * 能拿到什么，内部符号不外泄
__all__ = [
    "AuthManager", "RateLimiter", "get_current_user",
    "UserLevel", "RequirePermission", "create_app", "APIServer",
]
