"""
后端 API 服务模块
==================
FastAPI 企业级后端服务：接口鉴权、请求限流、操作审计、多用户隔离。
覆盖9.5节：用户身份权限隔离、会话权限管控、工具权限分级、接口鉴权、操作日志审计、资源访问限流。
"""
from modules.api.auth import (
    AuthManager, RateLimiter, get_current_user,
    UserLevel, RequirePermission,
)
from modules.api.server import create_app, APIServer

__all__ = [
    "AuthManager", "RateLimiter", "get_current_user",
    "UserLevel", "RequirePermission", "create_app", "APIServer",
]
