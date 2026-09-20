"""
企业身份与权限包
================
把「谁是谁、能干什么」从代码里搬进 MySQL，形成可运维的 RBAC 体系：

    rbac_service  →  RBACService：API Key → Principal（身份+角色+权限+数据域+知识库可见范围）
    models        →  Principal：贯穿一次请求的权限载体，Agent 全链路只认它
    mysql_client  →  MySQLClient：带自动重连的极简 MySQL 访问层
    audit         →  审计与 LLM 用量落库（写失败只告警，绝不阻断主流程）
    masking       →  敏感字段脱敏（按角色 mask_level 分级）

设计原则：
    1. 权限判定只在入口做一次，之后全链路传递 Principal，避免逐层查库
    2. MySQL 不可用时整体降级为「内存鉴权 + 宽松权限」，Agent 照常可跑
    3. 拒绝也要落审计（result='deny'），越权尝试是安全审计的重点对象
"""
from modules.enterprise.models import Principal
from modules.enterprise.rbac_service import RBACService, get_rbac_service

__all__ = ["Principal", "RBACService", "get_rbac_service"]
