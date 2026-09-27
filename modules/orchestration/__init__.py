"""
编排与治理层
============
中心化编排 + 标准化消息协议 + 统一观测链路。

四层的分工与依赖方向：

    protocol      标准信封与路由裁决（被所有层依赖）
    tracer        span 树观测（独立，只被编排器与 harness 使用）
    control       限流、配额、重试、超时（独立）
    guard         权限、敏感词、参数校验（依赖 protocol）
    orchestrator  中心化编排器：把上面三层串成一条必经之路
    bus           Agent 侧门面：读走总线，写必经编排器

【为什么分成四个文件而不是一个大文件】
    这四件事的变更频率完全不同：协议几乎不改，观测随排障需求常改，
    限流参数按环境调，权限规则按合规要求改。混在一起，
    改一次限流参数就要重新评审权限逻辑。
"""
from modules.orchestration.bus import OrchestratedBus
from modules.orchestration.control import (
    Quota,
    RateLimiter,
    TokenBucket,
    remaining_seconds,
    retry_with_backoff,
)
from modules.orchestration.guard import DEFAULT_KIND_PERMISSIONS, GuardPolicy, PolicyGuard
from modules.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from modules.orchestration.protocol import (
    PROTOCOL_VERSION,
    DeliveryStatus,
    Envelope,
    RejectCode,
    RouteAction,
    RouteDecision,
    envelope_to_log_dict,
)
from modules.orchestration.tracer import Span, SpanStatus, Trace, Tracer

__all__ = [
    # 协议
    "PROTOCOL_VERSION", "Envelope", "DeliveryStatus", "RouteAction", "RouteDecision",
    "RejectCode", "envelope_to_log_dict",
    # 观测
    "Tracer", "Trace", "Span", "SpanStatus",
    # 负载控制
    "TokenBucket", "RateLimiter", "Quota", "retry_with_backoff", "remaining_seconds",
    # 安全
    "GuardPolicy", "PolicyGuard", "DEFAULT_KIND_PERMISSIONS",
    # 编排
    "Orchestrator", "OrchestratorConfig", "OrchestratedBus",
]
