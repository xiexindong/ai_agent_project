"""
Human-in-the-Loop 子包：让人能在 Agent 跑的过程中插手。

【与已有的 review_queue 是什么关系】
    两者都是"人来看一眼"，但介入时机完全不同：

        review_queue（memory/pg_manager）  事后离线复核
            对话已经结束，可疑记录被插旗进队列表，
            人什么时候看都行，看了也不影响已经给出的答案。
            适合抽检、合规留痕。

        hitl（本包）                       事中阻塞式拍板
            Agent 跑到一半停下来等，人不回复就一直等（或超时兜底），
            回复内容会改变接下来的执行（放行 / 改参数 / 跳过 / 终止）。
            适合高危动作、依据不足、结论冲突。

    一句话区分：**review_queue 是"事后看回放"，hitl 是"直播时喊停"**。

【最小用法】
    gate = build_gate(enabled=True, sensitive_tools=["data_query"])
    go, args = gate.ask_tool("data_query", {"id": "1"})     # go=False 就跳过这步

    # 单元测试 / CI（没有人）：注入自动答复器
    gate = build_gate(enabled=True, resolver=auto_resolver(HumanAction.APPROVE))
"""
from typing import Optional

from modules.common.logger import get_logger
from modules.hitl.gate import HumanGate, Resolver, auto_resolver, scripted_resolver
from modules.hitl.models import (
    HumanAction,
    HumanReply,
    InterruptKind,
    InterruptRecord,
    InterruptRequest,
    InterruptStatus,
    new_interrupt_id,
    now_ms,
)
from modules.hitl.notify import (
    InterruptNotifier,
    build_notifier,
    format_interrupt_text,
    get_global_notifier,
    set_global_notifier,
)
from modules.hitl.policy import HITLPolicy
from modules.hitl.store import InMemoryInterruptStore, InterruptStore

_logger = get_logger("hitl")


# 【什么】一行构造闸门
# 【作用】调用方不必自己拼 policy / store / resolver 三件套
def build_gate(
    enabled: bool = False,
    sensitive_tools: Optional[list] = None,
    resolver=None,
    timeout_seconds: float = 60.0,
    on_event=None,
    **policy_kwargs,
) -> HumanGate:
    """构造一个人工闸门。

    Args:
        enabled:        总开关；关掉时所有 ask 立即放行
        sensitive_tools: 需要批准的工具名
        resolver:       自动答复器（测试/CI 用 auto_resolver）
        timeout_seconds: 单次等待上限
        on_event:       请求登记后的通知钩子（InterruptNotifier.notify）
    """
    policy = HITLPolicy(
        enabled=enabled,
        sensitive_tools=list(sensitive_tools or []),
        timeout_seconds=timeout_seconds,
        **policy_kwargs,
    )
    return HumanGate(policy=policy, resolver=resolver, on_event=on_event)


# ---------------------------------------------------------------------------
# 【什么】进程级共享闸门
# 【作用】审批接口和 Agent 链路必须用同一个 gate 实例，否则
#        人在网页上点了批准，Agent 那边等的还是另一个 store，永远收不到。
#
#   【为什么必须是全局单例】
#       harness 每次 run 都可能新建，若每个 harness 自带一个 gate，
#       HTTP 审批端点就不知道该往哪个 gate 提交。
#       共享一个进程级 gate 后，"谁抛出的中断"与"谁来回复"自然对上。
#
#   【依赖警告】单实例部署才成立。多副本部署时审批请求可能打到另一台机器，
#       此时必须把 store 换成共享存储（如 PG），接口一行不用改。
# ---------------------------------------------------------------------------
_GLOBAL_GATE: Optional["HumanGate"] = None


def get_global_gate(enabled: Optional[bool] = None, **policy_kwargs) -> "HumanGate":
    """取进程级共享闸门（第一次调用时按配置建造）。"""
    global _GLOBAL_GATE
    if _GLOBAL_GATE is None:
        from config import get_settings
        s = get_settings()
        _GLOBAL_GATE = build_gate(
            enabled=s.hitl_enabled if enabled is None else enabled,
            sensitive_tools=[t for t in s.hitl_sensitive_tools.split(",") if t.strip()],
            kinds=[k for k in s.hitl_kinds.split(",") if k.strip()],
            timeout_seconds=s.hitl_timeout_seconds,
            **policy_kwargs,
        )
        # 【挂上通知器】不通过 build_gate 传参，而是造完再补：
        #   通知器要读配置建造，而 build_gate 本身不依赖配置 ——
        #   这样单测里 new 一个独立 gate 时不会被全局通知器打扰。
        try:
            _GLOBAL_GATE.on_event = get_global_notifier().notify
        except Exception as e:
            _logger.warning(f"HITL 通知器装配失败（仍可审批，只是没人被通知）: {e}")
    return _GLOBAL_GATE


def set_global_gate(gate: Optional["HumanGate"]) -> None:
    """替换共享闸门（演示、测试注入自动答复器时用）。"""
    global _GLOBAL_GATE
    _GLOBAL_GATE = gate


__all__ = [
    "get_global_gate", "set_global_gate",
    "InterruptKind", "HumanAction", "InterruptStatus",
    "InterruptRequest", "HumanReply", "InterruptRecord",
    "new_interrupt_id", "now_ms",
    "HITLPolicy",
    "InterruptStore", "InMemoryInterruptStore",
    "HumanGate", "Resolver", "auto_resolver", "scripted_resolver",
    "build_gate",
    "InterruptNotifier", "build_notifier", "format_interrupt_text",
    "get_global_notifier", "set_global_notifier",
]
