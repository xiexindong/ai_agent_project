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
from typing import Optional  # 可选类型标注

from modules.common.logger import get_logger  # 项目统一日志器工厂
from modules.hitl.gate import HumanGate, Resolver, auto_resolver, scripted_resolver  # 闸门与自动答复器
from modules.hitl.models import (  # 导入数据契约
    HumanAction,  # 人工决策动作枚举
    HumanReply,  # 人工回复模型
    InterruptKind,  # 中断点类型枚举
    InterruptRecord,  # 请求 + 回复归档模型
    InterruptRequest,  # 中断请求模型
    InterruptStatus,  # 生命周期状态枚举
    new_interrupt_id,  # 短 ID 生成工具
    now_ms,  # 毫秒时间戳工具
)
from modules.hitl.notify import (  # 导入通知器
    InterruptNotifier,  # 通知器类
    build_notifier,  # 按配置造通知器
    format_interrupt_text,  # 请求格式化文本工具
    get_global_notifier,  # 取进程级共享通知器
    set_global_notifier,  # 替换进程级共享通知器
)
from modules.hitl.policy import HITLPolicy  # 触发策略
from modules.hitl.store import InMemoryInterruptStore, InterruptStore  # 存储协议与进程内实现

_logger = get_logger("hitl")  # 本模块日志器


# 【什么】一行构造闸门
# 【作用】调用方不必自己拼 policy / store / resolver 三件套
# 【为什么】构造细节收敛一处，调用方只给开关与名单
def build_gate(
    enabled: bool = False,  # 总开关；关掉时所有 ask 立即放行
    sensitive_tools: Optional[list] = None,  # 需要人工批准的工具名
    resolver=None,  # 自动答复器（测试 / CI 用 auto_resolver）
    timeout_seconds: float = 60.0,  # 单次等待上限
    on_event=None,  # 请求登记后的通知钩子（InterruptNotifier.notify）
    **policy_kwargs,  # 其余策略字段透传（kinds / tool_action 等）
) -> HumanGate:  # 返回造好的闸门
    """构造一个人工闸门。

    Args:
        enabled:        总开关；关掉时所有 ask 立即放行
        sensitive_tools: 需要批准的工具名
        resolver:       自动答复器（测试/CI 用 auto_resolver）
        timeout_seconds: 单次等待上限
        on_event:       请求登记后的通知钩子（InterruptNotifier.notify）
    """
    policy = HITLPolicy(  # 组装策略对象
        enabled=enabled,  # 总开关
        sensitive_tools=list(sensitive_tools or []),  # 敏感工具名单，缺省为空
        timeout_seconds=timeout_seconds,  # 等待上限
        **policy_kwargs,  # 透传其余策略配置
    )
    return HumanGate(policy=policy, resolver=resolver, on_event=on_event)  # 用策略造闸门（存储用默认内存实现）


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
_GLOBAL_GATE: Optional["HumanGate"] = None  # 进程级共享闸门实例，首次使用时建造


# 【什么】取进程级共享闸门
# 【作用】首次调用按配置建造，之后返回同一实例
# 【为什么】审批接口和 Agent 链路必须用同一个 gate，否则人点了批准 Agent 永远收不到
def get_global_gate(enabled: Optional[bool] = None, **policy_kwargs) -> "HumanGate":  # enabled 可显式覆盖配置开关
    """取进程级共享闸门（第一次调用时按配置建造）。"""
    global _GLOBAL_GATE  # 声明读写模块级变量
    if _GLOBAL_GATE is None:  # 尚未建造
        from config import get_settings  # 延迟导入，避免模块级循环依赖
        s = get_settings()  # 读全局配置
        _GLOBAL_GATE = build_gate(  # 按配置建造闸门
            enabled=s.hitl_enabled if enabled is None else enabled,  # 显式传参优先于配置
            sensitive_tools=[t for t in s.hitl_sensitive_tools.split(",") if t.strip()],  # 逗号分隔字符串转工具名单
            kinds=[k for k in s.hitl_kinds.split(",") if k.strip()],  # 逗号分隔字符串转中断点白名单
            timeout_seconds=s.hitl_timeout_seconds,  # 等待上限
            **policy_kwargs,  # 透传其余配置
        )
        # 【挂上通知器】不通过 build_gate 传参，而是造完再补：
        #   通知器要读配置建造，而 build_gate 本身不依赖配置 ——
        #   这样单测里 new 一个独立 gate 时不会被全局通知器打扰。
        try:
            _GLOBAL_GATE.on_event = get_global_notifier().notify  # 把共享通知器挂为事件钩子
        except Exception as e:  # 通知器装配失败
            _logger.warning(f"HITL 通知器装配失败（仍可审批，只是没人被通知）: {e}")  # 只记警告，不阻断
    return _GLOBAL_GATE  # 返回共享闸门


# 【什么】替换进程级共享闸门
# 【作用】演示、测试注入自动答复器时换实现
# 【为什么】不替换的话测试会走真人等待，永远卡住
def set_global_gate(gate: Optional["HumanGate"]) -> None:  # 传 None 可清空
    """替换共享闸门（演示、测试注入自动答复器时用）。"""
    global _GLOBAL_GATE  # 声明写模块级变量
    _GLOBAL_GATE = gate  # 替换共享实例


__all__ = [  # 公开导出清单
    "get_global_gate", "set_global_gate",  # 共享闸门的取与换
    "InterruptKind", "HumanAction", "InterruptStatus",  # 枚举契约
    "InterruptRequest", "HumanReply", "InterruptRecord",  # 数据模型
    "new_interrupt_id", "now_ms",  # 工具函数
    "HITLPolicy",  # 触发策略
    "InterruptStore", "InMemoryInterruptStore",  # 存储协议与实现
    "HumanGate", "Resolver", "auto_resolver", "scripted_resolver",  # 闸门与答复器
    "build_gate",  # 一行构造
    "InterruptNotifier", "build_notifier", "format_interrupt_text",  # 通知器
    "get_global_notifier", "set_global_notifier",  # 共享通知器的取与换
]
