"""
标准化消息协议（Envelope）
==========================
定义「所有 Agent 通信都必须遵守的信封格式」和「编排器的路由裁决格式」。

【为什么要有信封】
    Agent 之间直接传字符串能跑通，但跑不通生产：
        1. 无法回答"这条消息是从哪一步发出来的"（缺 trace_id / parent_id）
        2. 无法回答"为什么这条没送到"（缺 delivery 状态与 reason）
        3. 无法回答"是不是有人在互相踢皮球"（缺 hop / round 计数）
    信封把这三件事固化成字段，编排器才有的可判、可拦、可追溯。

【与 AgentMessage 的关系】
    Envelope 继承 AgentMessage，因此过去所有接收 AgentMessage 的代码
    （bus.inbox / inbox_text / objections）不需要改就能读到更全的信息。
    新增字段全部有默认值，旧代码构造出的对象依然是合法信封。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from modules.multiagent.models import AgentMessage, MessageKind

# 【协议版本号】
#   进 trace 与落库。将来改字段语义时，靠它区分"老格式的数据"和"新格式的数据"，
#   否则回溯半年前的链路会发现字段含义对不上，但看不出是哪一版写的。
PROTOCOL_VERSION = "1.0"


# ---------------------------------------------------------------------------
# 【什么】一条消息最终落到什么状态
# 【作用】溯源时先问"送到了吗"，再问"为什么"
#   PENDING   刚建出来，还没过编排器
#   DELIVERED 通过全部校验，已进总线
#   REJECTED  被校验拦下（权限/敏感词/参数），调用方需要知道"被拒了"
#   DROPPED   被负载控制丢弃（限流/配额），不通知调用方，只留痕
#   ESCALATED 判定为无效交互（踢皮球/超跳数），已升级给主管而非送达对方
# ---------------------------------------------------------------------------
class DeliveryStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    REJECTED = "rejected"
    DROPPED = "dropped"
    ESCALATED = "escalated"


class RouteAction(str, Enum):
    """编排器对一条消息作出的裁决动作。"""

    DELIVER = "deliver"      # 正常投递
    REJECT = "reject"        # 校验不通过，明确拒绝（调用方可见）
    DROP = "drop"            # 负载原因静默丢弃（只留痕，不打断发送方）
    ESCALATE = "escalate"    # 判定为空转，交给主管/人工，不投给原接收方


class RejectCode(str, Enum):
    """拒绝/丢弃原因码：监控按它聚合，不解析中文文案。"""

    OK = "ok"
    BAD_PROTOCOL = "bad_protocol"        # 字段缺失或非法
    TOO_LARGE = "too_large"              # 正文超长
    SENDER_UNKNOWN = "sender_unknown"    # 发送者不在花名册
    RECEIVER_UNKNOWN = "receiver_unknown"
    KIND_FORBIDDEN = "kind_forbidden"    # 该角色不允许发这种消息
    SENSITIVE = "sensitive"              # 命中敏感词且策略为拒绝
    RATE_LIMITED = "rate_limited"        # 限流
    QUOTA_EXCEEDED = "quota_exceeded"    # 配额用尽
    MAX_HOPS = "max_hops"                # 跳数上限（防 A→B→A→B 死循环）
    LOOP_DETECTED = "loop_detected"      # 内容重复/乒乓，判定为无效交互


class Envelope(AgentMessage):
    """标准化的通信信封。

    【新增字段各自解决什么问题】
        trace_id  —— 把消息挂到某次运行的观测链路上（一个 trace 下所有 span 同源）
        parent_id —— 这条消息是回复哪条（拼得出对话链，不是一袋散消息）
        round     —— 协作轮次，跨轮的消息默认不投递（旧轮次的异议不污染新轮）
        hop       —— 第几跳，配合 max_hops 防 A、B 无限来回
        sender_role / priority / meta / size_bytes —— 校验与调度用

    【输入 → 输出对照】
        输入: Envelope(sender="analyst_1", receiver="researcher_1",
                       content="口径是含税吗", kind=MessageKind.CLARIFY, hop=1)
        输出: 该对象 .size_bytes == 6（正文 6 个字符），
              .delivery == DeliveryStatus.PENDING，.trace_id == ""
    """

    trace_id: str = Field(default="", description="所属观测链路 ID")
    run_id: str = Field(default="", description="所属运行 ID")
    parent_id: str = Field(default="", description="回复的上一条消息 id；顶级消息为空")
    round: int = Field(default=0, ge=0, description="协作轮次，从 0 开始")
    hop: int = Field(default=0, ge=0, description="第几跳，回复产生的消息 = 父消息 hop + 1")
    sender_role: str = Field(default="", description="发送者角色，供权限矩阵判定")
    priority: int = Field(default=0, description="数值越大越优先，仅用于排序")
    delivery: DeliveryStatus = Field(default=DeliveryStatus.PENDING, description="最终投递状态")
    reason: str = Field(default="", description="裁决理由，被拦下时必有值")
    meta: dict[str, Any] = Field(default_factory=dict, description="结构化附加数据")
    protocol: str = Field(default=PROTOCOL_VERSION, description="写入时使用的协议版本")

    # 【为什么 size_bytes 做成计算属性而不是字段】
    #   正文可能被敏感词打码改写，存成字段就会和 content 不一致；
    #   实时算永远是对的。参数校验读它时也不用担心"还没算过"。
    @property
    def size_bytes(self) -> int:
        return len(self.content or "")

    @property
    def is_delivered(self) -> bool:
        return self.delivery == DeliveryStatus.DELIVERED

    def mark(self, status: DeliveryStatus, reason: str = "") -> "Envelope":
        """原地标记状态并回填理由，返回自身便于链式使用。"""
        self.delivery = status
        if reason:
            self.reason = reason
        return self

    def reply(self, sender: str, content: str, kind: MessageKind = MessageKind.CLARIFY,
              **kw) -> "Envelope":
        """基于本条消息构造一条回复。

        【为什么 hop 由协议负责 +1 而不是让调用方传】
            让调用方自己填 hop，等于把"防死循环"的成败交给了每个写消息的人，
            漏填一次就回到 0，上限形同虚设。协议里 +1 就没这个口子。
        """
        return Envelope(
            sender=sender,
            receiver=self.sender,          # 默认回给发信方
            kind=kind,
            content=content,
            trace_id=self.trace_id,
            run_id=self.run_id,
            parent_id=self.id,
            round=self.round,
            hop=self.hop + 1,
            **kw,
        )


class RouteDecision(BaseModel):
    """编排器对一条消息作出的裁决。

    【为什么要把"裁决"也建模】
        只返回 bool 的话，线上只能看到"消息没送到"，
        看不出是被权限拦了、被限流丢了、还是被判定为踢皮球 —— 这三类
        排障动作完全不同（改权限 / 扩容 / 改提示词）。
    """

    action: RouteAction = Field(description="裁决动作")
    code: RejectCode = Field(default=RejectCode.OK, description="原因码，监控按它聚合")
    reason: str = Field(default="", description="人读的裁决说明")
    envelope: Optional[Envelope] = Field(default=None, description="最终信封；被拦下时也返回，便于留痕")
    elapsed_ms: int = Field(default=0, description="裁决耗时")

    @property
    def delivered(self) -> bool:
        return self.action == RouteAction.DELIVER

    @classmethod
    def ok(cls, envelope: Envelope, reason: str = "", elapsed_ms: int = 0) -> "RouteDecision":
        return cls(action=RouteAction.DELIVER, code=RejectCode.OK,
                   reason=reason, envelope=envelope, elapsed_ms=elapsed_ms)

    @classmethod
    def blocked(cls, action: RouteAction, code: RejectCode, reason: str,
                envelope: Optional[Envelope] = None, elapsed_ms: int = 0) -> "RouteDecision":
        return cls(action=action, code=code, reason=reason,
                   envelope=envelope, elapsed_ms=elapsed_ms)


def envelope_to_log_dict(env: Envelope, content_limit: int = 200) -> dict:
    """把信封压成一行可写日志/进 trace 的字典。

    【为什么正文要截断】
        一条消息最多 4000 字，几十条就把日志文件撑爆；
        排查时先看"谁发给谁、什么类型、什么结果"，需要全文再按 id 查总线。
    """
    return {
        "id": env.id,
        "trace_id": env.trace_id,
        "parent_id": env.parent_id,
        "sender": env.sender,
        "receiver": env.receiver,
        "kind": env.kind.value,
        "round": env.round,
        "hop": env.hop,
        "delivery": env.delivery.value,
        "reason": env.reason,
        "size_bytes": env.size_bytes,
        "content": (env.content or "")[:content_limit],
    }
