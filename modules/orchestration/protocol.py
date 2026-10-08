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
from __future__ import annotations  # 注解延迟求值：允许类型标注前置引用尚未定义的名字（如返回 "Envelope"）

from enum import Enum  # 枚举基类：DeliveryStatus/RouteAction/RejectCode 用
from typing import Any, Optional  # 类型标注：任意值/可选

from pydantic import BaseModel, Field  # 数据模型基类与字段声明：Envelope/RouteDecision 都是 pydantic 模型

from modules.multiagent.models import AgentMessage, MessageKind  # 基础消息模型与消息类型枚举

# 【协议版本号】
#   进 trace 与落库。将来改字段语义时，靠它区分"老格式的数据"和"新格式的数据"，
#   否则回溯半年前的链路会发现字段含义对不上，但看不出是哪一版写的。
PROTOCOL_VERSION = "1.0"  # 当前协议版本：Envelope.protocol 字段的默认值


# ---------------------------------------------------------------------------
# 【什么】一条消息最终落到什么状态
# 【作用】溯源时先问"送到了吗"，再问"为什么"
#   PENDING   刚建出来，还没过编排器
#   DELIVERED 通过全部校验，已进总线
#   REJECTED  被校验拦下（权限/敏感词/参数），调用方需要知道"被拒了"
#   DROPPED   被负载控制丢弃（限流/配额），不通知调用方，只留痕
#   ESCALATED 判定为无效交互（踢皮球/超跳数），已升级给主管而非送达对方
# ---------------------------------------------------------------------------
class DeliveryStatus(str, Enum):  # str mixin：序列化直接成字符串
    PENDING = "pending"  # 刚建出来，还没过编排器
    DELIVERED = "delivered"  # 通过全部校验，已进总线
    REJECTED = "rejected"  # 被校验拦下（权限/敏感词/参数）
    DROPPED = "dropped"  # 被负载控制丢弃（限流/配额）
    ESCALATED = "escalated"  # 判定为无效交互，已升级给主管


# ---------------------------------------------------------------------------
# 【什么】编排器对一条消息作出的裁决动作
# 【作用】裁决的"动词"部分：送出去 / 拒掉 / 丢掉 / 升级
# 【为什么与 DeliveryStatus 分开】DeliveryStatus 是信封的状态（结果），
#   RouteAction 是编排器的动作（决策）；一个动作决定一个状态，但语义层级不同
# ---------------------------------------------------------------------------
class RouteAction(str, Enum):
    """编排器对一条消息作出的裁决动作。"""

    DELIVER = "deliver"      # 正常投递
    REJECT = "reject"        # 校验不通过，明确拒绝（调用方可见）
    DROP = "drop"            # 负载原因静默丢弃（只留痕，不打断发送方）
    ESCALATE = "escalate"    # 判定为空转，交给主管/人工，不投给原接收方


# ---------------------------------------------------------------------------
# 【什么】拒绝/丢弃原因码
# 【作用】监控按它聚合，不解析中文文案；文案可以改，码不能改
# 【为什么全是机器可读的短码】告警规则、大盘聚合都按码走，按文案走一改文案就崩
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 【什么】标准化的通信信封
# 【作用】所有 Agent 通信的唯一载体：基本信息 + 治理字段 + 观测字段全在这一个对象上
# 【为什么继承 AgentMessage】旧代码不用改就能读到更全的信息（见模块 docstring）
# ---------------------------------------------------------------------------
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

    trace_id: str = Field(default="", description="所属观测链路 ID")  # 挂到哪条 trace：同一次运行的所有消息同源
    run_id: str = Field(default="", description="所属运行 ID")  # 挂到哪次运行：与 RunResult 对上号
    parent_id: str = Field(default="", description="回复的上一条消息 id；顶级消息为空")  # 对话链的边
    round: int = Field(default=0, ge=0, description="协作轮次，从 0 开始")  # 第几轮：配额按它记"每轮账"
    hop: int = Field(default=0, ge=0, description="第几跳，回复产生的消息 = 父消息 hop + 1")  # 第几跳：防死循环用
    sender_role: str = Field(default="", description="发送者角色，供权限矩阵判定")  # 角色：权限矩阵按它判定
    priority: int = Field(default=0, description="数值越大越优先，仅用于排序")  # 优先级：目前只排序用
    delivery: DeliveryStatus = Field(default=DeliveryStatus.PENDING, description="最终投递状态")  # 投递状态：出生即 PENDING
    reason: str = Field(default="", description="裁决理由，被拦下时必有值")  # 人读的裁决理由
    meta: dict[str, Any] = Field(default_factory=dict, description="结构化附加数据")  # 自由扩展位
    protocol: str = Field(default=PROTOCOL_VERSION, description="写入时使用的协议版本")  # 协议版本：写入时打上

    # 【为什么 size_bytes 做成计算属性而不是字段】
    #   正文可能被敏感词打码改写，存成字段就会和 content 不一致；
    #   实时算永远是对的。参数校验读它时也不用担心"还没算过"。
    @property
    def size_bytes(self) -> int:  # 正文长度（字符数；名字叫 bytes 是历史遗留，语义是字符数）
        return len(self.content or "")  # None 正文按 0 算

    # 【什么】只读属性：是否已送达
    # 【作用】调用方判断"发出去没"的便捷写法：env.is_delivered
    @property
    def is_delivered(self) -> bool:  # True = 已进总线
        return self.delivery == DeliveryStatus.DELIVERED  # 只有 DELIVERED 才算送达

    # 【什么】原地标记状态并回填理由
    # 【作用】编排器每过一关都要改状态；返回自身支持链式：env.mark(...).xxx
    def mark(self, status: DeliveryStatus, reason: str = "") -> "Envelope":
        """原地标记状态并回填理由，返回自身便于链式使用。"""
        self.delivery = status  # 改状态
        if reason:  # 传了理由才回填：没理由不动旧值
            self.reason = reason  # 回填理由
        return self  # 返回自身：链式

    # 【什么】基于本条消息构造一条回复
    # 【作用】回复时不用手填 trace_id/run_id/round/hop：全部从本条继承
    # 【为什么 hop 在这里 +1】协议层统一 +1，调用方漏填的口子被堵死（见 docstring）
    def reply(self, sender: str, content: str, kind: MessageKind = MessageKind.CLARIFY,
              **kw) -> "Envelope":
        """基于本条消息构造一条回复。

        【为什么 hop 由协议负责 +1 而不是让调用方传】
            让调用方自己填 hop，等于把"防死循环"的成败交给了每个写消息的人，
            漏填一次就回到 0，上限形同虚设。协议里 +1 就没这个口子。
        """
        return Envelope(
            sender=sender,  # 回复的发送者 = 调用方指定的名字
            receiver=self.sender,          # 默认回给发信方
            kind=kind,  # 默认 CLARIFY（澄清/追问）：回复的最常见类型
            content=content,  # 回复正文
            trace_id=self.trace_id,  # 继承观测链路
            run_id=self.run_id,  # 继承运行 id
            parent_id=self.id,  # 父消息就是本条：对话链接上
            round=self.round,  # 继承轮次：回复不换轮
            hop=self.hop + 1,  # 跳数 +1：协议层统一，调用方漏填也没事
            **kw,  # 其余字段（sender_role/meta/...）由调用方透传
        )


# ---------------------------------------------------------------------------
# 【什么】编排器对一条消息作出的裁决
# 【作用】send 的唯一返回类型：动作 + 原因码 + 人读理由 + 最终信封 + 耗时
# 【为什么建模成 pydantic】裁决要进日志/进 API 响应，序列化必须是一行 model_dump 的事
# ---------------------------------------------------------------------------
class RouteDecision(BaseModel):
    """编排器对一条消息作出的裁决。

    【为什么要把"裁决"也建模】
        只返回 bool 的话，线上只能看到"消息没送到"，
        看不出是被权限拦了、被限流丢了、还是被判定为踢皮球 —— 这三类
        排障动作完全不同（改权限 / 扩容 / 改提示词）。
    """

    action: RouteAction = Field(description="裁决动作")  # 动作：deliver/reject/drop/escalate
    code: RejectCode = Field(default=RejectCode.OK, description="原因码，监控按它聚合")  # 机器读的原因码
    reason: str = Field(default="", description="人读的裁决说明")  # 人读的理由
    envelope: Optional[Envelope] = Field(default=None, description="最终信封；被拦下时也返回，便于留痕")  # 最终信封
    elapsed_ms: int = Field(default=0, description="裁决耗时")  # 裁决耗时：监控看"编排器本身慢不慢"

    # 【什么】只读属性：是否送达
    # 【作用】调用方判断"成了没"的便捷写法：decision.delivered
    @property
    def delivered(self) -> bool:  # True = 动作是 DELIVER
        return self.action == RouteAction.DELIVER  # 只有 DELIVER 才算成

    # 【什么】构造"送达"裁决的工厂
    # 【作用】成功路径的固定五元组：DELIVER + OK + 理由 + 信封 + 耗时
    @classmethod
    def ok(cls, envelope: Envelope, reason: str = "", elapsed_ms: int = 0) -> "RouteDecision":  # 工厂：送达裁决
        return cls(action=RouteAction.DELIVER, code=RejectCode.OK,
                   reason=reason, envelope=envelope, elapsed_ms=elapsed_ms)  # 固定动作与原因码，其余参数透传

    # 【什么】构造"拦下"裁决的工厂
    # 【作用】三种拦法（reject/drop/escalate）共用一个入口：动作由调用方指定
    @classmethod
    def blocked(cls, action: RouteAction, code: RejectCode, reason: str,
                envelope: Optional[Envelope] = None, elapsed_ms: int = 0) -> "RouteDecision":  # 工厂：拦下裁决
        return cls(action=action, code=code, reason=reason,
                   envelope=envelope, elapsed_ms=elapsed_ms)  # 五个字段全由调用方定：三种拦法只差 action


# ---------------------------------------------------------------------------
# 【什么】信封压平成日志字典
# 【作用】写日志/进 trace 的统一格式：一行一个信封，字段固定
# 【为什么正文要截断】见 docstring
# ---------------------------------------------------------------------------
def envelope_to_log_dict(env: Envelope, content_limit: int = 200) -> dict:
    """把信封压成一行可写日志/进 trace 的字典。

    【为什么正文要截断】
        一条消息最多 4000 字，几十条就把日志文件撑爆；
        排查时先看"谁发给谁、什么类型、什么结果"，需要全文再按 id 查总线。
    """
    return {
        "id": env.id,  # 消息 id：按它回查总线
        "trace_id": env.trace_id,  # 链路 id：按它拉整棵树
        "parent_id": env.parent_id,  # 父消息 id：拼对话链
        "sender": env.sender,  # 发送者
        "receiver": env.receiver,  # 接收者（升级后可能是主管）
        "kind": env.kind.value,  # 消息类型（枚举取值成字符串）
        "round": env.round,  # 轮次
        "hop": env.hop,  # 跳数
        "delivery": env.delivery.value,  # 投递状态（枚举取值成字符串）
        "reason": env.reason,  # 裁决理由
        "size_bytes": env.size_bytes,  # 正文长度
        "content": (env.content or "")[:content_limit],  # 正文截断：None 兜底成空串再截
    }
