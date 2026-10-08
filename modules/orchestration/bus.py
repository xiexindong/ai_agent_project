"""
消息总线的编排器门面
====================
Agent 手里拿到的 bus 其实是这个门面：读消息走总线，发消息一律走编排器。

【为什么要有门面而不是让 Agent 直接拿 MessageBus】
    直接给 MessageBus，Agent 就能绕过编排器往总线里塞消息——
    权限、限流、防环全部失效，中心化编排形同虚设。
    门面只暴露"读"和"经编排器写"两个能力，写路径无法绕过。

【与 MessageBus 的分工】
    MessageBus        存储与检索（谁看得到哪些消息）
    OrchestratedBus   出入口治理（能不能发、发不发得出去）
"""
from __future__ import annotations  # 注解延迟求值：允许类型标注前置引用尚未定义的名字

from typing import Optional  # 可选类型标注

from modules.multiagent.models import MessageKind  # 消息类型枚举（finding/objection/clarify/delegate）
from modules.orchestration.protocol import Envelope  # 标准化消息信封：post 的返回类型


# ---------------------------------------------------------------------------
# 【什么】消息总线的编排器门面
# 【作用】读操作原样透传给底层总线；写操作（post）强制改道编排器
# 【为什么】Agent 只能拿到这个门面：写路径无法绕过编排器的校验/限流/防环
# ---------------------------------------------------------------------------
class OrchestratedBus:  # 挂在 RunContext.bus 上，Agent 无感知地走中心化编排
    """挂在 RunContext.bus 上，Agent 无感知地走中心化编排。"""

    # 【什么】门面构造：持有底层总线与编排器两个引用
    # 【作用】读走 bus、写走 orchestrator，分工在构造时就定死
    def __init__(self, bus, orchestrator):  # bus=存储侧总线；orchestrator=治理侧编排器
        self.bus = bus  # 底层总线：所有读操作的目标
        self.orchestrator = orchestrator  # 编排器：所有写操作的必经之路

    # ------------------------------------------------------------------
    # 【为什么 post 的返回值是信封而不是 bool】
    #   发送方有时要拿消息 id 去构造回复（parent_id），
    #   返回信封就同时满足"知道发没发出去"（看 delivery）和"拿到 id"。
    #   被拦下时也返回信封，只是 delivery 不是 DELIVERED。
    # ------------------------------------------------------------------
    def post(self, sender: str, content: str, receiver: str = "*",  # 基本信息：谁发、发什么、发给谁
             kind: MessageKind = MessageKind.FINDING, **kw) -> Optional[Envelope]:  # 类型 + 透传给编排器的其余参数
        decision = self.orchestrator.send(sender, content, receiver, kind, **kw)  # 写路径强制走编排器：校验/限流/防环全在这里过
        return decision.envelope  # 只把信封还给调用方：裁决细节编排器已留痕，调用方看 delivery 便知结果

    # 【什么】读收件箱并拼成可进 prompt 的文本
    # 【作用】Agent 的"我现在能看到什么消息"入口；格式直接为拼 prompt 优化
    def inbox_text(self, name: str, limit: int = 20) -> str:  # name=收件人名；limit=最多读几条
        """读自己能看到的消息，拼成可进 prompt 的文本。"""
        return self.bus.inbox_text(name, limit)  # 纯透传：读路径不需要治理

    # 【什么】读收件箱（结构化信封列表）
    # 【作用】需要拿消息 id / parent_id 做进一步操作时用这个方法（而不是 inbox_text）
    def inbox(self, name: str, limit: int = 20) -> list[Envelope]:  # 返回信封列表
        return self.bus.inbox(name, limit)  # 纯透传

    # 【什么】读异议消息
    # 【作用】主管/汇总环节单独拉"有人反对"的消息列表用
    def objections(self, limit: int = 50) -> list[Envelope]:  # 只看 objection 类型的消息
        return self.bus.objections(limit)  # 纯透传

    # 【什么】读全部消息（不分接收者）
    # 【作用】编排器回查父消息、统计已投递数时用
    def all(self, limit: int = 50) -> list[Envelope]:  # 全量读取（带 limit 防撑爆）
        return self.bus.all(limit)  # 纯透传

    # 【什么】清空总线
    # 【作用】新运行开始前调用：不带旧运行的消息包袱
    def clear(self) -> None:  # 清空全部消息
        self.bus.clear()  # 纯透传

    # 【什么】len(bus) 语法支持
    # 【作用】len(facade) == len(底层总线)：外部代码不需要知道门面的存在
    def __len__(self) -> int:  # 消息总数
        return len(self.bus)  # 纯透传
