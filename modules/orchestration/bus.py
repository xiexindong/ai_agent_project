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
from __future__ import annotations

from typing import Optional

from modules.multiagent.models import MessageKind
from modules.orchestration.protocol import Envelope


class OrchestratedBus:
    """挂在 RunContext.bus 上，Agent 无感知地走中心化编排。"""

    def __init__(self, bus, orchestrator):
        self.bus = bus
        self.orchestrator = orchestrator

    # ------------------------------------------------------------------
    # 【为什么 post 的返回值是信封而不是 bool】
    #   发送方有时要拿消息 id 去构造回复（parent_id），
    #   返回信封就同时满足"知道发没发出去"（看 delivery）和"拿到 id"。
    #   被拦下时也返回信封，只是 delivery 不是 DELIVERED。
    # ------------------------------------------------------------------
    def post(self, sender: str, content: str, receiver: str = "*",
             kind: MessageKind = MessageKind.FINDING, **kw) -> Optional[Envelope]:
        decision = self.orchestrator.send(sender, content, receiver, kind, **kw)
        return decision.envelope

    def inbox_text(self, name: str, limit: int = 20) -> str:
        """读自己能看到的消息，拼成可进 prompt 的文本。"""
        return self.bus.inbox_text(name, limit)

    def inbox(self, name: str, limit: int = 20) -> list[Envelope]:
        return self.bus.inbox(name, limit)

    def objections(self, limit: int = 50) -> list[Envelope]:
        return self.bus.objections(limit)

    def all(self, limit: int = 50) -> list[Envelope]:
        return self.bus.all(limit)

    def clear(self) -> None:
        self.bus.clear()

    def __len__(self) -> int:
        return len(self.bus)
