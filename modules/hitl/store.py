"""
中断请求存储模块
================
管「请求发出去了、回复回来了、两边怎么碰上面」。

【为什么存储和判定要分开】
    store 只负责存取与等待唤醒，不关心"该不该中断""回复了之后怎么办"。
    分开之后可以换实现：
        InMemoryInterruptStore  进程内，demo / 单元测试 / 单实例部署
        （未来）PGInterruptStore  复用 sys_review_queue 落库，多实例共享审批队列
    Agent 侧代码一行都不用改。

【阻塞等待为什么用 Event 而不是轮询】
    轮询（每 200ms 查一次 status）有两个代价：
        1. 白占一个线程，并行度高的团队会把线程池吃光
        2. 人点了批准，最坏要等一个轮询周期才生效，体感是"点了没反应"
    Event.wait(timeout) 是内核级等待：人一提交立刻唤醒，超时也能准时兜底。

【依赖警告】
    本实现是进程内的。多副本部署时，抛中断的实例和提交回复的实例
    可能不是同一个（审批请求打到另一台机器），此时必须换成共享存储，
    否则会出现"人点了批准，但那个实例永远等不到"的挂起。
"""
from __future__ import annotations

import threading
from typing import Optional, Protocol

from modules.common.logger import get_logger
from modules.hitl.models import (
    HumanReply,
    InterruptRecord,
    InterruptRequest,
    InterruptStatus,
    now_ms,
)

_logger = get_logger("hitl_store")


class InterruptStore(Protocol):
    """存储能力契约。换实现只需满足这五个方法。"""

    def save(self, req: InterruptRequest) -> InterruptRequest: ...
    def get(self, request_id: str) -> Optional[InterruptRequest]: ...
    def pending(self, limit: int = 50) -> list[InterruptRequest]: ...
    def submit(self, reply: HumanReply) -> bool: ...
    def wait(self, request_id: str, timeout_seconds: float) -> Optional[HumanReply]: ...


class InMemoryInterruptStore:
    """进程内存储：字典存请求、Event 做等待唤醒、锁保线程安全。"""

    def __init__(self, max_records: int = 500):
        self._reqs: dict[str, InterruptRequest] = {}
        self._replies: dict[str, HumanReply] = {}
        self._events: dict[str, threading.Event] = {}
        self._records: list[InterruptRecord] = []
        # 【为什么要有上限】审批记录只进不出会撑爆内存。
        #   超上限丢最旧的：最近的请求才是排查时关心的。
        self._max_records = max(10, int(max_records))
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 【写】登记一次中断请求，同时给它配一个 Event
    #   预热 Event 而不是等到 wait 时才建，避免"人先回复、Agent 后等待"的
    #   竞态：那样 wait 会建一个新 Event，永远等不到已提交的回复。
    # ------------------------------------------------------------------
    def save(self, req: InterruptRequest) -> InterruptRequest:
        with self._lock:
            self._reqs[req.id] = req
            self._events.setdefault(req.id, threading.Event())
            self._records.append(InterruptRecord(request=req))
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]
            _logger.info(f"中断请求已登记: {req.id} kind={req.kind.value} title={req.title[:40]}")
            return req

    def get(self, request_id: str) -> Optional[InterruptRequest]:
        with self._lock:
            return self._reqs.get(request_id)

    # ------------------------------------------------------------------
    # 【读】待处理清单：审批页轮询它
    #   只返回 WAITING 的；已回复/已超时的不再出现，
    #   否则审批页会一直显示早就处理过的卡片。
    # ------------------------------------------------------------------
    def pending(self, limit: int = 50) -> list[InterruptRequest]:
        with self._lock:
            items = [r for r in self._reqs.values() if r.status == InterruptStatus.WAITING]
            items.sort(key=lambda r: r.created_at)
            return items[: max(1, int(limit))]

    # ------------------------------------------------------------------
    # 【人这一侧】提交回复：存下来 → 置状态 → 唤醒等待中的 Agent
    #   返回 False 的三种情况：请求不存在 / 已不是 WAITING（重复提交或已超时）
    #   重复提交要挡住：网络重试可能把同一次点击发两遍，
    #   第二遍若覆盖第一遍，审计上就查不出真正生效的是哪次决策。
    # ------------------------------------------------------------------
    def submit(self, reply: HumanReply) -> bool:
        with self._lock:
            req = self._reqs.get(reply.request_id)
            if req is None:
                _logger.warning(f"回复对应的请求不存在: {reply.request_id}")
                return False
            if req.status != InterruptStatus.WAITING:
                _logger.warning(f"请求已结束，忽略重复回复: {reply.request_id} status={req.status.value}")
                return False
            req.status = InterruptStatus.RESOLVED
            self._replies[reply.request_id] = reply
            for rec in self._records:
                if rec.request.id == reply.request_id:
                    rec.reply = reply
                    rec.resolved_at = now_ms()
                    break
            self._events.setdefault(reply.request_id, threading.Event()).set()
            _logger.info(
                f"人工回复已生效: {reply.request_id} action={reply.action.value} "
                f"operator={reply.operator or '-'}"
            )
            return True

    # ------------------------------------------------------------------
    # 【Agent 这一侧】阻塞等待：人来就返回回复，超时返回 None
    #   返回 None 表示"没人来"，由调用方（HumanGate）按 default_action 兜底，
    #   并把状态置为 EXPIRED —— 这里不替调用方决定兜底动作，
    #   因为不同中断点的兜底策略不同（工具调用默认拒绝，出答案默认放行）。
    # ------------------------------------------------------------------
    def wait(self, request_id: str, timeout_seconds: float) -> Optional[HumanReply]:
        event = self._events.get(request_id)
        if event is None:
            return None
        arrived = event.wait(timeout=max(0.0, float(timeout_seconds)))
        with self._lock:
            reply = self._replies.get(request_id)
        if reply is not None:
            return reply
        # 【为什么这里不改状态为 EXPIRED】
        #   兜底动作由 HumanGate 决定（不同中断点默认动作不同），
        #   store 只报告"没人来"。若在这里先置 EXPIRED，
        #   gate 随后提交的那条超时回复就会被 submit 以"已结束"为由拒收，
        #   审计记录里就只剩"超时"而看不到"最终按什么动作执行了"。
        if not arrived:
            _logger.warning(f"中断请求超时: {request_id}（{timeout_seconds}s 无人处理）")
        return None

    # ------------------------------------------------------------------
    # 【归档】完整留痕：谁在什么时候、为什么被打断、人怎么回的
    # ------------------------------------------------------------------
    def records(self) -> list[InterruptRecord]:
        with self._lock:
            return list(self._records)

    def cancel(self, request_id: str) -> bool:
        """流程提前结束时作废一个还在等的请求（并唤醒等待方，避免线程挂住）。"""
        with self._lock:
            req = self._reqs.get(request_id)
            if req is None or req.status != InterruptStatus.WAITING:
                return False
            req.status = InterruptStatus.CANCELLED
            self._events.setdefault(request_id, threading.Event()).set()
            return True
