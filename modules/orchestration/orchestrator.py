"""
中心化编排器
============
所有 Agent 通信的唯一入口。发出去的每条消息都要经过：
    校验（权限/敏感/参数） → 限流 → 配额 → 防环 → 投递 → 上报

【为什么必须中心化】
    点对点自由通信在生产上会出三类事故：
        1. 死循环：A 问 B、B 回问 A，谁也不收口，token 一直烧
        2. 不可观测：消息散在各 Agent 的调用里，出问题时拼不出链路
        3. 无法管控：想加一条"不许外发客户手机号"的规则，得改所有 Agent
    中心化之后，这三件事各有一个唯一的落点：本文件。

【防死循环的三道闸】
    1. 跳数上限（max_hops）      一条对话链最多几跳
    2. 每轮配额（per_sender_round）  同一个人在一轮里最多发几条
    3. 重复/乒乓检测             内容重复或 A→B→A→B 模式直接升级给主管
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, Field

from modules.common.logger import get_logger
from modules.multiagent.models import MessageKind
from modules.orchestration.control import Quota, RateLimiter
from modules.orchestration.guard import GuardPolicy, PolicyGuard
from modules.orchestration.protocol import (
    DeliveryStatus,
    Envelope,
    RouteAction,
    RouteDecision,
    RejectCode,
    envelope_to_log_dict,
)
from modules.orchestration.tracer import SpanStatus, Trace, Tracer

_logger = get_logger("orchestrator")


class OrchestratorConfig(BaseModel):
    """编排器的治理参数。"""

    max_hops: int = Field(default=2, ge=0, description="一条对话链最多几跳，超出升级主管")
    repeat_limit: int = Field(default=2, ge=1,
                              description="同一对收发方的相同内容最多出现几次，超出判定为无效重复")
    pingpong_limit: int = Field(default=2, ge=1,
                                description="A→B 与 B→A 各出现几次判定为踢皮球")
    escalate_to: str = Field(default="", description="升级给谁；为空则用主管名")
    trace_enabled: bool = Field(default=True, description="是否为每条消息写观测 span")


class Orchestrator:
    """消息的唯一出入口。Agent 拿到的 bus 只是它的门面。"""

    def __init__(
        self,
        bus,
        tracer: Optional[Tracer] = None,
        guard: Optional[PolicyGuard] = None,
        limiter: Optional[RateLimiter] = None,
        quota: Optional[Quota] = None,
        config: Optional[OrchestratorConfig] = None,
        supervisor: str = "",
    ):
        self.bus = bus                                        # 消息总线（存储侧）
        self.tracer = tracer                                  # 观测（可为空=不记 span）
        self.guard = guard or PolicyGuard(GuardPolicy())
        self.limiter = limiter or RateLimiter()
        self.quota = quota or Quota(max_hops=(config.max_hops if config else 2))
        self.config = config or OrchestratorConfig()
        self.supervisor = supervisor
        # 【为什么配额的 max_hops 要与配置对齐一次】
        #   Quota 自己有 max_hops 字段，构造时若没传就用自己的默认 2。
        #   这里显式同步一次，避免"改了 config 但配额没跟上"的静默错位。
        self.quota.max_hops = self.config.max_hops

        self.round_no: int = 0                     # 当前协作轮次
        self.trace: Optional[Trace] = None         # 当前运行绑定的观测链路
        self.escalations: list[dict] = []          # 被判定为无效交互的记录，供主管/人工查看
        # 内容去重：{("发送者","接收者","内容"): 次数}
        self._content_hits: dict[tuple[str, str, str], int] = defaultdict(int)
        # 方向计数：{("发送者","接收者"): 次数}（不分轮次，专抓乒乓）
        self._pair_hits: dict[tuple[str, str], int] = defaultdict(int)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def bind_trace(self, trace: Optional[Trace]) -> None:
        """绑定本次运行的观测链路，之后每条消息都挂到它的根 span 下。"""
        self.trace = trace

    def new_round(self) -> int:
        """进入下一轮协作，返回新的轮次号。

        【为什么要有轮次概念】
            上一轮的异议、求助如果不随轮次失效，新一批 Agent 开工时
            会读到与当前任务无关的旧消息——表现为"莫名其妙被批评"。
        """
        self.round_no += 1
        return self.round_no

    def reset(self) -> None:
        """开始一次新运行：清空计数与升级记录。"""
        self.round_no = 0
        self.escalations.clear()
        self._content_hits.clear()
        self._pair_hits.clear()
        self.quota.reset_run()
        self.trace = None

    # ------------------------------------------------------------------
    # 核心：发一条消息
    # ------------------------------------------------------------------
    def send(
        self,
        sender: str,
        content: str,
        receiver: str = "*",
        kind: MessageKind = MessageKind.FINDING,
        *,
        sender_role: str = "",
        parent_id: str = "",
        ref_task_id: str = "",
        meta: Optional[dict] = None,
        run_id: str = "",
    ) -> RouteDecision:
        """发一条消息，返回编排器的裁决。

        【返回值为什么必须是 RouteDecision 而不是 bool】
            调用方需要区分"被权限拒了"（要去申请权限）和"被限流丢了"
            （等一会儿重试即可）。布尔值把这两种情况压成一个 False，
            线上就只能靠猜。
        """
        t0 = time.time()
        trace = self.trace
        parent = self._find(parent_id)

        env = Envelope(
            sender=sender, receiver=receiver, kind=kind, content=content or "",
            sender_role=sender_role, ref_task_id=ref_task_id, meta=meta or {},
            round=self.round_no,
            # 【hop 怎么算】回复别人的消息才算一跳；主动发起的消息是第 0 跳。
            hop=(parent.hop + 1) if parent is not None else 0,
            parent_id=parent.id if parent is not None else "",
            trace_id=trace.trace_id if trace else "",
            run_id=run_id,
        )

        # ---- 第 1 关：权限 / 敏感词 / 参数 --------------------------------
        verdict = self.guard.check(env)
        if not verdict.passed:
            env.mark(DeliveryStatus.REJECTED, verdict.reason)
            self._record(env, verdict.code, SpanStatus.ERROR)
            return RouteDecision.blocked(RouteAction.REJECT, verdict.code, verdict.reason,
                                         env, self._ms(t0))

        # ---- 第 2 关：限流 ------------------------------------------------
        ok, by = self.limiter.acquire(sender)
        if not ok:
            env.mark(DeliveryStatus.DROPPED, f"限流（{by}）")
            self._record(env, RejectCode.RATE_LIMITED, SpanStatus.TIMEOUT)
            return RouteDecision.blocked(RouteAction.DROP, RejectCode.RATE_LIMITED,
                                         f"被 {by} 限流，消息未投递", env, self._ms(t0))

        # ---- 第 3 关：配额与跳数 ------------------------------------------
        allowed, why = self.quota.check(sender, env.round, env.hop)
        if not allowed:
            # 【跳数超限为什么要升级而不是丢弃】
            #   跳数超限意味着"两个人已经来回好几轮还没结论"，
            #   这是个需要人/主管知道的信号；丢掉就永远发现不了这对组合有问题。
            if env.hop > self.config.max_hops:
                env.mark(DeliveryStatus.ESCALATED, why)
                self._escalate(env, RejectCode.MAX_HOPS, why)
                return RouteDecision.blocked(RouteAction.ESCALATE, RejectCode.MAX_HOPS,
                                             why, env, self._ms(t0))
            env.mark(DeliveryStatus.DROPPED, why)
            self._record(env, RejectCode.QUOTA_EXCEEDED, SpanStatus.TIMEOUT)
            return RouteDecision.blocked(RouteAction.DROP, RejectCode.QUOTA_EXCEEDED,
                                         why, env, self._ms(t0))

        # ---- 第 4 关：防死循环（重复 / 乒乓）------------------------------
        loop = self._detect_loop(env)
        if loop is not None:
            env.mark(DeliveryStatus.ESCALATED, loop)
            self._escalate(env, RejectCode.LOOP_DETECTED, loop)
            return RouteDecision.blocked(RouteAction.ESCALATE, RejectCode.LOOP_DETECTED,
                                         loop, env, self._ms(t0))

        # ---- 投递 --------------------------------------------------------
        env.mark(DeliveryStatus.DELIVERED)
        self.bus.post_envelope(env)
        self.quota.consume(sender, env.round)
        self._content_hits[(sender, env.receiver, content or "")] += 1
        self._pair_hits[(sender, env.receiver)] += 1
        self._record(env, RejectCode.OK, SpanStatus.OK)
        _logger.debug(f"消息已投递: {sender} → {env.receiver} [{kind.value}] hop={env.hop}")
        return RouteDecision.ok(env, "已投递", self._ms(t0))

    # ------------------------------------------------------------------
    # 防死循环
    # ------------------------------------------------------------------
    def _detect_loop(self, env: Envelope) -> Optional[str]:
        """判定这条消息是否属于无效来回。返回 None 表示正常。"""
        key = (env.sender, env.receiver, env.content)
        if self._content_hits[key] >= self.config.repeat_limit:
            return (f"{env.sender} 重复发送相同内容给 {env.receiver} "
                    f"（第 {self._content_hits[key] + 1} 次），判定为无效重复")

        # 【乒乓怎么判】
        #   同一对收发方，正反两个方向都发过至少 pingpong_limit 次，
        #   说明他们在互相推而不是推进任务。这不是"错"，是"卡住了"，
        #   所以要升级给主管收口，而不是简单地不让发。
        a2b = self._pair_hits.get((env.sender, env.receiver), 0)
        b2a = self._pair_hits.get((env.receiver, env.sender), 0)
        if env.receiver != "*" and a2b >= self.config.pingpong_limit and b2a >= self.config.pingpong_limit:
            return (f"{env.sender} 与 {env.receiver} 已来回 {a2b}/{b2a} 次，"
                    f"判定为踢皮球，升级给主管")
        return None

    def _escalate(self, env: Envelope, code: RejectCode, why: str) -> None:
        """把无效交互交给主管，而不是投给原接收方。"""
        target = self.config.escalate_to or self.supervisor or "supervisor"
        item = {**envelope_to_log_dict(env), "code": code.value, "escalate_to": target}
        self.escalations.append(item)
        env.receiver = target          # 改投主管：消息不丢，但换了个收口的人
        self.bus.post_envelope(env)
        self.quota.consume(env.sender, env.round)
        _logger.warning(f"消息升级给 {target}: {why}")

    # ------------------------------------------------------------------
    # 观测与工具
    # ------------------------------------------------------------------
    def _record(self, env: Envelope, code: RejectCode, status: SpanStatus) -> None:
        """把这次收发写进观测链路。"""
        if self.tracer is None or self.trace is None or not self.config.trace_enabled:
            return
        span = self.tracer.start_span(
            self.trace,
            name=f"msg.{env.kind.value}.{env.sender}→{env.receiver}",
            kind="message",
            parent_id=self.trace.root_id or "",
        )
        span.inputs.update(envelope_to_log_dict(env, content_limit=120))
        self.tracer.end_span(
            span, status=status,
            outputs={"delivery": env.delivery.value, "code": code.value,
                     "hop": env.hop, "round": env.round,
                     "size_bytes": env.size_bytes},
            error=env.reason if status == SpanStatus.ERROR else "",
        )

    def _find(self, msg_id: str) -> Optional[Envelope]:
        if not msg_id:
            return None
        for m in self.bus.all(limit=200):
            if m.id == msg_id:
                return m
        return None

    @staticmethod
    def _ms(t0: float) -> int:
        return int((time.time() - t0) * 1000)

    def stats(self) -> dict:
        """编排器的运行统计：给监控大盘与排障用。"""
        return {
            "round": self.round_no,
            "delivered": len([1 for m in self.bus.all(limit=500)
                              if m.delivery == DeliveryStatus.DELIVERED]),
            "escalated": len(self.escalations),
            "escalations": self.escalations[-10:],
            "quota": self.quota.stats(),
            "limiter": self.limiter.stats(),
        }
