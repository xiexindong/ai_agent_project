"""
负载控制：限流、配额、重试、超时
================================
全部放在编排层，Agent 自己不关心也做不到这些事。

【为什么负载控制必须在编排层而不是 Agent 内部】
    放在 Agent 内部会出现三个问题：
        1. 每个 Agent 各限各的，全局总量仍然打满下游（4 个 Agent × 各自 10 QPS）
        2. 重试放在 Agent 里，失败一次 Agent 自己就重试，
           编排器看到的仍是"一次调用"，超时预算被悄悄吃掉
        3. 限流开关要改时得去改每个 Agent
    统一在编排层：全局一桶令牌、全局一份配额、重试次数计入总预算。
"""
from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional, TypeVar

from modules.common.logger import get_logger

_logger = get_logger("control")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# 【什么】令牌桶限流器
# 【作用】全局与「按发送者」两级限流，桶里没令牌就拒绝
#
#   【为什么用令牌桶而不是简单计数窗口】
#       计数窗口在窗口切换的瞬间会放行两倍流量（窗口末尾 + 窗口开头各打满），
#       下游就在这一刻被冲垮。令牌桶按速率持续补充，不存在这个尖峰。
#
#   【输入 → 输出对照】
#       bucket = TokenBucket(capacity=3, refill_per_sec=1.0)
#       [bucket.try_consume() for _ in range(4)]
#       输出: [True, True, True, False]     第 4 次桶已空
#       time.sleep(1.0); bucket.try_consume()
#       输出: True                          1 秒后补回 1 个令牌
# ---------------------------------------------------------------------------
class TokenBucket:
    def __init__(self, capacity: int = 40, refill_per_sec: float = 20.0):
        self.capacity = max(1, int(capacity))                 # 桶容量 = 允许的瞬时突发上限
        self.refill_per_sec = max(0.0, float(refill_per_sec))  # 每秒补充的令牌数 = 长期平均速率
        self._tokens = float(self.capacity)
        self._last = time.time()
        self._lock = threading.Lock()

    def try_consume(self, n: int = 1) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self._last
        if elapsed <= 0:
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
        self._last = now

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


class RateLimiter:
    """全局桶 + 每发送者桶的双层限流。

    【为什么要双层】
        只有全局桶：一个失控的 Agent 能把整队令牌吃光，其他人全被饿死。
        只有 per-sender 桶：总量仍然可能超过下游承载。
        双层 = 总量可控 + 单体不会霸占。
    """

    def __init__(self, capacity: int = 40, refill_per_sec: float = 20.0,
                 per_sender_capacity: int = 10, per_sender_refill: float = 5.0):
        self.global_bucket = TokenBucket(capacity, refill_per_sec)
        self._per_capacity = max(1, int(per_sender_capacity))
        self._per_refill = max(0.0, float(per_sender_refill))
        self._senders: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _bucket_for(self, sender: str) -> TokenBucket:
        with self._lock:
            b = self._senders.get(sender)
            if b is None:
                b = TokenBucket(self._per_capacity, self._per_refill)
                self._senders[sender] = b
            return b

    def acquire(self, sender: str = "", n: int = 1) -> tuple[bool, str]:
        """取令牌。返回 (是否放行, 被谁拦下)。"""
        if not self.global_bucket.try_consume(n):
            return False, "global"
        if sender:
            if not self._bucket_for(sender).try_consume(n):
                # 【为什么 per-sender 拦下后不把全局令牌还回去】
                #   还回去等于"被限流的发送者可以无限重试直到把全局桶也耗光"；
                #   不还，它这一轮的失败就会计入全局消耗，形成自然背压。
                return False, f"sender:{sender}"
        return True, ""

    def stats(self) -> dict:
        return {
            "global_available": round(self.global_bucket.available, 2),
            "senders": {k: round(v.available, 2) for k, v in self._senders.items()},
        }


# ---------------------------------------------------------------------------
# 【什么】消息配额
# 【作用】在"速率"之外再卡"总量"，防止长期慢速刷消息把上下文拖垮
#
#   【三道上限各自防什么】
#       per_run        整次运行最多多少条（防总量失控）
#       per_round      一轮最多多少条（防单轮爆炸）
#       per_sender_round  同一发送者一轮最多几条（防某个 Agent 话痨刷屏）
#       max_hops       一条对话链最多几跳（防 A→B→A→B 无限来回）
# ---------------------------------------------------------------------------
class Quota:
    def __init__(self, per_run: int = 100, per_round: int = 30,
                 per_sender_round: int = 3, max_hops: int = 2):
        self.per_run = max(1, int(per_run))
        self.per_round = max(1, int(per_round))
        self.per_sender_round = max(1, int(per_sender_round))
        self.max_hops = max(0, int(max_hops))
        self._total = 0
        self._round_counts: dict[int, int] = {}
        self._sender_counts: dict[tuple[int, str], int] = {}
        self._lock = threading.Lock()

    def check(self, sender: str, round_no: int, hop: int) -> tuple[bool, str]:
        with self._lock:
            if self._total >= self.per_run:
                return False, f"已达运行总量上限 {self.per_run}"
            if self._round_counts.get(round_no, 0) >= self.per_round:
                return False, f"第 {round_no} 轮已达上限 {self.per_round}"
            if self._sender_counts.get((round_no, sender), 0) >= self.per_sender_round:
                return False, (f"{sender} 在第 {round_no} 轮已达上限 "
                               f"{self.per_sender_round}")
            if hop > self.max_hops:
                return False, f"跳数 {hop} 超过上限 {self.max_hops}"
            return True, ""

    def consume(self, sender: str, round_no: int) -> None:
        with self._lock:
            self._total += 1
            self._round_counts[round_no] = self._round_counts.get(round_no, 0) + 1
            key = (round_no, sender)
            self._sender_counts[key] = self._sender_counts.get(key, 0) + 1

    def reset_run(self) -> None:
        with self._lock:
            self._total = 0
            self._round_counts.clear()
            self._sender_counts.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"total": self._total,
                    "by_round": dict(self._round_counts),
                    "limits": {"per_run": self.per_run, "per_round": self.per_round,
                               "per_sender_round": self.per_sender_round,
                               "max_hops": self.max_hops}}


def retry_with_backoff(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple = (Exception,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> T:
    """指数退避重试。

    【为什么退避要加随机抖动】
        固定间隔重试会让多个并行任务在同一秒一起重试，
        形成"重试风暴"——本来只是抖一下，被重试打成了真的过载。
        加 ±25% 抖动把重试时刻打散。

    【输入 → 输出对照】
        attempts=3, base_delay=0.5
        第 1 次失败 → 等 ~0.5s
        第 2 次失败 → 等 ~1.0s
        第 3 次失败 → 抛出最后一次异常（不再重试）
    """
    last: BaseException = RuntimeError("未执行")
    for i in range(1, max(1, int(attempts)) + 1):
        try:
            return fn()
        except retry_on as e:                       # type: ignore[misc]
            last = e
            if i >= attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (i - 1)))
            delay *= random.uniform(0.75, 1.25)     # 抖动：打散并行重试时刻
            _logger.warning(f"重试 {i}/{attempts - 1} 次后仍失败，{delay:.2f}s 后重试: {e}")
            if on_retry:
                try:
                    on_retry(i, e)
                except Exception as hook_err:
                    _logger.debug(f"重试回调异常（忽略）: {hook_err}")
            time.sleep(delay)
    raise last


def remaining_seconds(deadline: float) -> float:
    """距绝对截止时刻还剩几秒。用于把总超时传导到每个下游调用。"""
    return max(0.0, deadline - time.time())
