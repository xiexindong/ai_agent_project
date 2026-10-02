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
    # ------------------------------------------------------------------
    # 【什么】令牌桶的构造：容量 + 回血速率 + 记账三件套
    # 【作用】capacity 管突发上限、refill_per_sec 管长期平均速率；
    #        _tokens/_last 是记账（当前令牌数/上次补水时刻），_lock 防并发扣减错乱
    # 【输入输出对照】TokenBucket(capacity=3, refill_per_sec=1.0)：
    #   启动时 _tokens=3.0（满桶），_last=当前时间
    # ------------------------------------------------------------------
    def __init__(self, capacity: int = 40, refill_per_sec: float = 20.0):
        self.capacity = max(1, int(capacity))                 # 桶容量 = 允许的瞬时突发上限
        self.refill_per_sec = max(0.0, float(refill_per_sec))  # 每秒补充的令牌数 = 长期平均速率
        self._tokens = float(self.capacity)                   # 当前令牌数：出生即满桶
        self._last = time.time()                              # 上次补水时刻：第一次补水从构造时算起
        self._lock = threading.Lock()                         # 多线程同时扣令牌要排队，不然两个线程都看到"还有 1 个"

    # ------------------------------------------------------------------
    # 【什么】取 n 个令牌：够就扣掉返回 True，不够返回 False
    # 【作用】每次调用前先问一句"桶里还有吗"；False 由上层翻译成"限流"报给调用方
    # 【关键点】先 _refill 再比较：令牌不是定时器在补，是"取的时候顺手补"——
    #   上次取到现在过了多久，就按速率补多少，省掉一个后台线程
    # 【输入输出对照】capacity=3, refill=1.0：
    #   try_consume() → True（3→2）→ True（2→1）→ True（1→0）→ False（桶空）
    # ------------------------------------------------------------------
    def try_consume(self, n: int = 1) -> bool:
        with self._lock:                                      # 拿锁：补令牌 + 比大小 + 扣减三步必须一气呵成
            self._refill()                                    # 先按流逝的时间补令牌（见 _refill 注释）
            if self._tokens >= n:                             # 够 n 个才放行
                self._tokens -= n                             # 扣掉 n 个（n=1 是常态）
                return True
            return False                                      # 不够：一个都不扣，下次再来（不存在"扣一半"）

    # ------------------------------------------------------------------
    # 【什么】按流逝时间补令牌（懒补水：只在被访问时才补）
    # 【作用】elapsed × refill_per_sec = 这段时间该补的量；min 封顶防溢出
    # 【为什么要 min 封顶】不封顶的话，空闲一小时回来能补出
    #   1小时×20/s = 72000 个令牌，容量上限就失去意义（突发限制被架空）
    # 【输入输出对照】capacity=3, refill=1.0，两次访问间隔 2.5 秒：
    #   elapsed=2.5 → 补 2.5 个 → _tokens = min(3, 原+2.5)，永远不超 3
    # ------------------------------------------------------------------
    def _refill(self) -> None:
        now = time.time()                                     # 当前时刻
        elapsed = now - self._last                            # 距上次补水过了几秒
        if elapsed <= 0:                                      # 系统时钟回拨或同秒连调：补 0，避免负数令牌
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)  # 补量 = 时间 × 速率，封顶在容量
        self._last = now                                      # 记账：下次补水从现在算起

    # ------------------------------------------------------------------
    # 【什么】只看还剩几个令牌，不扣（监控用）
    # 【作用】给 stats/日志读水位；取值前也要先补水，不然读到的是过期数字
    # ------------------------------------------------------------------
    @property
    def available(self) -> float:
        with self._lock:
            self._refill()                                    # 先补再报：报出去的数字才是"此刻"的
            return self._tokens


class RateLimiter:
    """全局桶 + 每发送者桶的双层限流。

    【为什么要双层】
        只有全局桶：一个失控的 Agent 能把整队令牌吃光，其他人全被饿死。
        只有 per-sender 桶：总量仍然可能超过下游承载。
        双层 = 总量可控 + 单体不会霸占。

    【输入 → 输出对照】全局桶容量 5、每人桶容量 2：
        acquire("A", 1) → (True,  "")             A 第 1 次：全局 5→4，A 的桶 2→1
        acquire("A", 1) → (True,  "")             A 第 2 次：全局 4→3，A 的桶 1→0
        acquire("A", 1) → (False, "sender:A")     A 第 3 次：全局先扣到 2，A 自己的桶空 → 被"个人闸"拦下，全局不退还
        acquire("B", 1) → (True,  "")             B 第 1 次：B 的桶是独立的（2→1），不受 A 影响
        （若全局桶先空，则返回 (False, "global")，无论谁都被总闸拦下）
    """

    def __init__(self, capacity: int = 40, refill_per_sec: float = 20.0,
                 per_sender_capacity: int = 10, per_sender_refill: float = 5.0):
        self.global_bucket = TokenBucket(capacity, refill_per_sec)      # 全局桶：所有人共用，管总量
        self._per_capacity = max(1, int(per_sender_capacity))           # 每人桶容量：max(1,...) 保底，传 0 = 永久封死所有人
        self._per_refill = max(0.0, float(per_sender_refill))           # 每人桶回血速率（个/秒）
        self._senders: dict[str, TokenBucket] = {}                      # {发送者名: 它专属的桶}，第一次来才建
        self._lock = threading.Lock()                                   # _senders 是共享 dict，并发建桶要拿锁

    # ------------------------------------------------------------------
    # 【什么】取（或新建）某个发送者的专属桶
    # 【作用】每人桶按需创建：没来过的发送者第一次 acquire 才建桶，
    #        避免为根本不存在的发送者预建一堆空桶占内存
    # 【输入输出对照】_senders={} 时：
    #   _bucket_for("A") → 新建桶(capacity=10, refill=5.0) 并存入，返回它
    #   _bucket_for("A") → 直接命中 dict 里那个，不再新建
    # ------------------------------------------------------------------
    def _bucket_for(self, sender: str) -> TokenBucket:
        with self._lock:                                                # 拿锁：两个线程同时查同一个新名字，只建一个桶
            b = self._senders.get(sender)                               # 查 dict：这个发送者来过吗？来过就有他的桶
            if b is None:                                               # None = 第一次来，dict 里还没有他的桶
                b = TokenBucket(self._per_capacity, self._per_refill)   # 现场造一个桶，规格用构造时定的每人桶参数
                self._senders[sender] = b                               # 存进 dict：下次再来直接命中，同一个桶持续记账
            return b                                                    # 新建的、已有的，都返回这个人的桶

    # ------------------------------------------------------------------
    # 【什么】取令牌：先过全局闸，再过个人闸
    # 【作用】两道闸任一不过就拦下并说明被谁拦（global / sender:名字）；
    #        全过才返回 True。注意顺序：全局先扣、个人后扣
    # 【为什么不个人先扣】个人桶空是常态（超个人的频次很常见），
    #   让"被个人闸拦下的消耗"也计入全局账上，形成自然背压（见下方注释）
    #   先查个人能省掉全局桶的一次扣减；但这里选择全局先扣是为了
    # ------------------------------------------------------------------
    def acquire(self, sender: str = "", n: int = 1) -> tuple[bool, str]:
        """取令牌。返回 (是否放行, 被谁拦下)。"""
        if not self.global_bucket.try_consume(n):       # 全局桶扣不动：总闸已满，无论谁都被拦
            return False, "global"
        if sender:
            if not self._bucket_for(sender).try_consume(n):
                # 【为什么 per-sender 拦下后不把全局令牌还回去】
                #   还回去等于"被限流的发送者可以无限重试直到把全局桶也耗光"；
                #   不还，它这一轮的失败就会计入全局消耗，形成自然背压。
                return False, f"sender:{sender}"        # 全局扣了、个人扣不动：被自己那道闸拦下，全局令牌不退还
        return True, ""                                 # 两道闸都过了：放行（空 sender 只过全局闸）

    # ------------------------------------------------------------------
    # 【什么】当前各桶剩余令牌快照
    # 【作用】给监控/日志看限流水位；available 取值时会先补令牌再报数
    # 【输入输出对照】刚 acquire 掉 2 个、全局容量 40：
    #   stats() → {"global_available": 38.0, "senders": {"A": 8.0}}
    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
    # 【什么】配额记账的构造：四道上限 + 三本账 + 一把锁
    # 【作用】上限在"速率"之外卡"总量"——限流拦猛攻，配额拦慢刷
    # 【输入输出对照】Quota(per_run=100, per_round=30, per_sender_round=3, max_hops=2)：
    #   _total=0（总账）、_round_counts={}（按轮记账）、
    #   _sender_counts={}（按 (轮,发送者) 记账），全部从零开始
    # ------------------------------------------------------------------
    def __init__(self, per_run: int = 100, per_round: int = 30,
                 per_sender_round: int = 3, max_hops: int = 2):
        self.per_run = max(1, int(per_run))                      # 整次运行消息总量上限（防总量失控）
        self.per_round = max(1, int(per_round))                  # 单轮消息总量上限（防单轮爆炸）
        self.per_sender_round = max(1, int(per_sender_round))    # 同一人单轮条数上限（防话痨刷屏）
        self.max_hops = max(0, int(max_hops))                    # 一条对话链最多几跳（防 A→B→A→B 来回打乒乓）
        self._total = 0                                          # 总账：本次运行已发多少条
        self._round_counts: dict[int, int] = {}                  # 按轮记账：{第几轮: 该轮已发几条}
        self._sender_counts: dict[tuple[int, str], int] = {}     # 按(轮,发送者)记账：{("第1轮","A"): A 这轮发了几条}
        self._lock = threading.Lock()                            # 多线程并发 check/consume 要排队，否则两边同时查都看到"没超"

    # ------------------------------------------------------------------
    # 【什么】只查不扣：这条消息能不能发
    # 【作用】四道闸按"从粗到细"过一遍：总量 → 本轮 → 本人本轮 → 跳数，
    #        任一超了就返回 (False, 中文原因)，全过返回 (True, "")
    # 【为什么 check 和 consume 分开】先查后扣是两步：
    #   消息真正发出去才算消费，被拦/发送失败都不该占配额
    # 【输入输出对照】per_sender_round=3，A 在第 1 轮已发 3 条：
    #   check("A", 1, 1) → (False, "A 在第 1 轮已达上限 3")
    #   check("B", 1, 1) → (True, "")          B 只发了 1 条，不受 A 影响
    # ------------------------------------------------------------------
    def check(self, sender: str, round_no: int, hop: int) -> tuple[bool, str]:
        with self._lock:                                         # 拿锁：查的瞬间不能有人正在改账本
            if self._total >= self.per_run:                      # 闸① 总量：整次运行的消息数到顶了
                return False, f"已达运行总量上限 {self.per_run}"
            if self._round_counts.get(round_no, 0) >= self.per_round:  # 闸② 本轮：这一轮大家合计发太多了
                return False, f"第 {round_no} 轮已达上限 {self.per_round}"
            if self._sender_counts.get((round_no, sender), 0) >= self.per_sender_round:  # 闸③ 本人本轮：这个人这轮话太多了
                return False, (f"{sender} 在第 {round_no} 轮已达上限 "
                               f"{self.per_sender_round}")
            if hop > self.max_hops:                              # 闸④ 跳数：一条链转手次数超了（防乒乓对话）
                return False, f"跳数 {hop} 超过上限 {self.max_hops}"
            return True, ""                                      # 四道全过：能发

    # ------------------------------------------------------------------
    # 【什么】记账：消息真发出去了，三本账各 +1
    # 【作用】check 放行、消息落地后才调；不调 = 配额形同虚设（账不上、永不超）
    # 【输入输出对照】consume("A", 1) 后：
    #   _total 0→1、_round_counts {1:1}、_sender_counts {(1,"A"):1}
    # ------------------------------------------------------------------
    def consume(self, sender: str, round_no: int) -> None:
        with self._lock:
            self._total += 1                                     # 总账 +1
            self._round_counts[round_no] = self._round_counts.get(round_no, 0) + 1  # 本轮账 +1（该轮第一条时 get 兜底 0）
            key = (round_no, sender)                             # 复合键：同一个发送者在不同轮是两条账
            self._sender_counts[key] = self._sender_counts.get(key, 0) + 1  # 本人本轮账 +1

    # ------------------------------------------------------------------
    # 【什么】新一轮提问时清零重开（总账和分账全部归零）
    # 【作用】配额是"一轮运行的预算"不是终身预算；不重置 = 第一次跑满后永远被拦
    # ------------------------------------------------------------------
    def reset_run(self) -> None:
        with self._lock:
            self._total = 0
            self._round_counts.clear()
            self._sender_counts.clear()

    # ------------------------------------------------------------------
    # 【什么】当前各账本快照（监控用）
    # 【输入输出对照】A 在第 1 轮发了 2 条后：
    #   stats() → {"total": 2, "by_round": {1: 2}, "limits": {per_run: 100, ...}}
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            return {"total": self._total,
                    "by_round": dict(self._round_counts),
                    "limits": {"per_run": self.per_run, "per_round": self.per_round,
                               "per_sender_round": self.per_sender_round,
                               "max_hops": self.max_hops}}


# ---------------------------------------------------------------------------
# 【什么】指数退避重试：失败别马上重试，越失败等越久，还要加随机抖动
# 【作用】给"抖一下就好"的临时故障（网络闪断/下游瞬时过载）一次自愈机会，
#        同时用递增间隔防止把"抖一下"打成"真过载"
# ---------------------------------------------------------------------------


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
    last: BaseException = RuntimeError("未执行")    # 兜底异常：循环一次都没进就 raise（理论上到不了）
    for i in range(1, max(1, int(attempts)) + 1):   # 第 1 次～第 attempts 次：首跑也算一次 attempt
        try:
            return fn()                             # 成功：立即返回，不再碰重试逻辑
        except retry_on as e:                       # type: ignore[misc]
            last = e                                # 记下最新异常：最终失败时抛的是最后一次的（信息最全）
            if i >= attempts:                       # 次数用尽：不再等待，跳出循环抛异常
                break
            delay = min(max_delay, base_delay * (2 ** (i - 1)))  # 指数退避：0.5 → 1.0 → 2.0 ... 封顶 max_delay
            delay *= random.uniform(0.75, 1.25)     # 抖动：打散并行重试时刻
            _logger.warning(f"重试 {i}/{attempts - 1} 次后仍失败，{delay:.2f}s 后重试: {e}")
            if on_retry:                            # 可选钩子：调用方想在每次重试时记指标/告警就递进来
                try:
                    on_retry(i, e)
                except Exception as hook_err:       # 钩子崩了不能拖垮重试本身
                    _logger.debug(f"重试回调异常（忽略）: {hook_err}")
            time.sleep(delay)                       # 真正的等待：退避时间在这里生效
    raise last                                      # 次数耗尽仍失败：抛最后一次异常，交给上层降级


def remaining_seconds(deadline: float) -> float:
    """距绝对截止时刻还剩几秒。用于把总超时传导到每个下游调用。"""
    return max(0.0, deadline - time.time())         # max(0,·)：已过截止时刻返回 0（=立刻超时），不返回负数让下游误判还有时间
