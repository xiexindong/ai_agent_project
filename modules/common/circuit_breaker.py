"""
熔断器模块（Circuit Breaker）
============================
下游（LLM 主模型、外部工具、第三方接口）持续故障时「快速失败」，
避免每个请求都先打一次注定失败的调用、白等到超时。

三态状态机：
    CLOSED（正常）──── 连续失败达到阈值 ────► OPEN（打开）
    OPEN          ──── 冷却期已过 ──────────► HALF_OPEN（半开）
    HALF_OPEN 探针成功 ────────────────────► CLOSED（失败计数清零）
    HALF_OPEN 探针失败 ────────────────────► OPEN（重新开始计时）

【和「重试」「降级」的分工】三者解决的不是同一个问题：
    重试：一次请求内，对同一个下游再试几次      —— 治「偶发抖动」
    熔断：跨请求，判定下游「已经不可用」，直接拦掉 —— 治「持续故障」，省的是超时等待
    降级：被拦或调用失败后，换一个下游顶上      —— 治「可用性」，用户仍拿到答案
    执行顺序：先问熔断 allow() → 放行才调用 → 失败走降级 → 失败累计到阈值就熔断。

【为什么要按 key 隔离】主模型、备用模型、每个外部工具各自独立计数。
    一个下游挂了不该连累另一个 —— 熔断器最怕的就是「一处故障放大成全局不可用」。

【为什么 HALF_OPEN 只放行一个请求】冷却是定时器，不代表下游真的好了。
    一次性放开全部流量去试探，会把刚恢复的下游再打挂（惊群）。
    放一个探针：成功才恢复，失败立刻重新打开。
"""
import threading
import time
from typing import Dict

from modules.common.logger import get_logger

_logger = get_logger("circuit_breaker")

CLOSED = "closed"        # 闭合：正常放行，失败计数累加中
OPEN = "open"            # 打开：冷却期内直接拦掉，根本不调用下游
HALF_OPEN = "half_open"  # 半开：冷却结束，只放行一个探针请求


# 【什么】按 key 隔离的三态熔断器
# 【作用】判定某个下游"已经不可用"后跨请求拦截，省掉每次注定失败的超时等待
class CircuitBreaker:
    """按 key 隔离的三态熔断器，线程安全（内部一把锁，可多线程共享）。

    典型用法：
        if not cb.allow("deepseek-chat"):
            return 走备用通道()                 # 熔断中：快速失败，不做无谓等待
        try:
            r = 调用下游()
        except 瞬时故障:
            cb.record_failure("deepseek-chat")  # 只记「瞬时故障」，业务错误不记
            raise
        cb.record_success("deepseek-chat")
    """

    # 【什么】构造熔断器（阈值与冷却时间在此固化）
    # 【作用】默认连续失败 5 次打开、冷却 30 秒；运行期这两个值不再变
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = max(1, int(failure_threshold))     # 连续失败达 5 次就打开；下限 1
        self.recovery_timeout = max(0.0, float(recovery_timeout))   # 打开后冷却 30 秒才转半开；下限 0
        self._lock = threading.Lock()
        self._failures: Dict[str, int] = {}      # key -> 连续失败次数
        self._opened_at: Dict[str, float] = {}   # key -> 打开时刻（monotonic，不受系统改时间影响）
        self._probing: Dict[str, bool] = {}      # key -> 半开探针是否已放行

    # ------------------------------------------------------------------
    # 对外三个动作：问、报成功、报失败 
    # ------------------------------------------------------------------
    # 【什么】问一句：本次请求能不能放行
    # 【作用】调用下游前先过这一关；返回 False 就直接走降级，不做无谓等待
    def allow(self, key: str) -> bool:
        """本次请求是否放行到下游。返回 False = 正在熔断，调用方应直接走降级。

        三个分支分别对应三种状态：
            CLOSED    → True（放行）
            OPEN      → False（拦掉；此时是「不调用」而不是「调用后失败」）
            HALF_OPEN → 只放行一个探针，其余仍拦
        """
        with self._lock:
            opened_at = self._opened_at.get(key)
            if opened_at is None:
                return True                       # CLOSED
            if time.monotonic() - opened_at < self.recovery_timeout:
                return False                      # OPEN：冷却未到，快速失败
            if self._probing.get(key):
                return False                      # HALF_OPEN：探针已在途
            self._probing[key] = True             # HALF_OPEN：放行唯一探针
            _logger.warning(f"熔断器半开，放行探测请求: {key}")
            return True

    # 【什么】上报一次成功
    # 【作用】清空该 key 的失败计数与打开标记，让下游回到完全放行状态
    def record_success(self, key: str) -> None:
        """下游调用成功：失败计数清零；若此前处于打开/半开则标记为已恢复。"""
        with self._lock:
            was_broken = key in self._opened_at
            self._failures.pop(key, None)
            self._opened_at.pop(key, None)
            self._probing.pop(key, None)
        if was_broken:
            _logger.info(f"熔断器闭合，下游已恢复: {key}")

    # 【什么】上报一次失败
    # 【作用】累加连续失败次数，达到阈值就打开熔断；已打开时重置冷却计时
    def record_failure(self, key: str) -> None:
        """下游调用失败。

        注意：只应记录「瞬时故障」（超时、连接失败、5xx、429）。
        把 400 参数错误之类的业务错误记进来，一次代码 bug 就能误熔断整条链路。
        """
        with self._lock:
            if key in self._opened_at:
                # 半开探针失败 → 重新打开，冷却计时归零
                self._opened_at[key] = time.monotonic()
                self._probing.pop(key, None)
                count, just_opened = self._failures.get(key, 0), False
            else:
                count = self._failures.get(key, 0) + 1
                self._failures[key] = count
                just_opened = count >= self.failure_threshold
                if just_opened:
                    self._opened_at[key] = time.monotonic()
                    self._probing.pop(key, None)
        if just_opened:
            _logger.error(
                f"熔断器打开：{key} 连续失败 {count} 次，"
                f"{self.recovery_timeout}s 内跳过该下游直接走降级"
            )

    # ------------------------------------------------------------------
    # 观测接口：给健康检查 / 监控用，不参与判断
    # ------------------------------------------------------------------
    # 【什么】查单个 key 当前处于哪一态
    # 【作用】给健康检查与监控看；不参与放行判断，判断只认 allow()
    def state(self, key: str) -> str:
        """当前状态：closed / open / half_open。"""
        with self._lock:
            opened_at = self._opened_at.get(key)
            if opened_at is None:
                return CLOSED
            if time.monotonic() - opened_at < self.recovery_timeout:
                return OPEN
            return HALF_OPEN

    # 【什么】全部 key 的状态快照
    # 【作用】运维/健康检查一次性看到所有下游的状态与失败次数；键名排序后输出
    def snapshot(self) -> dict:
        """全部下游的状态快照，形如 {"deepseek-chat": {"state": "open", "failures": 5}}。"""
        with self._lock:
            failures = dict(self._failures)   # 先拷一份再出锁，避免遍历时别的线程在改
            keys = set(self._opened_at) | set(failures)
        return {
            k: {"state": self.state(k), "failures": failures.get(k, 0)}
            for k in sorted(keys)
        }

    # 【什么】手动清空熔断状态
    # 【作用】运维确认下游已恢复、或单元测试要干净起点时调用
    def reset(self, key: str = "") -> None:
        """清空状态。传 key 只清一个，不传清全部（运维手动恢复 / 测试用）。"""
        with self._lock:
            if key:
                for d in (self._failures, self._opened_at, self._probing):
                    d.pop(key, None)
                return
            self._failures.clear()
            self._opened_at.clear()
            self._probing.clear()
