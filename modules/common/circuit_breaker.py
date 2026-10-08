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
import threading                               # 标准库：一把锁保证多线程下状态读写原子性
import time                                    # 标准库：time.monotonic 计时，不受系统改时间影响
from typing import Dict                        # 类型标注：三个状态字典的键值类型

from modules.common.logger import get_logger   # 取带模块标识的 logger，本模块统一用 "circuit_breaker" 标识

_logger = get_logger("circuit_breaker")        # 模块级 logger：打开/半开/恢复事件都打这里

CLOSED = "closed"        # 闭合：正常放行，失败计数累加中
OPEN = "open"            # 打开：冷却期内直接拦掉，根本不调用下游
HALF_OPEN = "half_open"  # 半开：冷却结束，只放行一个探针请求


# 【什么】按 key 隔离的三态熔断器
# 【作用】判定某个下游"已经不可用"后跨请求拦截，省掉每次注定失败的超时等待
# 【为什么】线程池并发场景下多个请求共享同一份状态，内部必须持锁，且状态按下游 key 隔离互不连累
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
    # 【为什么】阈值与冷却是稳定性策略，运行期可变会让行为不可预测，故构造时一次定死
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = max(1, int(failure_threshold))     # 连续失败达 5 次就打开；下限 1
        self.recovery_timeout = max(0.0, float(recovery_timeout))   # 打开后冷却 30 秒才转半开；下限 0
        self._lock = threading.Lock()          # 唯一一把锁：三个状态字典的所有读写都在锁内完成
        self._failures: Dict[str, int] = {}      # key -> 连续失败次数
        self._opened_at: Dict[str, float] = {}   # key -> 打开时刻（monotonic，不受系统改时间影响）
        self._probing: Dict[str, bool] = {}      # key -> 半开探针是否已放行

    # ------------------------------------------------------------------
    # 对外三个动作：问、报成功、报失败 
    # ------------------------------------------------------------------
    # 【什么】问一句：本次请求能不能放行
    # 【作用】调用下游前先过这一关；返回 False 就直接走降级，不做无谓等待
    # 【为什么】熔断省的就是"注定失败的超时等待"，所以判断必须发生在调用之前而不是之后
    def allow(self, key: str) -> bool:
        """本次请求是否放行到下游。返回 False = 正在熔断，调用方应直接走降级。

        三个分支分别对应三种状态：
            CLOSED    → True（放行）
            OPEN      → False（拦掉；此时是「不调用」而不是「调用后失败」）
            HALF_OPEN → 只放行一个探针，其余仍拦
        """
        with self._lock:                       # 状态判断必须原子，全程持锁
            opened_at = self._opened_at.get(key)                   # 取打开时刻；取不到说明从未打开（CLOSED）
            if opened_at is None:              # 无打开记录
                return True                       # CLOSED
            if time.monotonic() - opened_at < self.recovery_timeout:   # 冷却期未满
                return False                      # OPEN：冷却未到，快速失败
            if self._probing.get(key):         # 已有探针在途
                return False                      # HALF_OPEN：探针已在途
            self._probing[key] = True             # HALF_OPEN：放行唯一探针
            _logger.warning(f"熔断器半开，放行探测请求: {key}")       # 记录探针放行事件，便于观测恢复过程
            return True                          # 探针放行

    # 【什么】上报一次成功
    # 【作用】清空该 key 的失败计数与打开标记，让下游回到完全放行状态
    # 【为什么】成功是下游已恢复的唯一可信信号，必须立即全量复位，否则已恢复的下游仍被拦
    def record_success(self, key: str) -> None:
        """下游调用成功：失败计数清零；若此前处于打开/半开则标记为已恢复。"""
        with self._lock:                       # 状态清理必须原子
            was_broken = key in self._opened_at                    # 先记下此前是否在熔断，供锁外决定要不要打日志
            self._failures.pop(key, None)      # 清失败计数；不存在也不报错
            self._opened_at.pop(key, None)     # 清打开标记，回到 CLOSED
            self._probing.pop(key, None)       # 清探针标记
        if was_broken:                         # 此前确实在熔断，这次成功意味着恢复
            _logger.info(f"熔断器闭合，下游已恢复: {key}")           # 记录恢复事件

    # 【什么】上报一次失败
    # 【作用】累加连续失败次数，达到阈值就打开熔断；已打开时重置冷却计时
    # 【为什么】只记瞬时故障不记业务错误——把 400 记进来，一次代码 bug 就能误熔断整条链路
    def record_failure(self, key: str) -> None:
        """下游调用失败。

        注意：只应记录「瞬时故障」（超时、连接失败、5xx、429）。
        把 400 参数错误之类的业务错误记进来，一次代码 bug 就能误熔断整条链路。
        """
        with self._lock:                       # 计数与状态翻转必须原子
            if key in self._opened_at:         # 已在打开/半开状态：这次失败的是探针
                # 半开探针失败 → 重新打开，冷却计时归零
                self._opened_at[key] = time.monotonic()            # 冷却计时从这一刻重算
                self._probing.pop(key, None)                       # 清探针标记，等下一轮冷却
                count, just_opened = self._failures.get(key, 0), False   # 计数不变（只是重开），本次不算"刚打开"
            else:                              # 正常闭合状态：累加失败计数
                count = self._failures.get(key, 0) + 1             # 连续失败次数 +1
                self._failures[key] = count                        # 写回计数
                just_opened = count >= self.failure_threshold      # 达到阈值即触发打开
                if just_opened:                # 本次失败触发了熔断
                    self._opened_at[key] = time.monotonic()        # 记录打开时刻，冷却开始
                    self._probing.pop(key, None)                   # 确保无残留探针标记
        if just_opened:                        # 只在"刚打开"这一次打错误日志，避免冷却期内刷屏
            _logger.error(
                f"熔断器打开：{key} 连续失败 {count} 次，"
                f"{self.recovery_timeout}s 内跳过该下游直接走降级"
            )

    # ------------------------------------------------------------------
    # 观测接口：给健康检查 / 监控用，不参与判断
    # ------------------------------------------------------------------
    # 【什么】查单个 key 当前处于哪一态
    # 【作用】给健康检查与监控看；不参与放行判断，判断只认 allow()
    # 【为什么】观测与决策分离：allow() 是热路径要最小开销，state() 可以慢但不能反过来影响放行
    def state(self, key: str) -> str:
        """当前状态：closed / open / half_open。"""
        with self._lock:                       # 读状态同样持锁，保证读到的是一致快照
            opened_at = self._opened_at.get(key)                   # 取打开时刻
            if opened_at is None:              # 从未打开
                return CLOSED                    # 正常态
            if time.monotonic() - opened_at < self.recovery_timeout:   # 冷却未满
                return OPEN                      # 打开态
            return HALF_OPEN                     # 冷却已过：半开态

    # 【什么】全部 key 的状态快照
    # 【作用】运维/健康检查一次性看到所有下游的状态与失败次数；键名排序后输出
    # 【为什么】先拷贝再出锁：遍历字典时别的线程可能在改，直接在锁里遍历会拉长临界区
    def snapshot(self) -> dict:
        """全部下游的状态快照，形如 {"deepseek-chat": {"state": "open", "failures": 5}}。"""
        with self._lock:                       # 持锁拷贝，缩短临界区
            failures = dict(self._failures)   # 先拷一份再出锁，避免遍历时别的线程在改
            keys = set(self._opened_at) | set(failures)            # 打开过的与有失败记录的 key 取并集
        return {
            k: {"state": self.state(k), "failures": failures.get(k, 0)}   # 每个 key 输出当前状态与失败次数
            for k in sorted(keys)              # 键名排序，输出稳定便于比对
        }

    # 【什么】手动清空熔断状态
    # 【作用】运维确认下游已恢复、或单元测试要干净起点时调用
    # 【为什么】自动恢复依赖探针，运维确认已恢复时手动清状态可立即结束熔断，不用等冷却
    def reset(self, key: str = "") -> None:
        """清空状态。传 key 只清一个，不传清全部（运维手动恢复 / 测试用）。"""
        with self._lock:                       # 清状态也要持锁
            if key:                            # 传了 key：只清这一个
                for d in (self._failures, self._opened_at, self._probing):   # 遍历三个状态字典
                    d.pop(key, None)           # 有就删，没有就跳过
                return                         # 清完单个即返回
            self._failures.clear()             # 清空全部失败计数
            self._opened_at.clear()            # 清空全部打开标记
            self._probing.clear()              # 清空全部探针标记
