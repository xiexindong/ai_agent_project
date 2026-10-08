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
from __future__ import annotations  # 启用延迟注解求值，允许类型提示前向引用

import threading  # Event 等待唤醒与锁
from typing import Optional, Protocol  # 类型标注与协议类

from modules.common.logger import get_logger  # 项目统一日志器工厂
from modules.hitl.models import (  # 导入 HITL 数据契约
    HumanReply,  # 人工回复模型
    InterruptRecord,  # 请求 + 回复归档模型
    InterruptRequest,  # 中断请求模型
    InterruptStatus,  # 生命周期状态枚举
    now_ms,  # 毫秒时间戳工具
)

_logger = get_logger("hitl_store")  # 本模块日志器


# 【什么】存储能力协议
# 【作用】定义存储实现必须满足的五个方法
# 【为什么】存储与判定分开后可换实现（内存 / PG），Agent 侧一行不改
class InterruptStore(Protocol):
    """存储能力契约。换实现只需满足这五个方法。"""

    def save(self, req: InterruptRequest) -> InterruptRequest: ...  # 登记请求
    def get(self, request_id: str) -> Optional[InterruptRequest]: ...  # 按 ID 取请求
    def pending(self, limit: int = 50) -> list[InterruptRequest]: ...  # 取待处理清单
    def submit(self, reply: HumanReply) -> bool: ...  # 提交人工回复
    def wait(self, request_id: str, timeout_seconds: float) -> Optional[HumanReply]: ...  # 阻塞等待回复


# 【什么】进程内存储实现
# 【作用】字典存请求、Event 做等待唤醒、锁保线程安全
# 【为什么】demo / 单测 / 单实例部署够用；零外部依赖
class InMemoryInterruptStore:
    """进程内存储：字典存请求、Event 做等待唤醒、锁保线程安全。"""

    def __init__(self, max_records: int = 500):  # max_records: 归档记录上限
        self._reqs: dict[str, InterruptRequest] = {}  # 请求表：id → 请求
        self._replies: dict[str, HumanReply] = {}  # 回复表：请求 id → 回复
        self._events: dict[str, threading.Event] = {}  # 等待事件表：请求 id → Event
        self._records: list[InterruptRecord] = []  # 归档列表：请求 + 回复成对
        # 【为什么要有上限】审批记录只进不出会撑爆内存。
        #   超上限丢最旧的：最近的请求才是排查时关心的。
        self._max_records = max(10, int(max_records))  # 上限最小 10，防误配成 0
        self._lock = threading.RLock()  # 保护全部共享状态的可重入锁

    # ------------------------------------------------------------------
    # 【写】登记一次中断请求，同时给它配一个 Event
    #   预热 Event 而不是等到 wait 时才建，避免"人先回复、Agent 后等待"的
    #   竞态：那样 wait 会建一个新 Event，永远等不到已提交的回复。
    # ------------------------------------------------------------------
    # 【什么】登记一次中断请求
    # 【作用】存请求、预热 Event、追加归档
    # 【为什么】预热 Event 避免"人先回复、Agent 后等待"的竞态
    def save(self, req: InterruptRequest) -> InterruptRequest:  # 返回存入的请求
        with self._lock:  # 全程加锁
            self._reqs[req.id] = req  # 请求入表
            self._events.setdefault(req.id, threading.Event())  # 预热等待 Event（已存在则不覆盖）
            self._records.append(InterruptRecord(request=req))  # 追加归档记录
            if len(self._records) > self._max_records:  # 超过归档上限
                self._records = self._records[-self._max_records:]  # 丢最旧的，只留最近的
            _logger.info(f"中断请求已登记: {req.id} kind={req.kind.value} title={req.title[:40]}")  # 记登记日志
            return req  # 返回请求

    # 【什么】按 ID 取请求
    # 【作用】给外部查询单个请求用
    # 【为什么】读也要加锁，避免读到写一半的中间态
    def get(self, request_id: str) -> Optional[InterruptRequest]:  # 找不到返回 None
        with self._lock:  # 加锁读
            return self._reqs.get(request_id)  # 字典查找

    # ------------------------------------------------------------------
    # 【读】待处理清单：审批页轮询它
    #   只返回 WAITING 的；已回复/已超时的不再出现，
    #   否则审批页会一直显示早就处理过的卡片。
    # ------------------------------------------------------------------
    # 【什么】取待处理清单
    # 【作用】审批页轮询它拿"还等着人拍板"的请求
    # 【为什么】只返回 WAITING；已回复 / 已超时的不应再出现在审批页
    def pending(self, limit: int = 50) -> list[InterruptRequest]:  # limit: 最多返回条数
        with self._lock:  # 加锁读
            items = [r for r in self._reqs.values() if r.status == InterruptStatus.WAITING]  # 过滤出等待中的请求
            items.sort(key=lambda r: r.created_at)  # 按创建时间升序，先来的先处理
            return items[: max(1, int(limit))]  # 截断到上限（至少 1 条）

    # ------------------------------------------------------------------
    # 【人这一侧】提交回复：存下来 → 置状态 → 唤醒等待中的 Agent
    #   返回 False 的三种情况：请求不存在 / 已不是 WAITING（重复提交或已超时）
    #   重复提交要挡住：网络重试可能把同一次点击发两遍，
    #   第二遍若覆盖第一遍，审计上就查不出真正生效的是哪次决策。
    # ------------------------------------------------------------------
    # 【什么】提交人工回复
    # 【作用】存回复 → 置状态 → 唤醒等待中的 Agent
    # 【为什么】重复提交必须挡住，否则审计查不出真正生效的是哪次决策
    def submit(self, reply: HumanReply) -> bool:  # 返回是否受理成功
        with self._lock:  # 全程加锁
            req = self._reqs.get(reply.request_id)  # 找对应请求
            if req is None:  # 请求不存在
                _logger.warning(f"回复对应的请求不存在: {reply.request_id}")  # 记警告
                return False  # 拒收
            if req.status != InterruptStatus.WAITING:  # 请求已结束（重复提交或已超时）
                _logger.warning(f"请求已结束，忽略重复回复: {reply.request_id} status={req.status.value}")  # 记警告
                return False  # 拒收，保住第一次决策
            req.status = InterruptStatus.RESOLVED  # 置为已回复
            self._replies[reply.request_id] = reply  # 回复入表
            for rec in self._records:  # 回填归档里的回复
                if rec.request.id == reply.request_id:  # 找到对应归档
                    rec.reply = reply  # 写入回复
                    rec.resolved_at = now_ms()  # 记录解决时间
                    break  # 找到即停
            self._events.setdefault(reply.request_id, threading.Event()).set()  # 唤醒等待中的 Agent 线程
            _logger.info(  # 记生效日志
                f"人工回复已生效: {reply.request_id} action={reply.action.value} "  # 请求与动作
                f"operator={reply.operator or '-'}"  # 操作人（空显示 -）
            )
            return True  # 受理成功

    # ------------------------------------------------------------------
    # 【Agent 这一侧】阻塞等待：人来就返回回复，超时返回 None
    #   返回 None 表示"没人来"，由调用方（HumanGate）按 default_action 兜底，
    #   并把状态置为 EXPIRED —— 这里不替调用方决定兜底动作，
    #   因为不同中断点的兜底策略不同（工具调用默认拒绝，出答案默认放行）。
    # ------------------------------------------------------------------
    # 【什么】阻塞等待回复
    # 【作用】人来就返回回复，超时返回 None
    # 【为什么】用 Event 内核级等待：人一提交立刻唤醒，超时准时返回
    def wait(self, request_id: str, timeout_seconds: float) -> Optional[HumanReply]:  # None = 没人来
        event = self._events.get(request_id)  # 取该请求的等待 Event
        if event is None:  # 请求未登记过
            return None  # 直接返回"没人来"
        arrived = event.wait(timeout=max(0.0, float(timeout_seconds)))  # 内核级等待；True=被唤醒，False=超时
        with self._lock:  # 加锁读回复
            reply = self._replies.get(request_id)  # 取回复
        if reply is not None:  # 有回复
            return reply  # 返回真人 / 自动答复
        # 【为什么这里不改状态为 EXPIRED】
        #   兜底动作由 HumanGate 决定（不同中断点默认动作不同），
        #   store 只报告"没人来"。若在这里先置 EXPIRED，
        #   gate 随后提交的那条超时回复就会被 submit 以"已结束"为由拒收，
        #   审计记录里就只剩"超时"而看不到"最终按什么动作执行了"。
        if not arrived:  # 是超时返回的（而非被唤醒但没拿到回复的边界情况）
            _logger.warning(f"中断请求超时: {request_id}（{timeout_seconds}s 无人处理）")  # 记超时警告
        return None  # 报告"没人来"，兜底动作交给 HumanGate

    # ------------------------------------------------------------------
    # 【归档】完整留痕：谁在什么时候、为什么被打断、人怎么回的
    # ------------------------------------------------------------------
    # 【什么】取归档记录
    # 【作用】返回请求 + 回复成对的完整留痕
    # 【为什么】审计与排查依赖这份记录；返回拷贝防外部改动
    def records(self) -> list[InterruptRecord]:  # 返回归档列表拷贝
        with self._lock:  # 加锁读
            return list(self._records)  # 浅拷贝后返回

    # 【什么】作废一个等待中的请求
    # 【作用】流程提前结束时置 CANCELLED 并唤醒等待方
    # 【为什么】不唤醒的话等待线程要挂到超时才释放
    def cancel(self, request_id: str) -> bool:  # 返回是否作废成功
        """流程提前结束时作废一个还在等的请求（并唤醒等待方，避免线程挂住）。"""
        with self._lock:  # 全程加锁
            req = self._reqs.get(request_id)  # 找请求
            if req is None or req.status != InterruptStatus.WAITING:  # 不存在或已结束
                return False  # 无法作废
            req.status = InterruptStatus.CANCELLED  # 置为已作废
            self._events.setdefault(request_id, threading.Event()).set()  # 唤醒等待线程
            return True  # 作废成功
