"""
本地模拟分布式多 Agent（消息队列架构）
========================================
回答"什么时候上 Kafka/RabbitMQ 的分布式多 Agent、没有远程机器怎么验证架构"：
真实分布式 = 各 Agent 节点独立部署在不同机器，靠消息代理（Broker）中转任务与结果。
没有远程时，用"多进程 + 跨进程队列"在单机上完整模拟这套语义：

    主进程                                         每个 Agent 节点 = 一个独立进程
    ┌────────────────────────────┐                ┌──────────────────────────┐
    │ DistributedCoordinator      │                │ _node_main 循环           │
    │  ├─ SimulatedBroker(线程)   │  投递(任务队列) → │  取消息 → 执行 → 发结果    │
    │  │   路由/重投/死信/统计     │  ←── 结果队列 ── │  → ack 回执               │
    │  └─ collect/冲突/降级       │  ←── ack 队列 ── │                          │
    └────────────────────────────┘                └──────────────────────────┘

【模拟了消息代理的哪些核心语义】（换真 Kafka/RabbitMQ 时一一对应）
    topic 路由        publish(topic) → 订阅该 topic 的节点轮询投递（消费组语义）
    ack 回执          节点处理成功才 ack；异常则 nack 立即重投
    可见性超时        投出 visibility_timeout 秒仍无 ack → 视为节点失联，重投
    重试上限 + 死信   同一条消息重投 max_retries 次仍失败 → 进死信队列（DLQ）
    at-least-once    未 ack 的消息一定会再来，节点要能承受重复消费

【生产四件套在分布式形态下的落点】
    ① 委派深度限制  _process_delegations：depth+1 超上限 → 拒绝并记录
    ② 子任务超时    collect 的整体 deadline，到期没结果的子任务进入降级名单
    ③ 失败降级      结果缺失（超时/死信）→ 该子任务标 degraded，其余照常拼答案
    ④ 子结果冲突    冗余派发（redundancy 份同任务）→ 多数决收敛，少数派留痕

【输入输出对照】2 节点 + 2 个普通任务：
    run([{"task_id":"t1","instruction":"查营收"},{"task_id":"t2","instruction":"算环比"}])
    → answers = {"t1": "已执行: 查营收", "t2": "已执行: 算环比"}
    → broker_stats = {"published": 2, "delivered": 2, "acked": 2, "redelivered": 0, "dead_lettered": 0}

【进程模式的约束（Windows spawn）】注入的 executor 必须是模块级函数或可 pickle 对象，
    子进程里 pickle 反解它；线程模式（use_processes=False）没有这个限制，测试用它提速。
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import threading
import time
import uuid
from typing import Any, Callable, Optional

from modules.common.logger import get_logger

_logger = get_logger("multiagent.distributed")


# ==================================================================
# 队列工厂：进程模式用 Manager 队列（跨进程），线程模式用普通队列（同进程）
# ==================================================================
class _QueueFactory:
    """按模式造队列：两种队列接口一致（put/get/get_nowait/Empty）。

    【为什么进程模式必须 Manager】普通 queue.Queue 只在本进程内有意义，
    子进程拿到它只是拿到一份序列化副本，put/get 互不相通；
    Manager().Queue() 是代理对象，背后由独立的管理进程中转，跨进程真互通。
    """

    def __init__(self, use_processes: bool):
        self._manager = mp.Manager() if use_processes else None

    def new_queue(self) -> Any:
        return self._manager.Queue() if self._manager is not None else _queue.Queue()


# ==================================================================
# 模块级默认执行器（必须模块级：spawn 子进程要按引用 pickle 它）
# ==================================================================
def _default_node_executor(payload: dict) -> dict:
    """节点的默认执行逻辑（确定性模拟，生产替换点）。

    按 payload["mode"] 分五种行为，覆盖四件套的验证场景：
        normal   正常完成，content = "已执行: {instruction}"
        fail     抛异常 → 节点 nack → 代理重投 → 超过重试上限进死信
        slow     sleep payload["seconds"] 秒 → 验证子任务超时降级
        biased   content 拼上 payload["seed"] → 验证冗余派发的冲突多数决
        delegate 返回 delegate_request → 验证跨队列的委派链与深度闸
    """
    mode = payload.get("mode", "normal")
    instruction = payload.get("instruction", "")
    if mode == "fail":
        raise RuntimeError(f"模拟执行失败: {instruction}")
    if mode == "slow":
        time.sleep(float(payload.get("seconds", 5)))
    if mode == "biased":
        return {"content": f"已执行: {instruction} | 结论 {payload.get('seed', 'A')}", "success": True}
    if mode == "delegate":
        return {
            "content": f"需要下层数据: {instruction}",
            "success": True,
            "delegate_request": {
                "topic": payload.get("delegate_topic", "research"),
                "instruction": payload.get("delegate_instruction", "承接委派的子任务"),
            },
        }
    return {"content": f"已执行: {instruction}", "success": True}


# ==================================================================
# 节点主循环（模块级：spawn 子进程的 target 必须可 pickle）
# ==================================================================
def _node_main(node_name: str, delivery_q: Any, ack_q: Any, result_q: Any,
               executor: Callable[[dict], dict], poll: float = 0.2) -> None:
    """Agent 节点主循环：取消息 → 执行 → 发结果 → ack/nack。

    【at-least-once 的含义】没 ack 的消息代理一定会重投，
    所以执行体要么幂等（重复执行无害），要么靠 task_id 去重——
    本模拟的默认执行器是纯函数，天然幂等。
    """
    _logger.info(f"节点[{node_name}] 启动 pid={mp.current_process().pid}")
    while True:
        try:
            msg = delivery_q.get(timeout=poll)
        except _queue.Empty:
            continue                          # 没消息继续等
        if msg is None:                       # 毒丸：stop() 时投递，节点优雅退出
            _logger.info(f"节点[{node_name}] 收到退出信号")
            break
        tag = msg.get("tag", "")
        payload = msg.get("payload", {})
        started = time.time()
        try:
            out = executor(payload) or {}
            result_q.put({
                "task_id": payload.get("task_id", ""),
                "node": node_name,
                "pid": mp.current_process().pid,
                "content": str(out.get("content", "")),
                "success": bool(out.get("success", True)),
                "delegate_request": out.get("delegate_request"),
                "depth": int(payload.get("depth", 1)),
                "duration_ms": int((time.time() - started) * 1000),
            })
            ack_q.put(("ack", tag))           # 成功才 ack：结果发出去了，这条消息算消费完
        except Exception as e:
            # 【nack 而不是静默吞】代理收到 nack 立即重投；反复失败由代理的
            #   重试上限送进死信，节点这边不无限重试同一个毒消息
            ack_q.put(("nack", tag))
            _logger.warning(f"节点[{node_name}] 执行异常(已nack): {type(e).__name__}: {e}")


# ==================================================================
# 模拟消息代理：topic 路由 + ack + 可见性超时重投 + 死信
# ==================================================================
class SimulatedBroker:
    """进程内消息代理（后台线程跑路由循环）。

    【一句话看破】三张账本：待投（按 topic 轮询分发给订阅者）、
    在途（投出未 ack，超时要重投）、死信（重投耗尽的消息留档）。
    生产换 Kafka/RabbitMQ 时，这三张账本分别对应
    分区/消费组、未确认投递（unacked）、死信队列（DLQ）。
    """

    def __init__(self, queue_factory: _QueueFactory,
                 visibility_timeout: float = 4.0, max_retries: int = 3):
        self._qf = queue_factory
        self._visibility_timeout = visibility_timeout   # 投出多久没 ack 视为节点失联
        self._max_retries = max_retries                 # 最多重投几次，超了进死信
        self._publish_q = queue_factory.new_queue()     # 发布请求（发布方 → 代理）
        self.ack_q = queue_factory.new_queue()          # ack/nack 回执（节点 → 代理）
        self._subs: dict[str, list[Any]] = {}           # topic → 订阅节点的投递队列列表
        self._rr: dict[str, int] = {}                   # topic → 轮询游标（消费组负载均衡）
        self._inflight: dict[str, dict] = {}            # tag → {msg, ts}
        self._dlq: list[dict] = []                      # 死信留档
        self._stats = {"published": 0, "delivered": 0, "acked": 0,
                       "redelivered": 0, "dead_lettered": 0}
        # 【为什么用 RLock】重投路径会被持锁调用（nack 处理/超时巡检都在锁内
        #   弹出在途条目后直接重投），普通 Lock 不可重入会当场死锁
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # 【什么】注册一个订阅者到 topic，返回它的专属投递队列
    def register(self, topic: str) -> Any:
        dq = self._qf.new_queue()
        with self._lock:
            self._subs.setdefault(topic, []).append(dq)
        return dq

    # 【什么】发布一条消息到 topic（队列自带线程安全，这里不用拿锁）
    def publish(self, topic: str, payload: dict) -> str:
        tag = uuid.uuid4().hex[:8]
        self._publish_q.put(("pub", topic, {"tag": tag, "payload": payload, "retries": 0}))
        return tag

    # 【什么】投毒丸给所有订阅队列（停止节点用）
    def send_poison(self) -> None:
        with self._lock:
            for qs in self._subs.values():
                for dq in qs:
                    try:
                        dq.put(None)
                    except Exception:
                        pass

    # 【什么】代理主循环：发布→路由，回执→销账，在途超时→重投/死信
    def _loop(self) -> None:
        while not self._stop.is_set():
            # ① 取发布请求并路由（同 topic 轮询分发 = 消费组负载均衡）
            try:
                while True:
                    kind, topic, msg = self._publish_q.get_nowait()
                    if kind != "pub":
                        continue
                    self._route(topic, msg, first=True)
            except _queue.Empty:
                pass
            # ② 取回执：ack 销账；nack 立即重投（或死信）
            try:
                while True:
                    action, tag = self.ack_q.get_nowait()
                    with self._lock:
                        entry = self._inflight.pop(tag, None)
                    if entry is None:
                        continue
                    if action == "ack":
                        with self._lock:
                            self._stats["acked"] += 1
                    else:
                        self._route(entry["msg"]["payload"].get("__topic", "tasks"),
                                    entry["msg"], first=False)
            except _queue.Empty:
                pass
            # ③ 在途超时巡检：投出太久没 ack 的消息重投/死信
            now = time.time()
            with self._lock:
                expired = [t for t, e in self._inflight.items()
                           if now - e["ts"] >= self._visibility_timeout]
                expired_msgs = [self._inflight.pop(t)["msg"] for t in expired]
            for msg in expired_msgs:
                self._route(msg["payload"].get("__topic", "tasks"), msg, first=False)
            time.sleep(0.05)

    # 【什么】把消息投给 topic 的下一个订阅者；重投耗尽则进死信
    # 【输入输出对照】max_retries=2：首投失败 → retries=1 重投；
    #   再失败 → retries=2 重投；又失败 → retries(2) >= max_retries(2) → 死信
    def _route(self, topic: str, msg: dict, first: bool) -> None:
        with self._lock:
            if not first:
                if msg["retries"] >= self._max_retries:
                    self._stats["dead_lettered"] += 1
                    self._dlq.append(dict(msg))
                    _logger.warning(f"消息 {msg['tag']} 重投 {msg['retries']} 次仍失败，进死信")
                    return
                msg["retries"] += 1
                self._stats["redelivered"] += 1
            else:
                self._stats["published"] += 1
            subs = self._subs.get(topic) or []
            if not subs:
                return                          # 没订阅者：丢弃（模拟环境，记日志足够）
            idx = self._rr.get(topic, 0) % len(subs)
            self._rr[topic] = idx + 1
            subs[idx].put(msg)
            self._stats["delivered"] += 1
            self._inflight[msg["tag"]] = {"msg": msg, "ts": time.time()}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="simbroker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    @property
    def dlq(self) -> list[dict]:
        with self._lock:
            return list(self._dlq)


# ==================================================================
# 分布式协调器：提交、收集、委派链、冲突收敛、降级
# ==================================================================
class DistributedCoordinator:
    """分布式多 Agent 的客户端侧：发布任务 → 等结果 → 四件套收敛。

    【输入输出对照】冗余 3 份、种子 A/B/A 的同一任务（冲突场景）：
        run([{"task_id":"t1","instruction":"营收","mode":"biased",
              "redundancy":3,"seeds":["A","B","A"]}])
        → answers = {"t1": "已执行: 营收 | 结论 A"}
        → conflicts = [{"task_id":"t1","variants":[A,B,A],"winner":"...A","rule":"多数决"}]
    """

    def __init__(self, max_depth: int = 1, task_timeout: float = 12.0,
                 visibility_timeout: float = 4.0, max_retries: int = 3,
                 use_processes: bool = True):
        self.max_depth = max_depth                 # 四件套①：委派链深度上限
        self.task_timeout = task_timeout           # 四件套②：子任务等结果超时
        self.use_processes = use_processes
        self._qf = _QueueFactory(use_processes)
        self.broker = SimulatedBroker(self._qf, visibility_timeout=visibility_timeout,
                                      max_retries=max_retries)
        self.result_q = self._qf.new_queue()
        self._nodes: list = []                     # 进程/线程句柄
        self._denied: list[dict] = []              # 被深度闸拒绝的委派链记录

    # ------------------------------------------------------------------
    # 【什么】加一个 Agent 节点（进程或线程，订阅某 topic）
    # ------------------------------------------------------------------
    def add_node(self, name: str, topic: str,
                 executor: Callable[[dict], dict] = _default_node_executor) -> None:
        dq = self.broker.register(topic)
        args = (name, dq, self.broker.ack_q, self.result_q, executor)
        if self.use_processes:
            p = mp.Process(target=_node_main, args=args, daemon=True, name=f"node-{name}")
            p.start()
            self._nodes.append(p)
        else:
            t = threading.Thread(target=_node_main, args=args, daemon=True, name=f"node-{name}")
            t.start()
            self._nodes.append(t)
        _logger.info(f"节点已注册: {name} topic={topic} 模式={'进程' if self.use_processes else '线程'}")

    # ------------------------------------------------------------------
    # 【什么】发布任务（支持冗余份数：同任务发 k 份做冲突检测）
    # 【作用】消息里带 __topic 供代理重投时找回路由；seeds 给 biased 模式变体
    # ------------------------------------------------------------------
    def submit(self, task: dict) -> None:
        redundancy = int(task.get("redundancy", 1))
        seeds = task.get("seeds") or [None] * redundancy
        for k in range(redundancy):
            payload = dict(task)
            payload["__topic"] = task.get("topic", "tasks")
            if seeds[k] is not None:
                payload["seed"] = seeds[k]
            self.broker.publish(payload["__topic"], payload)

    # ------------------------------------------------------------------
    # 【四件套②+③】收集结果：deadline 内等齐；同任务多份结果全部收下（冲突要用）
    # 【输入输出对照】task_timeout=1.0、t2 是 slow(3s)：
    #   收到 t1 → pending 剩 {t2}；1 秒到 → got 里没有 t2 → degraded=[{"task_id":"t2"}]
    # 【为什么要宽限期 grace】冗余副本同 task_id：第一份到达就清空 pending，
    #   若立刻收工，第 2/3 份副本可能还在路上——冲突检测就永远只看到 1 份。
    #   收齐后多等 grace 秒（默认 0.3s），没新消息才算真收工。
    # ------------------------------------------------------------------
    def collect(self, expected_ids: list[str], timeout: float, grace: float = 0.3) -> dict[str, list[dict]]:
        deadline = time.time() + timeout
        pending = set(expected_ids)
        got: dict[str, list[dict]] = {}
        idle_since: Optional[float] = None
        while time.time() < deadline:
            try:
                r = self.result_q.get(timeout=0.1)
            except _queue.Empty:
                if not pending:
                    if idle_since is None:
                        idle_since = time.time()
                    elif time.time() - idle_since >= grace:
                        break                      # 收齐且宽限期内无新副本 → 收工
                continue
            idle_since = None
            tid = r.get("task_id", "")
            got.setdefault(tid, []).append(r)      # 副本一律入账，冲突检测靠全量
            pending.discard(tid)
        return got

    # ------------------------------------------------------------------
    # 【四件套④】冲突收敛：同任务多份结果内容不一致 → 多数决，平票取先到
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_conflicts(results_by_task: dict[str, list[dict]]) -> tuple[dict, list[dict]]:
        answers: dict[str, str] = {}
        conflicts: list[dict] = []
        for tid, rs in results_by_task.items():
            if not rs:
                continue
            contents = [r.get("content", "") for r in rs]
            if len(set(contents)) <= 1:
                answers[tid] = contents[0]
                continue
            counts: dict[str, int] = {}
            for c in contents:
                counts[c] = counts.get(c, 0) + 1
            best = max(counts.values())
            winners = [c for c, n in counts.items() if n == best]
            if len(winners) == 1:
                winner, rule = winners[0], "多数决"
            else:
                winner, rule = contents[0], "平票取先到"
            answers[tid] = winner
            conflicts.append({"task_id": tid, "variants": contents, "winner": winner, "rule": rule})
        return answers, conflicts

    # ------------------------------------------------------------------
    # 【四件套①】委派链深度闸：节点结果里带 delegate_request 时逐层放行/拒绝
    # 【输入输出对照】max_depth=1：第 1 层任务请求委派 → 子任务 depth=2 > 1 →
    #   denied 记录 {"task_id":"t1","to_depth":2,"reason":"..."}，链路到此为止
    # ------------------------------------------------------------------
    def _process_delegations(self, results: list[dict], deadline: float) -> list[dict]:
        chained: list[dict] = []
        frontier = [r for r in results if r.get("delegate_request")]
        rounds = 0
        while frontier and rounds < 5:            # rounds 兜底：深度闸正常时不会走到这
            rounds += 1
            children: list[dict] = []
            for r in frontier:
                req = r["delegate_request"]
                r["delegate_request"] = None      # 消费掉，防二次处理
                child_depth = int(r.get("depth", 1)) + 1
                if child_depth > self.max_depth:
                    self._denied.append({"task_id": r.get("task_id", ""),
                                         "to_depth": child_depth,
                                         "reason": f"超过委派深度上限 {self.max_depth}"})
                    _logger.info(f"分布式委派被拒: {r.get('task_id')} 深度 {child_depth}")
                    continue
                children.append({
                    "task_id": f"{r.get('task_id', 'task')}>d{child_depth}",
                    "topic": req.get("topic", "tasks"),
                    "instruction": req.get("instruction", ""),
                    "depth": child_depth,
                })
            if not children:
                break
            for c in children:
                self.submit(c)
            remaining = max(0.5, deadline - time.time())
            got = self.collect([c["task_id"] for c in children], remaining)
            new_results = [r for rs in got.values() for r in rs]
            chained.extend(new_results)
            frontier = [r for r in new_results if r.get("delegate_request")]
        return chained

    # ------------------------------------------------------------------
    # 【什么】一次完整分布式运行：提交 → 收集 → 委派链 → 冲突 → 降级 → 汇总
    # ------------------------------------------------------------------
    def run(self, tasks: list[dict]) -> dict:
        self.broker.start()
        started = time.time()
        deadline = started + self.task_timeout
        ids = [t["task_id"] for t in tasks]
        for t in tasks:
            t.setdefault("topic", "tasks")
            t.setdefault("depth", 1)
            self.submit(t)
        got = self.collect(ids, self.task_timeout)
        results = [r for rs in got.values() for r in rs]
        # 委派链：处理节点冒出的 delegate_request（受深度闸约束）
        chained = self._process_delegations(results, deadline)
        for r in chained:
            got.setdefault(r["task_id"], []).append(r)
        # 冲突收敛 + 降级名单
        answers, conflicts = self._resolve_conflicts(got)
        degraded = [{"task_id": tid, "reason": f"等待结果超时({self.task_timeout}s) 或已进死信"}
                    for tid in ids if not got.get(tid)]
        return {
            "answers": answers,
            "results": [r for rs in got.values() for r in rs],
            "conflicts": conflicts,
            "degraded": degraded,
            "denied": list(self._denied),
            "broker_stats": self.broker.stats(),
            "duration_ms": int((time.time() - started) * 1000),
        }

    # ------------------------------------------------------------------
    # 【什么】停止：投毒丸收节点 → 停代理线程 → 回收子进程
    # ------------------------------------------------------------------
    def stop(self) -> None:
        try:
            self.broker.send_poison()
        except Exception:
            pass
        time.sleep(0.3)                            # 给节点一点时间消化毒丸
        self.broker.stop()
        for n in self._nodes:
            if isinstance(n, mp.Process) and n.is_alive():
                n.terminate()
        self._nodes = []
