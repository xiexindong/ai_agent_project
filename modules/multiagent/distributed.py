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
from __future__ import annotations  # 允许注解前向引用（如 str | None 写法在旧版本解释器也不炸）

import multiprocessing as mp  # 多进程：进程模式下每个 Agent 节点跑在独立进程里
import queue as _queue  # 队列原语：线程模式用它的 Queue，两种模式共用它的 Empty 异常
import threading  # 线程：代理路由循环与线程模式节点都用它
import time  # 计时：deadline、可见性超时、duration_ms 统计
import uuid  # 生成消息 tag（在途消息的唯一身份证）
from typing import Any, Callable, Optional  # 类型注解工具

from modules.common.logger import get_logger  # 项目统一日志器工厂

_logger = get_logger("multiagent.distributed")  # 本模块日志器，按模块名分级过滤


# ==================================================================
# 队列工厂：进程模式用 Manager 队列（跨进程），线程模式用普通队列（同进程）
# ==================================================================
class _QueueFactory:
    """按模式造队列：两种队列接口一致（put/get/get_nowait/Empty）。

    【为什么进程模式必须 Manager】普通 queue.Queue 只在本进程内有意义，
    子进程拿到它只是拿到一份序列化副本，put/get 互不相通；
    Manager().Queue() 是代理对象，背后由独立的管理进程中转，跨进程真互通。
    """

    def __init__(self, use_processes: bool):  # use_processes=True 走跨进程队列，False 走同进程队列
        self._manager = mp.Manager() if use_processes else None  # 进程模式才起管理进程；线程模式置 None 省开销

    def new_queue(self) -> Any:
        return self._manager.Queue() if self._manager is not None else _queue.Queue()  # 有管理进程返回代理队列，否则返回普通线程队列


# ==================================================================
# 模块级默认执行器（必须模块级：spawn 子进程要按引用 pickle 它）
# ==================================================================
def _default_node_executor(payload: dict) -> dict:  # payload：节点收到的任务消息体（含 mode/instruction/seed 等字段）
    """节点的默认执行逻辑（确定性模拟，生产替换点）。

    按 payload["mode"] 分五种行为，覆盖四件套的验证场景：
        normal   正常完成，content = "已执行: {instruction}"
        fail     抛异常 → 节点 nack → 代理重投 → 超过重试上限进死信
        slow     sleep payload["seconds"] 秒 → 验证子任务超时降级
        biased   content 拼上 payload["seed"] → 验证冗余派发的冲突多数决
        delegate 返回 delegate_request → 验证跨队列的委派链与深度闸
    """
    mode = payload.get("mode", "normal")  # 行为开关：缺省按正常任务处理
    instruction = payload.get("instruction", "")  # 任务指令文本，回声式拼进结果里
    if mode == "fail":
        raise RuntimeError(f"模拟执行失败: {instruction}")  # 故意抛错：触发节点 nack → 代理重投 → 死信全链路
    if mode == "slow":
        time.sleep(float(payload.get("seconds", 5)))  # 故意拖时间：让协调器的 collect 超时，验证降级名单
    if mode == "biased":
        return {"content": f"已执行: {instruction} | 结论 {payload.get('seed', 'A')}", "success": True}  # 按 seed 出不同结论，制造冲突供多数决收敛
    if mode == "delegate":
        return {
            "content": f"需要下层数据: {instruction}",  # 本层结论：还差下层数据
            "success": True,  # 本节点自己是成功的，只是要再派一层
            "delegate_request": {  # 委派请求：协调器看到它会再发一个子任务
                "topic": payload.get("delegate_topic", "research"),  # 子任务投给哪个 topic（缺省 research）
                "instruction": payload.get("delegate_instruction", "承接委派的子任务"),  # 子任务指令
            },
        }
    return {"content": f"已执行: {instruction}", "success": True}  # normal：回声式完成


# ==================================================================
# 节点主循环（模块级：spawn 子进程的 target 必须可 pickle）
# ==================================================================
def _node_main(node_name: str, delivery_q: Any, ack_q: Any, result_q: Any,  # node_name：节点名；delivery_q：收任务的投递队列；ack_q：回执队列；result_q：结果队列
               executor: Callable[[dict], dict], poll: float = 0.2) -> None:  # executor：任务执行体（须可 pickle）；poll：取消息的轮询超时秒数
    """Agent 节点主循环：取消息 → 执行 → 发结果 → ack/nack。

    【at-least-once 的含义】没 ack 的消息代理一定会重投，
    所以执行体要么幂等（重复执行无害），要么靠 task_id 去重——
    本模拟的默认执行器是纯函数，天然幂等。
    """
    _logger.info(f"节点[{node_name}] 启动 pid={mp.current_process().pid}")  # 启动日志带上 pid，验证"真的是独立进程"
    while True:  # 节点常驻：直到收到毒丸才退出
        try:
            msg = delivery_q.get(timeout=poll)  # 带超时取消息：既能等任务，又能定期醒来看看有没有退出信号之外的动静
        except _queue.Empty:
            continue                          # 没消息继续等
        if msg is None:                       # 毒丸：stop() 时投递，节点优雅退出
            _logger.info(f"节点[{node_name}] 收到退出信号")  # 记录退出日志，便于排查节点是否正常收尾
            break  # 跳出主循环，进程/线程随之结束
        tag = msg.get("tag", "")  # 消息身份证：ack/nack 时凭它销账
        payload = msg.get("payload", {})  # 真正的任务内容
        started = time.time()  # 执行开始时刻，算 duration_ms 用
        try:
            out = executor(payload) or {}  # 跑执行体；返回 None 时兜底成空 dict，后面 .get 不炸
            result_q.put({  # 结果先发出：协调器在 result_q 上等它
                "task_id": payload.get("task_id", ""),  # 回带任务 id，协调器按它归账
                "node": node_name,  # 哪个节点干的
                "pid": mp.current_process().pid,  # 哪个进程干的（验证跨进程）
                "content": str(out.get("content", "")),  # 结论文本，统一转成 str
                "success": bool(out.get("success", True)),  # 成功标记，缺省视为成功
                "delegate_request": out.get("delegate_request"),  # 透传委派请求（没有就是 None）
                "depth": int(payload.get("depth", 1)),  # 回带当前深度，协调器的深度闸要用
                "duration_ms": int((time.time() - started) * 1000),  # 本次执行耗时（毫秒）
            })
            ack_q.put(("ack", tag))           # 成功才 ack：结果发出去了，这条消息算消费完
        except Exception as e:
            # 【nack 而不是静默吞】代理收到 nack 立即重投；反复失败由代理的
            #   重试上限送进死信，节点这边不无限重试同一个毒消息
            ack_q.put(("nack", tag))  # 失败回执：代理凭 tag 找到在途消息立即重投
            _logger.warning(f"节点[{node_name}] 执行异常(已nack): {type(e).__name__}: {e}")  # 异常留痕：类型+消息，不堆栈刷屏


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

    def __init__(self, queue_factory: _QueueFactory,  # queue_factory：按进程/线程模式造队列的工厂
                 visibility_timeout: float = 4.0, max_retries: int = 3):  # visibility_timeout：投出多久没 ack 算失联；max_retries：重投上限
        self._qf = queue_factory  # 留存工厂：register 时还要再造订阅队列
        self._visibility_timeout = visibility_timeout   # 投出多久没 ack 视为节点失联
        self._max_retries = max_retries                 # 最多重投几次，超了进死信
        self._publish_q = queue_factory.new_queue()     # 发布请求（发布方 → 代理）
        self.ack_q = queue_factory.new_queue()          # ack/nack 回执（节点 → 代理）
        self._subs: dict[str, list[Any]] = {}           # topic → 订阅节点的投递队列列表
        self._rr: dict[str, int] = {}                   # topic → 轮询游标（消费组负载均衡）
        self._inflight: dict[str, dict] = {}            # tag → {msg, ts}
        self._dlq: list[dict] = []                      # 死信留档
        self._stats = {"published": 0, "delivered": 0, "acked": 0,  # 五本流水账：发布数/投递数/确认数
                       "redelivered": 0, "dead_lettered": 0}  # 重投数/死信数，run() 末尾原样返回给调用方
        # 【为什么用 RLock】重投路径会被持锁调用（nack 处理/超时巡检都在锁内
        #   弹出在途条目后直接重投），普通 Lock 不可重入会当场死锁
        self._lock = threading.RLock()  # 可重入锁：保护订阅表/在途账/统计
        self._stop = threading.Event()  # 停止旗标：路由循环每轮检查一次
        self._thread: Optional[threading.Thread] = None  # 后台路由线程句柄，start() 时才创建

    # 【什么】注册一个订阅者到 topic，返回它的专属投递队列
    def register(self, topic: str) -> Any:  # topic：订阅的主题名；返回该订阅者私有的投递队列
        dq = self._qf.new_queue()  # 给这个订阅者单独造一条队列：消费组内每人一条，轮询往里投
        with self._lock:
            self._subs.setdefault(topic, []).append(dq)  # 挂到订阅表：该 topic 已有队列则追加，没有则新建列表
        return dq  # 调用方（add_node）把它交给节点主循环去消费

    # 【什么】发布一条消息到 topic（队列自带线程安全，这里不用拿锁）
    def publish(self, topic: str, payload: dict) -> str:  # topic：目标主题；payload：任务内容；返回消息 tag
        tag = uuid.uuid4().hex[:8]  # 8 位短 id 当消息身份证：在途账/ack 销账全靠它
        self._publish_q.put(("pub", topic, {"tag": tag, "payload": payload, "retries": 0}))  # 进发布队列：路由线程异步取走再分发，retries 从 0 计
        return tag  # 返回 tag 给发布方（本模拟里主要供排查日志对照）

    # 【什么】投毒丸给所有订阅队列（停止节点用）
    def send_poison(self) -> None:
        with self._lock:  # 持锁遍历订阅表，防止遍历期间有新订阅者注册
            for qs in self._subs.values():  # 每个 topic 的订阅队列列表
                for dq in qs:  # 逐个订阅者投毒丸
                    try:
                        dq.put(None)  # None 即毒丸：节点 get 到它就 break 退出
                    except Exception:
                        pass  # 单条队列投递失败不拖累其他节点收毒丸

    # 【什么】代理主循环：发布→路由，回执→销账，在途超时→重投/死信
    def _loop(self) -> None:
        while not self._stop.is_set():  # 收到停止旗标就退出循环，线程随之结束
            # ① 取发布请求并路由（同 topic 轮询分发 = 消费组负载均衡）
            try:
                while True:  # 排空式处理：一轮把积压的发布请求全部路由掉
                    kind, topic, msg = self._publish_q.get_nowait()  # 非阻塞取：取不到就抛 Empty 跳出内层循环
                    if kind != "pub":
                        continue  # 只认 pub 类型，其他控制消息（预留）直接跳过
                    self._route(topic, msg, first=True)  # 首投：计 published，不重投计数
            except _queue.Empty:
                pass  # 发布队列排空，进入下一步
            # ② 取回执：ack 销账；nack 立即重投（或死信）
            try:
                while True:  # 同样排空式处理回执
                    action, tag = self.ack_q.get_nowait()  # 非阻塞取回执（ack 或 nack + 消息 tag）
                    with self._lock:
                        entry = self._inflight.pop(tag, None)  # 不管 ack/nack 都先从在途账销掉：后面要重投会重新入账
                    if entry is None:
                        continue  # 迟到/重复的回执（消息已销账或已死信）直接丢弃
                    if action == "ack":
                        with self._lock:
                            self._stats["acked"] += 1  # 确认数 +1，这条消息生命周期结束
                    else:
                        self._route(entry["msg"]["payload"].get("__topic", "tasks"),  # 从 payload 找回原 topic（submit 时塞的 __topic）
                                    entry["msg"], first=False)  # first=False 走重投分支：retries+1，超限进死信
            except _queue.Empty:
                pass  # 回执队列排空，进入下一步
            # ③ 在途超时巡检：投出太久没 ack 的消息重投/死信
            now = time.time()  # 当前时刻，跟投递时刻比对
            with self._lock:
                expired = [t for t, e in self._inflight.items()  # 找出所有"投出超过可见性超时仍无 ack"的消息 tag
                           if now - e["ts"] >= self._visibility_timeout]
                expired_msgs = [self._inflight.pop(t)["msg"] for t in expired]  # 超时消息从在途账弹出（锁内只弹账，不路由，缩短持锁时间）
            for msg in expired_msgs:
                self._route(msg["payload"].get("__topic", "tasks"), msg, first=False)  # 锁外重投：路由里还会再拿锁，靠 RLock 可重入也不怕
            time.sleep(0.05)  # 每轮睡 50ms：既不空转烧 CPU，又让重投延迟可忽略

    # 【什么】把消息投给 topic 的下一个订阅者；重投耗尽则进死信
    # 【输入输出对照】max_retries=2：首投失败 → retries=1 重投；
    #   再失败 → retries=2 重投；又失败 → retries(2) >= max_retries(2) → 死信
    def _route(self, topic: str, msg: dict, first: bool) -> None:  # topic：目标主题；msg：消息本体（含 tag/retries）；first：是否首投
        with self._lock:  # 整套路由都在锁内：轮询游标、在途账、统计要一致地变
            if not first:
                if msg["retries"] >= self._max_retries:  # 重投额度已耗尽
                    self._stats["dead_lettered"] += 1  # 死信数 +1
                    self._dlq.append(dict(msg))  # 拷一份进死信留档（防原消息被后续改动污染留档）
                    _logger.warning(f"消息 {msg['tag']} 重投 {msg['retries']} 次仍失败，进死信")  # 死信告警：排查节点为何反复失败
                    return  # 死信后不再投递，这条消息的生命周期到此结束
                msg["retries"] += 1  # 重投计数 +1
                self._stats["redelivered"] += 1  # 重投数 +1
            else:
                self._stats["published"] += 1  # 首投：发布数 +1
            subs = self._subs.get(topic) or []  # 取该 topic 的订阅队列列表
            if not subs:
                return                          # 没订阅者：丢弃（模拟环境，记日志足够）
            idx = self._rr.get(topic, 0) % len(subs)  # 轮询游标取模：订阅者增减时索引永不越界
            self._rr[topic] = idx + 1  # 游标前移：下一条消息给下一个订阅者（消费组负载均衡）
            subs[idx].put(msg)  # 投递到选中订阅者的专属队列
            self._stats["delivered"] += 1  # 投递数 +1
            self._inflight[msg["tag"]] = {"msg": msg, "ts": time.time()}  # 入在途账：开始等 ack，记下投递时刻供超时巡检

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return  # 已在跑就不重复起线程（幂等启动）
        self._thread = threading.Thread(target=self._loop, daemon=True, name="simbroker")  # 守护线程跑路由循环：主进程退出它不挡路
        self._thread.start()  # 启动路由线程

    def stop(self) -> None:
        self._stop.set()  # 立停止旗标：路由循环下一轮看到就退出
        if self._thread:
            self._thread.join(timeout=2.0)  # 最多等 2 秒让线程体面收尾，等不到也放行（守护线程不碍事）

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)  # 拷一份快照返回：调用方随便改不影响账本

    @property
    def dlq(self) -> list[dict]:
        with self._lock:
            return list(self._dlq)  # 同样拷快照：死信列表只读外曝


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

    def __init__(self, max_depth: int = 1, task_timeout: float = 12.0,  # max_depth：委派深度上限；task_timeout：等结果的整体超时秒数
                 visibility_timeout: float = 4.0, max_retries: int = 3,  # visibility_timeout：消息可见性超时；max_retries：消息重投上限
                 use_processes: bool = True):  # use_processes：True 节点跑独立进程（真分布式语义），False 跑线程（测试提速）
        self.max_depth = max_depth                 # 四件套①：委派链深度上限
        self.task_timeout = task_timeout           # 四件套②：子任务等结果超时
        self.use_processes = use_processes  # 记录模式：add_node 据此决定起进程还是线程
        self._qf = _QueueFactory(use_processes)  # 队列工厂：按模式造跨进程或同进程队列
        self.broker = SimulatedBroker(self._qf, visibility_timeout=visibility_timeout,  # 消息代理：路由/重投/死信全在它手里
                                      max_retries=max_retries)
        self.result_q = self._qf.new_queue()  # 全局结果队列：所有节点共用它回传结果（凭 task_id 归账）
        self._nodes: list = []                     # 进程/线程句柄
        self._denied: list[dict] = []              # 被深度闸拒绝的委派链记录

    # ------------------------------------------------------------------
    # 【什么】加一个 Agent 节点（进程或线程，订阅某 topic）
    # ------------------------------------------------------------------
    def add_node(self, name: str, topic: str,  # name：节点名（日志与结果里露面）；topic：订阅的队列主题
                 executor: Callable[[dict], dict] = _default_node_executor) -> None:  # executor：节点收消息后的执行体，缺省用模拟执行器
        dq = self.broker.register(topic)  # 注册订阅：拿到该节点专属的投递队列
        args = (name, dq, self.broker.ack_q, self.result_q, executor)  # 节点主循环的参数包（须可 pickle）
        if self.use_processes:
            p = mp.Process(target=_node_main, args=args, daemon=True, name=f"node-{name}")  # 起独立进程跑 _node_main
            p.start()  # 启动进程
            self._nodes.append(p)  # 挂到节点句柄列表，stop 时回收
        else:
            t = threading.Thread(target=_node_main, args=args, daemon=True, name=f"node-{name}")  # 起线程跑同一段主循环（测试模式）
            t.start()  # 启动线程
            self._nodes.append(t)  # 同样挂到列表统一回收
        _logger.info(f"节点已注册: {name} topic={topic} 模式={'进程' if self.use_processes else '线程'}")  # 记录节点与模式，便于排查日志

    # ------------------------------------------------------------------
    # 【什么】发布任务（支持冗余份数：同任务发 k 份做冲突检测）
    # 【作用】消息里带 __topic 供代理重投时找回路由；seeds 给 biased 模式变体
    # ------------------------------------------------------------------
    def submit(self, task: dict) -> None:  # task：任务字典（含 task_id/instruction/mode/redundancy/seeds 等）
        redundancy = int(task.get("redundancy", 1))  # 冗余份数：缺省 1 份不冗余
        seeds = task.get("seeds") or [None] * redundancy  # 种子列表：没有就用 None 补够 redundancy 份（非 biased 模式时不影响结果）
        for k in range(redundancy):  # 逐份投递
            payload = dict(task)  # 深拷贝一份：每份独立改 seed，不互相污染
            payload["__topic"] = task.get("topic", "tasks")  # 塞进 topic 别名：代理重投时凭它找回路由（publish 的 topic 和 msg 分开传的，要留在 payload 里）
            if seeds[k] is not None:
                payload["seed"] = seeds[k]  # 有种子时塞入：biased 模式据此出不同结论
            self.broker.publish(payload["__topic"], payload)  # 经代理发布：轮询选一个订阅该 topic 的节点投递

    # ------------------------------------------------------------------
    # 【四件套②+③】收集结果：deadline 内等齐；同任务多份结果全部收下（冲突要用）
    # 【输入输出对照】task_timeout=1.0、t2 是 slow(3s)：
    #   收到 t1 → pending 剩 {t2}；1 秒到 → got 里没有 t2 → degraded=[{"task_id":"t2"}]
    # 【为什么要宽限期 grace】冗余副本同 task_id：第一份到达就清空 pending，
    #   若立刻收工，第 2/3 份副本可能还在路上——冲突检测就永远只看到 1 份。
    #   收齐后多等 grace 秒（默认 0.3s），没新消息才算真收工。
    # ------------------------------------------------------------------
    def collect(self, expected_ids: list[str], timeout: float, grace: float = 0.3) -> dict[str, list[dict]]:  # expected_ids：本批应收的任务 id；timeout：整体超时；grace：收齐后再等的宽限期
        deadline = time.time() + timeout  # 算绝对截止时刻：跟 wall clock 对齐
        pending = set(expected_ids)  # 还没收到结果的任务集合：每收到一个就 discard 一个
        got: dict[str, list[dict]] = {}  # 已收账本：task_id → 该任务的所有副本结果列表
        idle_since: Optional[float] = None  # 首次收齐的时刻：宽限期计时起点，没收齐时保持 None
        while time.time() < deadline:  # 整体超时闸：到期直接退出循环
            try:
                r = self.result_q.get(timeout=0.1)  # 短超时取结果：既能响应，又能定期回去查 deadline
            except _queue.Empty:
                if not pending:  # 已全部收齐
                    if idle_since is None:
                        idle_since = time.time()  # 第一次发现收齐：记下时刻，开始算宽限期
                    elif time.time() - idle_since >= grace:
                        break                      # 收齐且宽限期内无新副本 → 收工
                continue  # 没收到东西就回去继续等（或重新查 deadline）
            idle_since = None  # 收到新副本就重置宽限期计时：可能还有后续副本在路上
            tid = r.get("task_id", "")  # 取任务 id 归账
            got.setdefault(tid, []).append(r)  # 副本一律入账，冲突检测靠全量
            pending.discard(tid)  # 该任务已有结果，从待收集合移除
        return got  # 返回账本：可能是全部收齐，也可能是超时截断

    # ------------------------------------------------------------------
    # 【四件套④】冲突收敛：同任务多份结果内容不一致 → 多数决，平票取先到
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_conflicts(results_by_task: dict[str, list[dict]]) -> tuple[dict, list[dict]]:  # results_by_task：task_id → 副本结果列表
        answers: dict[str, str] = {}  # 最终答案：task_id → 选中的内容
        conflicts: list[dict] = []  # 冲突记录：哪些任务出现了分歧
        for tid, rs in results_by_task.items():  # 逐个任务处理
            if not rs:
                continue  # 没结果的任务跳过去（降级名单里体现，不在这里处理）
            contents = [r.get("content", "") for r in rs]  # 把所有副本的 content 拎出来
            if len(set(contents)) <= 1:
                answers[tid] = contents[0]  # 全一致：没有冲突，直接采纳
                continue  # 处理下一个任务
            counts: dict[str, int] = {}  # 各结论的出现次数账本
            for c in contents:
                counts[c] = counts.get(c, 0) + 1  # 计数 +1
            best = max(counts.values())  # 最高票数
            winners = [c for c, n in counts.items() if n == best]  # 所有达到最高票数的结论
            if len(winners) == 1:
                winner, rule = winners[0], "多数决"  # 唯一最高票：多数决获胜
            else:
                winner, rule = contents[0], "平票取先到"  # 平票：取最先到达的那份结论（时间序保底）
            answers[tid] = winner  # 确定最终答案
            conflicts.append({"task_id": tid, "variants": contents, "winner": winner, "rule": rule})  # 留痕：原始变体、胜出者、判定规则
        return answers, conflicts  # 最终答案 + 冲突明细

    # ------------------------------------------------------------------
    # 【四件套①】委派链深度闸：节点结果里带 delegate_request 时逐层放行/拒绝
    # 【输入输出对照】max_depth=1：第 1 层任务请求委派 → 子任务 depth=2 > 1 →
    #   denied 记录 {"task_id":"t1","to_depth":2,"reason":"..."}，链路到此为止
    # ------------------------------------------------------------------
    def _process_delegations(self, results: list[dict], deadline: float) -> list[dict]:  # results：本批已收的结果；deadline：整条链的绝对超时时刻
        chained: list[dict] = []  # 委派链累计产物：每一轮子任务的结果都追加到这里
        frontier = [r for r in results if r.get("delegate_request")]  # 当前层要处理委派请求的结果集合
        rounds = 0  # 轮次计数：兜底防死循环
        while frontier and rounds < 5:            # rounds 兜底：深度闸正常时不会走到这
            rounds += 1  # 轮次 +1
            children: list[dict] = []  # 本轮要派发的子任务清单
            for r in frontier:
                req = r["delegate_request"]  # 取出委派请求
                r["delegate_request"] = None      # 消费掉，防二次处理
                child_depth = int(r.get("depth", 1)) + 1  # 子任务深度 = 当前 +1
                if child_depth > self.max_depth:  # 深度闸拦截
                    self._denied.append({"task_id": r.get("task_id", ""),  # 记拒绝账
                                         "to_depth": child_depth,
                                         "reason": f"超过委派深度上限 {self.max_depth}"})
                    _logger.info(f"分布式委派被拒: {r.get('task_id')} 深度 {child_depth}")  # 拒绝也记日志，便于排查
                    continue  # 不再往下派发，跳到下一个委派请求
                children.append({  # 通过深度闸：组装子任务
                    "task_id": f"{r.get('task_id', 'task')}>d{child_depth}",  # 子任务 id：父 id + 深度标记，链路可追溯
                    "topic": req.get("topic", "tasks"),  # 投到指定 topic，缺省回 tasks
                    "instruction": req.get("instruction", ""),  # 委派指令
                    "depth": child_depth,  # 回带深度，下轮再委派时继续 +1
                })
            if not children:
                break  # 全部被深度闸拒了：没有活可派，跳出循环
            for c in children:
                self.submit(c)  # 逐个派发子任务
            remaining = max(0.5, deadline - time.time())  # 剩余预算：至少留 0.5 秒，防 collect 拿到 0/负数超时
            got = self.collect([c["task_id"] for c in children], remaining)  # 收集本轮子任务结果
            new_results = [r for rs in got.values() for r in rs]  # 拍平成列表
            chained.extend(new_results)  # 累计进委派链产物
            frontier = [r for r in new_results if r.get("delegate_request")]  # 新结果里再冒出的委派请求，成为下一轮 frontier
        return chained  # 返回整条委派链累计产物

    # ------------------------------------------------------------------
    # 【什么】一次完整分布式运行：提交 → 收集 → 委派链 → 冲突 → 降级 → 汇总
    # ------------------------------------------------------------------
    def run(self, tasks: list[dict]) -> dict:  # tasks：本批任务列表；返回汇总报告（answers/results/conflicts/degraded/denied/broker_stats/duration_ms）
        self.broker.start()  # 幂等启动代理路由线程
        started = time.time()  # 整体开始时刻
        deadline = started + self.task_timeout  # 绝对超时时刻：委派链共享这一个预算池
        ids = [t["task_id"] for t in tasks]  # 本批任务 id 清单：collect 的 pending 从这里来
        for t in tasks:
            t.setdefault("topic", "tasks")  # 没指定 topic 的任务统一进默认 tasks 主题
            t.setdefault("depth", 1)  # 初始深度 1：第一层任务
            self.submit(t)  # 逐个发布
        got = self.collect(ids, self.task_timeout)  # 收第一轮结果（可能含副本和 delegate_request）
        results = [r for rs in got.values() for r in rs]  # 拍平成列表，喂给委派链处理器
        # 委派链：处理节点冒出的 delegate_request（受深度闸约束）
        chained = self._process_delegations(results, deadline)  # 拿到整条委派链累计产物
        for r in chained:
            got.setdefault(r["task_id"], []).append(r)  # 子任务结果也归账进 got，让冲突检测与降级名单能看到
        # 冲突收敛 + 降级名单
        answers, conflicts = self._resolve_conflicts(got)  # 四件套④：多数决收敛
        degraded = [{"task_id": tid, "reason": f"等待结果超时({self.task_timeout}s) 或已进死信"}  # 四件套②+③：没收到结果的进降级名单
                    for tid in ids if not got.get(tid)]
        return {
            "answers": answers,  # 最终答案（冲突已收敛）
            "results": [r for rs in got.values() for r in rs],  # 原始结果全集（含副本，供审计）
            "conflicts": conflicts,  # 冲突明细（variants/winner/rule）
            "degraded": degraded,  # 降级名单（超时/死信的任务）
            "denied": list(self._denied),  # 深度闸拒绝记录
            "broker_stats": self.broker.stats(),  # 代理流水账（published/delivered/acked/redelivered/dead_lettered）
            "duration_ms": int((time.time() - started) * 1000),  # 整体耗时
        }

    # ------------------------------------------------------------------
    # 【什么】停止：投毒丸收节点 → 停代理线程 → 回收子进程
    # ------------------------------------------------------------------
    def stop(self) -> None:
        try:
            self.broker.send_poison()  # 先投毒丸：让节点主循环拿到 None 就 break
        except Exception:
            pass  # 毒丸投递失败不阻塞后续回收
        time.sleep(0.3)                            # 给节点一点时间消化毒丸
        self.broker.stop()  # 再停代理路由线程
        for n in self._nodes:
            if isinstance(n, mp.Process) and n.is_alive():
                n.terminate()  # 还没退出的子进程强杀（线程模式则依赖 daemon 自然退出）
        self._nodes = []  # 清空句柄列表
