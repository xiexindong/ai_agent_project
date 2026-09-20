"""
阶段2 模拟演示：Redis 热会话缓存 + 消息队列可靠写
==================================================
只用标准库，零第三方依赖，直接跑：
    python examples/stage2_mock_demo.py

【为什么有这个脚本】
    项目当前（阶段1）是同步直写：/chat 生成 task_id → 同步写 MySQL + PG。
    阶段2 要解决两件事，这里用"假组件"把效果演出来：
      1. Redis：把正在聊的热会话放内存，减少 PG 读压力
      2. 消息队列：把写 PG 从同步改成异步投递 + 自动重试 + 幂等 + 死信

【四个假组件 → 真组件的对应关系】
    FakeRedis   dict + LRU + TTL          → redis / Redis Cluster
    FakePG      list 当表，可注入故障率    → PostgreSQL + pgvector
    FakeMySQL   list 当表，注入故障率      → MySQL 业务库
    FakeBroker  Queue + 消费者线程 + DLQ   → RocketMQ / Kafka + 消费者组

【四幕剧情】
    第一幕 无缓存：每轮都回源读 PG          → PG 读 20 次
    第二幕 加缓存：热会话命中缓存           → PG 读 3 次，命中率 85%
    第三幕 PG 抖动 40%：直写 vs MQ 重试     → 直写丢 8 条，MQ 丢 0 条
    第四幕 幂等与死信：重复投递 / 永久失败  → 去重只落 1 条，毒消息进 DLQ

【随机数的坑】
    全部场景都 random.seed(42)，保证每次跑出来的数字一模一样。
    面试现场演示最怕"上次丢 8 条、这次丢 5 条"，锁种子是必须的。

【实测运行输出】（seed=42，复跑必然一致）
    第一幕  没有 Redis          → PG 读 20 次
    第二幕  加 Redis（容量 3）   → 命中 17 / 未命中 3，命中率 85.0%，PG 读 3 次
            容量降到 2          → 写 S1 S2 S3 后剩 ['S2','S3']，淘汰 S1
    第三幕A PG 抖动 40% 直写    → MySQL 20 条，PG 只落地 8 条，丢 12 条
                               对账 mysql_only = 12 条
    第三幕B 同样抖动 走 MQ      → MySQL 20 条（尝试 29 次，幂等拦重复 9 条）
                               PG 20 条（尝试 29 次，重试 9 次），丢 0 条
                               对账 mysql_only = 0，pg_only = 0
    第四幕  同一条消息投 3 次    → PG 里 1 条，幂等拦下重复 2 条
            毒消息 1 条         → 重试 3 次后进 DLQ，DLQ 长度 1
"""
import json
import random
import queue
import threading
import time
from collections import OrderedDict

random.seed(42)

MAX_RETRIES = 3          # 重试 3 次仍失败 → 进死信队列
QUARANTINE_RETRY = 0     # 重试间隔设为 0，演示不想真等几秒


# ==========================================================================
# 组件一：FakeRedis（模拟 Redis）
# ==========================================================================
class FakeRedis:
    """用 OrderedDict + TTL 模拟 Redis：容量满时按 LRU 淘汰最久没用的 key。

    【为什么要 LRU 容量限制】
        缓存不设上限 = 内存迟早被会话撑爆。Redis 用 maxmemory-policy 淘汰，
        这里用 OrderedDict + popitem(last=False) 复刻同样的行为。

    【输入 → 输出对照】
        set("S1", [...]) → 缓存里有 1 个 key
        get("S1")        → 命中，hits 从 0 变 1
        get("不存在")     → None，misses 从 0 变 1
    """

    def __init__(self, max_keys: int = 3, default_ttl: float | None = None):
        self._store = OrderedDict()
        self.max_keys = max_keys
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key):
        if key not in self._store:
            self.misses += 1
            return None
        value, expire_at = self._store[key]
        # ------------------------------------------------------------------
        # 【TTL 过期为什么要"读的时候才删"】
        #   Redis 是惰性删除 + 定期抽样，不会给每个 key 起定时器。
        #   不这么做：几百万个定时器把内存和 CPU 吃掉。
        # ------------------------------------------------------------------
        if expire_at is not None and time.time() > expire_at:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key, value, ttl: float | None = None) -> None:
        expire = ttl if ttl is not None else self.default_ttl
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.time() + expire if expire else None)
        while len(self._store) > self.max_keys:
            self._store.popitem(last=False)
            self.evictions += 1

    def delete(self, key) -> None:
        self._store.pop(key, None)

    def keys(self) -> list:
        return list(self._store.keys())

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits / total * 100:.1f}%" if total else "0%",
            "evictions": self.evictions,
        }


# ==========================================================================
# 组件二：两座"数据库"
# ==========================================================================
class FakeMySQL:
    """模拟 MySQL 业务库（审计日志）。业务库一般最稳，故障率默认 0。"""

    def __init__(self, failure_rate: float = 0.0):
        self.rows = []
        self.attempts = 0
        self.failures = 0
        self.deduped = 0
        self.failure_rate = failure_rate

    def insert_audit(self, task_id: str, user_code: str) -> bool:
        # ------------------------------------------------------------------
        # 【为什么 MySQL 这一层也要幂等】这是整个 demo 最容易被忽略的一点：
        #   重试重放的是"整条 handler"，不是只重放 PG 那一步。
        #   第一次 MySQL 写成功、PG 抖动失败 → 重试时 MySQL 会再写一次，
        #   审计表多一份假记录，部门账单跟着重复计数。
        #
        # 【生产上的正确做法】
        #   task_id 建 UNIQUE 索引 + INSERT ... ON DUPLICATE KEY UPDATE，
        #   靠数据库约束保证原子幂等。注意别用"先 SELECT 再 INSERT"：
        #   并发下两个消费者同时查都存在不到、同时插入，竞态依然出脏数据。
        # ------------------------------------------------------------------
        self.attempts += 1
        if random.random() < self.failure_rate:
            self.failures += 1
            raise RuntimeError("MySQL 连接抖动")
        if any(r["task_id"] == task_id for r in self.rows):
            self.deduped += 1
            return False
        self.rows.append({"task_id": task_id, "user_code": user_code})
        return True


class FakePG:
    """模拟 PG 会话库（chat_history）。task_id 唯一 = 幂等的落点。"""

    def __init__(self, failure_rate: float = 0.0):
        self.rows = []
        self.attempts = 0
        self.failures = 0
        self.read_count = 0
        self.failure_rate = failure_rate
        self.poison_tasks = set()      # 放进去的 task_id 永远写失败（毒消息）

    def insert_history(self, task_id: str, session_id: str, content: str) -> bool:
        """返回 True = 真写进去了；False = 已存在，幂等跳过（不算失败）。"""
        self.attempts += 1
        if task_id in self.poison_tasks:
            self.failures += 1
            raise RuntimeError(f"PG 数据非法: {task_id}")
        if random.random() < self.failure_rate:
            self.failures += 1
            raise RuntimeError("PG 连接抖动")
        if any(r["task_id"] == task_id for r in self.rows):
            return False               # 幂等：同一个号已经落过，直接跳过
        self.rows.append({"task_id": task_id, "session_id": session_id, "content": content})
        return True

    def read_session(self, session_id: str) -> list:
        self.read_count += 1
        return [r for r in self.rows if r["session_id"] == session_id]


# ==========================================================================
# 组件三：FakeBroker（模拟消息队列）
# ==========================================================================
class Message:
    """一条消息 = task_id + 载荷 + 已重试次数。task_id 同时是幂等键。"""

    def __init__(self, task_id: str, payload: dict):
        self.task_id = task_id
        self.payload = payload
        self.retries = 0


class FakeBroker:
    """生产者投递即返回，消费者后台不断重试，超次数丢进死信队列。

    【三件事缺一不可】
        重试：对付抖动（PG 抖动 1 秒就恢复，重试一次就成功）
        幂等：对付重试本身带来的重复投递（不幂等 = 重试制造脏数据）
        死信：对付重试治不好的错（数据非法），否则无限重试 = 拖垮整个队列
    """

    def __init__(self, mysql: FakeMySQL, pg: FakePG, max_retries: int = MAX_RETRIES):
        self.mysql = mysql
        self.pg = pg
        self.max_retries = max_retries
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0
        self.published = 0
        self.committed = 0          # 真正写进 PG 的条数
        self.deduped = 0            # 幂等拦掉的重复条数
        self.total_retries = 0
        self.dlq: list = []
        self._worker = threading.Thread(target=self._consume_loop, daemon=True)
        self._worker.start()

    def publish(self, msg: Message) -> None:
        with self._lock:
            self._pending += 1
        self.published += 1
        self._queue.put(msg)

    def _handle(self, msg: Message) -> bool:
        # ------------------------------------------------------------------
        # 【顺序为什么是先 MySQL 后 PG】
        #   业务单据是"钱和证据"，先落它；AI 会话允许晚几毫秒。
        #   PG 失败可以重试补偿，MySQL 失败才是真事故。
        # ------------------------------------------------------------------
        self.mysql.insert_audit(msg.task_id, msg.payload["user_code"])
        written = self.pg.insert_history(
            msg.task_id, msg.payload["session_id"], msg.payload["content"]
        )
        if written:
            self.committed += 1
        else:
            self.deduped += 1        # 重复投递被幂等拦下
        return True

    def _consume_loop(self) -> None:
        while True:
            msg = self._queue.get()
            if msg is None:
                break
            try:
                self._handle(msg)
            except Exception as e:
                if msg.retries < self.max_retries:
                    msg.retries += 1
                    self.total_retries += 1
                    self._queue.put(msg)     # 重投递 ≈ RocketMQ 的重试队列
                else:
                    msg.last_error = str(e)
                    self.dlq.append(msg)     # 重试耗尽 = 死信
            finally:
                with self._lock:
                    self._pending -= 1

    def wait_idle(self, timeout: float = 5.0) -> None:
        """等队列排空（真实场景是看消费位点 lag，这里看 pending 计数）。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                done = self._pending == 0
            if done and self._queue.empty():
                return
            time.sleep(0.02)


# ==========================================================================
# 辅助：跨库对账（复用阶段1 的思路，按 task_id 比对两库）
# ==========================================================================
def reconcile(mysql: FakeMySQL, pg: FakePG) -> dict:
    m = {r["task_id"] for r in mysql.rows}
    p = {r["task_id"] for r in pg.rows}
    return {
        "mysql": len(m),
        "pg": len(p),
        "mysql_only": sorted(m - p),     # MySQL 有 PG 没有 = AI 会话丢了
        "pg_only": sorted(p - m),
    }


# ==========================================================================
# 第一幕：没有缓存时，每轮都回源读 PG
# ==========================================================================
def act1_no_cache():
    print("\n" + "=" * 66)
    print("第一幕  没有 Redis：20 轮对话，每轮都回源读 PG")
    print("=" * 66)
    pg = FakePG()
    for i in range(20):
        pg.insert_history(f"T-{i:03d}", "S1", f"第{i + 1}轮问答")
    # 一轮=s1,s1,s1...（同一会话），但没有缓存，build_context 每次都打 PG
    for i in range(20):
        pg.read_session("S1")
    print(f"PG 写入条数: {len(pg.rows)}")
    print(f"PG 读取次数: {pg.read_count}   ← 每轮一次回源，压力全在 PG")
    return pg.read_count


# ==========================================================================
# 第二幕：加 Redis 后，热会话命中缓存
# ==========================================================================
def act2_with_cache():
    print("\n" + "=" * 66)
    print("第二幕  加 Redis（容量 3，LRU 淘汰）：同样的 20 轮")
    print("=" * 66)
    pg = FakePG()
    cache = FakeRedis(max_keys=3, default_ttl=3600)
    # 20 轮 = 会话 S1 占 12 轮、S2 占 5 轮、S3 占 3 轮（真实分布：少数会话是热的）
    plan = ["S1"] * 12 + ["S2"] * 5 + ["S3"] * 3
    for i, sid in enumerate(plan):
        pg.insert_history(f"T-{i:03d}", sid, f"第{i + 1}轮问答")
    for sid in plan:
        # Cache-Aside 套路：先查缓存，没有才回源，回源后回填
        history = cache.get(sid)
        if history is None:
            history = pg.read_session(sid)
            cache.set(sid, history)
    info = cache.stats()
    print(f"会话分布: S1×12  S2×5  S3×3")
    print(f"缓存: 命中 {info['hits']} 次 / 未命中 {info['misses']} 次，命中率 {info['hit_rate']}")
    print(f"PG 读取次数: {pg.read_count}   ← 只有每个会话第一轮回源")
    print(f"LRU 淘汰次数: {info['evictions']}   ← 容量够就没淘汰（容量 3 个热会话正好放得下）")
    return pg.read_count, info["hit_rate"]


# ==========================================================================
# 第二幕补充：把容量调小，看 LRU 到底淘汰谁
# ==========================================================================
def act2b_lru_eviction():
    print("\n" + "-" * 66)
    print("补充  缓存容量从 3 降到 2，看 LRU 淘汰谁（面试常问淘汰策略）")
    print("-" * 66)
    cache = FakeRedis(max_keys=2)
    for sid in ("S1", "S2", "S3"):
        cache.set(sid, [f"{sid} 的历史对话"])
    print(f"依次写入 S1 S2 S3 → 缓存里的 key: {cache.keys()}")
    print(f"淘汰次数: {cache.evictions}   ← S1 最久没被用过，被挤掉")
    print("          S1 的下一轮会 miss 一次重新回源，不会报错、不会丢数据")
    print("          Redis 的 allkeys-lru / volatile-lru 就是这个行为")


# ==========================================================================
# 第三幕 A：PG 抖动 40%，同步直写会丢数据
# ==========================================================================
def act3_direct_write():
    print("\n" + "=" * 66)
    print("第三幕A  PG 抖动 40%，同步直写（现在阶段1 的写法）")
    print("=" * 66)
    mysql = FakeMySQL()
    pg = FakePG(failure_rate=0.4)
    lost = 0
    for i in range(20):
        task_id = f"T-direct-{i:03d}"
        mysql.insert_audit(task_id, "U001")
        try:
            pg.insert_history(task_id, "S1", f"第{i + 1}轮问答")
        except Exception as e:
            lost += 1                 # 抛异常给用户 = 用户这一轮的对话记录没了
            if i == 0:
                print(f"第 1 条就失败: {e}")
    print(f"MySQL 审计: {len(mysql.rows)} 条")
    print(f"PG 会话:    {len(pg.rows)} 条（尝试 {pg.attempts} 次，失败 {pg.failures} 次）")
    print(f"丢失对话:   {lost} 条   ← 用户看到的就是'刚才那轮没存上'")
    print(f"对账结果:   {reconcile(mysql, pg)}")
    return len(pg.rows), lost


# ==========================================================================
# 第三幕 B：同样的抖动，走 MQ 重试
# ==========================================================================
def act4_mq_write():
    print("\n" + "=" * 66)
    print("第三幕B  同样的抖动 40%，改成 MQ 投递 + 自动重试")
    print("=" * 66)
    mysql = FakeMySQL()
    pg = FakePG(failure_rate=0.4)
    broker = FakeBroker(mysql, pg, max_retries=MAX_RETRIES)
    for i in range(20):
        broker.publish(Message(
            f"T-mq-{i:03d}",
            {"session_id": "S1", "user_code": "U001", "content": f"第{i + 1}轮问答"},
        ))
    broker.wait_idle()
    print(f"投递消息:   {broker.published} 条（投递即返回，不等 PG 结果）")
    print(f"MySQL 审计: {len(mysql.rows)} 条（消费端尝试 {mysql.attempts} 次，"
          f"幂等拦下重复 {mysql.deduped} 条）")
    print("       ← 重试会重放整条 handler，所以 MySQL 那一步也必须幂等")
    print(f"PG 会话:    {broker.committed} 条（尝试 {pg.attempts} 次，失败 {pg.failures} 次，重试 {broker.total_retries} 次）")
    print(f"死信队列:   {len(broker.dlq)} 条")
    print(f"对账结果:   {reconcile(mysql, pg)}")
    print("→ 同样是 PG 抖掉一半，重试兜住了，用户那一轮没丢")
    return broker.committed, broker.total_retries, len(broker.dlq)


# ==========================================================================
# 第四幕：幂等（重复投递）+ 死信（毒消息）
# ==========================================================================
def act5_idempotent_and_dlq():
    print("\n" + "=" * 66)
    print("第四幕  幂等消费 + 死信队列（MQ 的两个必备件）")
    print("=" * 66)
    pg = FakePG()
    mysql = FakeMySQL()
    broker = FakeBroker(mysql, pg, max_retries=MAX_RETRIES)

    # 场景1：网络抖动导致同一条消息被投递 3 次
    msg = Message("T-dup-001", {"session_id": "S9", "user_code": "U001", "content": "同一轮"})
    for _ in range(3):
        broker.publish(Message(msg.task_id, msg.payload))
    broker.wait_idle()
    print(f"[幂等] 同一条消息投递 3 次 → PG 里 {len(pg.rows)} 条，"
          f"幂等拦下重复 {broker.deduped} 条")
    print("       （幂等键就是 task_id：入库前先查这个号在不在）")
    print("       不做幂等的后果：网络一抖，同一轮对话存 3 遍，"
          "向量召回会把重复文本塞给 LLM")

    # 场景2：数据非法，重试多少次都没用
    pg.poison_tasks.add("T-poison-001")
    broker.publish(Message(
        "T-poison-001", {"session_id": "S9", "user_code": "U001", "content": "非法数据"},
    ))
    broker.wait_idle()
    print(f"\n[死信] 毒消息 1 条 → 重试 {MAX_RETRIES} 次后进死信队列，DLQ 长度 {len(broker.dlq)}")
    if broker.dlq:
        m = broker.dlq[0]
        print(f"       DLQ 内容: task_id={m.task_id} retries={m.retries} "
              f"last_error={m.last_error}")
    print("       不做死信的后果：这条消息永久卡住队列头，后面所有写入全堵死")


# ==========================================================================
# 总结：接线清单（换成真组件时改哪里）
# ==========================================================================
def print_wiring_checklist():
    print("\n" + "=" * 66)
    print("接线清单：换成真 Redis / 真 MQ 要动的地方")
    print("=" * 66)
    print("  1. 写侧入口只有一个：modules/memory/memory_manager.py 的 save_turn()（第 35 行）")
    print("     现在: return self.pg.save_history(...)")
    print("     改后: broker.publish(Message(task_id, payload))  ← 只改这一处")
    print("  2. 读侧入口只有一个：build_context() 里取短期记忆那一行（第 90 行）")
    print("     现在: short_term = self.pg.get_short_term_history(session_id, user_id, n)")
    print("     改后: short_term = cache.get(key) or 回源 + cache.set(key, ...)")
    print("           （key 用 f'hot:{session_id}'，别只放 session_id，留个前缀好批量清）")
    print("  3. 消费侧新增一个 worker 进程：subscribe → 幂等写 PG → ack/重试/进 DLQ")
    print("  注：task_id 已经在入口生成并落到四张表里，换 MQ 不需要重构链路")


def main():
    print("#" * 66)
    print("#  阶段2 模拟：Redis 热会话 + 消息队列可靠写（零依赖，可复现）")
    print("#" * 66)
    read_no_cache = act1_no_cache()
    read_with_cache, hit_rate = act2_with_cache()
    act2b_lru_eviction()
    print(f"\n>>> 缓存收益: PG 读次数 {read_no_cache} → {read_with_cache}，"
          f"下降 {(read_no_cache - read_with_cache) / read_no_cache * 100:.0f}%，命中率 {hit_rate}")

    pg_direct, lost = act3_direct_write()
    pg_mq, retries, dlq = act4_mq_write()
    print(f"\n>>> 可靠性对比: 同样抖动下 PG 落库 {pg_direct} 条 → {pg_mq} 条，"
          f"丢失 {lost} 条 → 0 条（靠 {retries} 次重试兜住）")

    act5_idempotent_and_dlq()
    print_wiring_checklist()
    print("\n一句话：Redis 解决'读不动'，MQ 解决'写不稳'，"
          "task_id 解决'两边对不上'。")


if __name__ == "__main__":
    main()
