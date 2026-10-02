"""
工具并发控制模块
================
回答"同一个工具实例同一时间只能处理一个请求，为什么能支持多任务并发"：
    1. ToolConcurrencyGuard —— 信号量闸门：同名工具同时最多 N 个线程在执行，
       超出的排队等待（有界并发，不是无上限放行）。
    2. ToolInstancePool    —— 实例池：给"有状态工具"（连接型/游标型）准备 N 个
       独立实例，借出-归还，各线程拿到的是不同实例，互不踩踏。

两者的分工（面试考点）：
    无状态工具（calculator 这类纯函数型）：一个实例天然可被多线程同时调，
        只需要 Guard 限住"同时在跑的请求数"，保护下游（DB/子进程）不被打爆。
    有状态工具（共享游标/连接型）：多线程进同一个实例会互相覆盖内部状态，
        必须用 Pool 让每个线程借走自己的实例，用完归还。

    【比喻：把工具想象成一家小店】
        工具实例 = 一家小店；self.cursor 等成员变量 = 店里唯一的一张桌子；
        调用一次 run() = 一个客人进店点单。
        无状态工具 = 快餐车：点单、做完、端走，谁来了都一样；
        有状态工具 = 火锅店：桌子要一直摆着（锅底、配菜是上一个客人留下的）。
        Guard（信号量，默认上限 4）= 门口保安"店里最多同时 4 个客人"；
        Pool（实例池）= 连锁火锅：4 个客人分到 4 家店，各吃各的。
        关键差异一句话：保安只管"进几个人"，不管"进来之后用哪张桌子"——
        所以 Guard 防的是"人太多挤垮房子"（下游被打爆），
        防不了"抢桌子"（数据串台）；防抢桌子的只有 Pool。

    【什么是有状态 / 无状态】
    状态 = 实例身上会被调用过程写来写去的成员变量（self.cursor、self.conn
    这类）。判断标准只有一个：入参相同，结果还永远相同吗？

        无状态：不保存任何跨调用的变量，每次调用只依赖入参。
            calculator.run({"expression": "3*7"}) 调 100 次都是 21，
            两个线程同时调各自算各自的，互不影响（全程没有 self 被写）。

        有状态：把上一次调用的影响存进 self，下次调用会读到它。
            持有数据库游标的工具：self.cursor.execute("SELECT 1") 后，
            结果集挂在 self.cursor 上；若另一线程 execute("SELECT 2")，
            前一线程 fetchall() 拿到的是 "SELECT 2" 的数据，且不报错
            （静默串台）——结果取决于"之前谁调用过"，这就是有状态。

    注意：calculator 也有 self.name / self.description，但那是注册时写好、
    执行过程从不改写的"说明书"，不算状态；要盯的是执行路径上被写坏的 self。

    【什么是游标（cursor）】
    游标是数据库客户端库里的对象，职责两件事：把 SQL 发给数据库执行、
    把查回来的结果集一行一行搬给 Python。比喻成"传菜窗口"：
    后厨（数据库）把整批菜做好放进窗口，服务员（fetch 系列）端走一份少一份。

        conn（连接）   = 接通数据库的电话线：管登录、事务（commit 才算保存）；
        cur（游标）    = 电话线上开出的传菜窗口：发 SQL、收结果都走它。

        cur.execute(sql)     只负责执行，返回 None；结果集（一批行）挂在 cur 内部
        cur.fetchone()       从结果集取 1 行，取过不给第二次；没数据返回 None
        cur.fetchall()       端走"剩余"全部行（官方原话 all REMAINING rows）
                             重复调得到 []，不报错
        cur.rowcount /       不用 fetch 就能看的账本：受影响行数 / 列名
        cur.description

    【输入输出对照】表里 3 条记录时：
        cur.execute("SELECT id, content FROM agent_memory")  # 返回 None
        cur.fetchone()  # → (1, "用户偏好深色模式")           剩 2 行
        cur.fetchall()  # → [(2, "用户生日是 3 月 1 日"), (3, "用户在上海工作")]
        cur.fetchall()  # → []                                  取空了
        INSERT 后 fetchall  → 直接抛 ProgrammingError，不是返回 []
        不 fetch 就 execute 下一条 → 旧结果集被静默丢弃，fetchall 拿到新结果

    【游标与本模块的关系】
        ① 游标是"有状态"的典型来源：结果集挂在 cur 上，execute 与 fetch
           两步之间有时间缝，多线程共用一个游标 = 串台（见 Guard docstring 时间线）。
        ② 一个连接可以开多个游标（conn.cursor() 每调一次新开一个窗口，互不干扰），
           这正是 ToolInstancePool 可行的底层原因：每个实例开自己的游标，天然不串。
           （本项目 pg_manager.py 单线程模型，全类共用 self._cur；多线程场景需池化。）
"""
import threading
import time
from typing import Any, Callable, Dict, Optional

from modules.common.logger import get_logger
from config import get_settings

_logger = get_logger("tool.concurrency")


# 【什么】同名工具的并发闸门（信号量 + 在途统计）
# 【作用】限制"同名工具同时执行中的请求数"，超出的阻塞排队，不拒绝
class ToolConcurrencyGuard:
    """按工具名限流的并发闸门：每名一把 Semaphore，全局一张统计账。

    【输入输出对照】max_concurrent=2，4 个线程同时调同一个工具：
        t1 acquire → 拿到名额（在途 1）
        t2 acquire → 拿到名额（在途 2）
        t3 acquire → 阻塞排队（在途仍 2，t3 在等）
        t4 acquire → 阻塞排队
        t1 release → 在途 1，t3 立刻拿到名额开始执行
        结果：4 个请求全部完成，但任意时刻"同时在执行"的 ≤ 2。

    【不这么写会怎样】不加闸门时 4 个请求同时进入同一个工具实例：
        无状态工具 → 下游（DB 连接 / MCP 子进程）被 4 路并发打爆；
        有状态工具 → 实例内部状态被互相覆盖，结果串台且不报错。

    【边界：闸门救不了有状态工具】默认上限是 4（tool_max_concurrent），不是 1——
        加了闸门后仍可能有 2~4 个线程同时在同一个有状态实例里，串台依旧。
        闸门只治"下游被打爆"这半；"实例内状态被覆盖"那半的正解是
        ToolInstancePool（各借各的实例），或把该工具的上限压到 1（全程串行）。

    【为什么闸门救不了：时间线对照】有状态工具的 execute 和 fetchall 是两步，
        串台就藏在这两步之间的时间缝里。上限 4 时 2 个线程交错执行：

        t1: self.cursor.execute("SELECT 1")   # self.cursor = 1号结果集
        t2: self.cursor.execute("SELECT 2")   # self.cursor = 2号结果集，1号被顶掉
        t1: self.cursor.fetchall()            # → [("2",)]  t1 拿到 t2 的数据
        t2: self.cursor.fetchall()            # → []        t2 再取已空

        全程没有一行报错。闸门账本上只记"进来了几个人"（2 ≤ 4，合法），
        对屋里谁在写 self 一无所知——所以它认为一切正常。
    """

    def __init__(self, max_concurrent: Optional[int] = None):
        s = get_settings()
        # 默认 4 来自 config.TOOL_MAX_CONCURRENT；max(1,...) 保底：传 0 会变成全程串行
        self._default_limit = max(1, int(max_concurrent if max_concurrent is not None
                                          else getattr(s, "tool_max_concurrent", 4)))
        self._sems: Dict[str, threading.Semaphore] = {}    # 工具名 → 信号量
        self._in_flight: Dict[str, int] = {}               # 工具名 → 正在执行中的请求数（观测用）
        self._max_seen: Dict[str, int] = {}                # 工具名 → 历史同时在执行峰值
        self._lock = threading.Lock()                      # 保护上面三张账本的锁

    # 【什么】取（必要时新建）某工具的信号量
    # 【作用】懒创建：没被并发调过的工具不占资源；上限沿用创建时的默认值
    def _semaphore(self, name: str) -> threading.Semaphore:
        with self._lock:                       # 拿锁查表：防止两个线程同时给同一工具造第二把闸门
            if name not in self._sems:
                self._sems[name] = threading.Semaphore(self._default_limit)   # 首次用到才造（懒创建）；Semaphore = 计数器闸门，初始值即上限：acquire 减 1（减不动排队），release 加 1（放行一个排队者）
            return self._sems[name]

    # 【什么】带闸门执行一次工具调用
    # 【作用】acquire → 记账 → 执行 → release，函数本身不吞异常（异常也保证 release）
    def execute(self, name: str, fn: Callable[[], Any]) -> Any:
        sem = self._semaphore(name)            # 拿这把工具的闸门（第一次用则现造）
        acquired = sem.acquire(timeout=30)     # 最多等 30 秒；排队超时说明下游已严重堆积
        if not acquired:
            raise TimeoutError(f"工具 {name} 并发排队超时(30s)，当前在途 {self.in_flight(name)}")
        try:
            with self._lock:                   # 在途数 +1（拿锁记一笔账，纯观测用，不拦人）
                cur = self._in_flight.get(name, 0) + 1
                self._in_flight[name] = cur
                self._max_seen[name] = max(self._max_seen.get(name, 0), cur)
            return fn()                        # 真正执行工具调用（成功或抛异常都算"这一趟结束"）
        finally:
            with self._lock:                   # 在途数 -1，无论 fn 成功/失败都执行（账实相符）
                self._in_flight[name] = self._in_flight.get(name, 1) - 1
            sem.release()                      # 还名额：排在后面的线程立刻顶上

    # 【什么】查某工具当前在途/历史峰值
    # 【作用】观测与压测用：峰值长期等于上限 = 排队常发生，该加实例或加名额了
    def in_flight(self, name: str) -> int:
        with self._lock:
            return self._in_flight.get(name, 0)

    def max_seen(self, name: str) -> int:
        with self._lock:
            return self._max_seen.get(name, 0)


# 【什么】有状态工具的实例池（借出-归还）
# 【作用】每个线程借走一个专属实例，用完归还；池空时等待，避免多线程共用一个实例串状态
class ToolInstancePool:
    """工具实例池：factory 造实例，checkout/checkin 借还，池大小固定。

    【一句话看破】核心数据结构就是 self._idle 一个数组（list），没有更多了。
        所谓"池" = 数组 + 五条借还纪律：
        ① 借走用 pop() 移除（不删的话两人借到同一个，串台）
        ② 还回用 append() 追加
        ③ _created 计数器限制最多 new size 个（不限制就退化成每次现造）
        ④ 数组空且造满时去睡觉等（Condition.wait，不烧 CPU）
        ⑤ 等超 timeout 秒抛 TimeoutError（不设超时线程会挂死）
        把这五条纪律逐条拆开讲"不这么写会怎样"，就是面试的完整答案。

    【为什么有状态工具必须用池】以"共享一个游标"的工具为例：
        线程 A execute("SELECT 1") → 结果集 A 还没 fetch
        线程 B execute("SELECT 2") → 结果集 A 被顶掉
        线程 A fetchall()          → 拿到的是 B 的数据，且不报错（静默串台）
        池化后 A、B 各借各的实例，这个问题从根上消失。

    【输入输出对照】pool = ToolInstancePool(factory=make_conn_tool, size=2)：
        t1.checkout() → 实例 #1（池剩 1）
        t2.checkout() → 实例 #2（池剩 0）
        t3.checkout() → 阻塞等待（timeout=10s 内有人 checkin 才拿得到）
        t1.checkin(#1) → t3 立刻拿到实例 #1
    """

    def __init__(self, factory: Callable[[], Any], size: int = 4, name: str = "pool"):
        # ------------------------------------------------------------------
        # 【什么】factory 是"造实例的配方"，name 是池子的名字
        #
        # 【谁传值】本项目装配代码（main.py / tool_registry 组装工具处）。
        #   注意：本类当前尚未接线，暂无真实调用方；接线时传的是
        #   "无参数、返回新实例"的函数，例如：
        #       pool = ToolInstancePool(factory=lambda: DataQueryTool(), size=2)
        #
        # 【传的值是干什么的】池子需要 size 个"各自独立"的实例才能治串台，
        #   所以装配处不能只传 1 个现成对象（借给谁都是同一个，串台照旧），
        #   而是传"怎么造"的手法。池子平时只存配方不执行（启动零成本），
        #   checkout 发现架子空且没造满（_created < _size）才调
        #   self._factory() 按配方现造一个。
        #
        # 【name 的用途】只出现在超时报错信息里：
        #   f"工具实例池 {self._name} 借出超时(...)"，让人一看日志就知道
        #   是哪个池堵了。
        #
        # 【输入输出对照】
        #   pool = ToolInstancePool(factory=lambda: DataQueryTool(), size=2)
        #   此刻 pool._idle = []（一个实例都没造，只存了配方）
        #   pool.checkout() → 内部执行 factory() = DataQueryTool() → 现造 #1
        #   pool.checkout() → 现造 #2（_created=2，造满）
        #   pool.checkout() → 不再调 factory，等归还，10s 没等到抛 TimeoutError
        # ------------------------------------------------------------------
        self._factory = factory
        self._name = name
        self._size = max(1, int(size))          # 保底 1：传 0 池子永远空，永远借不到
        self._idle: list = []                   # 空闲实例栈（借出的不在池里）
        self._created = 0                       # 已创建总数（≤ size）
        self._lock = threading.Lock()         # 借/还实例时锁住工具架（_idle/_created），防止两人同时抓同一个实例
        # ------------------------------------------------------------------
        # 【什么】_lock 是钥匙，_available 是绑在这扇门上的对讲机
        #   把池子想成工具间：架上摆空闲实例（_idle），借/还都要先拿钥匙
        #   （Lock），防止两人同时抓同一个实例。
        #
        # 【Condition 解决的问题】普通锁只有"锁门/开门"两个动作。池空时借
        #   不到实例的线程总不能占着钥匙傻等——不然还实例的人进不了门，
        #   就死锁了。Condition 在锁上追加两个动作：
        #     wait()      = 交还钥匙 + 去长椅睡觉（被喊醒后自动重新拿钥匙）
        #     notify_all() = 按对讲机喊"有实例归还了"，喊醒所有睡觉的线程
        #
        # 【为什么传 self._lock】对讲机必须绑在管账本的那把钥匙上。
        #   若另造一把锁（Condition() 空参数自造锁）：还实例的人"放回架子"
        #   和"喊人"不在同一扇门内完成，可能架子还没放稳就喊人——睡觉的
        #   线程醒来看到空架子，白醒一趟甚至漏拿。绑同一把锁 = 放架子和
        #   喊人两个动作锁内连续完成，不会被打断。
        #
        # 【输入输出对照】size=2 全部借出，t3 想借：
        #   t3: 拿钥匙 → 架子空 → wait()：还钥匙+睡觉
        #   t1: checkin：拿钥匙 → 实例放回架子 → notify_all() 喊人 → 还钥匙
        #   t3: 被喊醒 → 自动重新拿钥匙 → 回循环顶 → 从架子拿走实例 #1
        # ------------------------------------------------------------------
        self._available = threading.Condition(self._lock)   # 池空时在这里等归还信号

    # 【什么】从池里借一个实例（阻塞等待，最多 timeout 秒）
    # 【作用】优先复用空闲实例；没有空闲且没造满就现造一个；造满了就等别人归还
    def checkout(self, timeout: float = 10.0) -> Any:
        with self._available:
            deadline = time.time() + timeout
            while True:
                if self._idle: 
                    return self._idle.pop()                      # ① 有空闲：直接复用，零创建成本
                                                                 #   pop() 从栈顶取走并移除一个实例：
                                                                 #   [实例#2, 实例#1].pop() → 返回#2，栈剩[#1] 就是一个数组
                                                                 #   取走的不再留在 _idle 里（借出的实例不在架上，checkin 才回来）
                if self._created < self._size:
                    self._created += 1
                    instance = self._factory()                   # ② 没空闲但没造满：现造一个
                                                                 #   等价于 new：若配方是 lambda: DataQueryTool()，
                                                                 #   这行就是 DataQueryTool()。不写死类名是因为池子
                                                                 #   是通用件，手法由装配处递配方进来。
                                                                 #   新实例不进 _idle 架子，造完直接借给当前线程。
                    self._available.notify_all()                 #   喊醒睡觉的线程重查条件（见下方 wait 分支）
                    return instance
                remaining = deadline - time.time()
                if remaining <= 0 or not self._available.wait(remaining):
                    raise TimeoutError(
                        f"工具实例池 {self._name} 借出超时({timeout}s)："
                        f"上限 {self._size} 个实例全部在用"
                    )
                # 被唤醒后回到循环顶部再抢一次空闲实例

    # 【什么】归还实例
    # 【作用】放回空闲栈并唤醒等待者；不归还会让池永久缩容（借一个少一个）
    def checkin(self, instance: Any) -> None:
        with self._available:
            self._idle.append(instance)
            self._available.notify_all()

    # 【什么】上下文管理器形式借还（推荐用法）
    # 【作用】with 结束自动归还，即使块内抛异常也不泄漏实例
    def borrow(self, timeout: float = 10.0):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            instance = self.checkout(timeout=timeout)
            try:
                yield instance
            finally:
                self.checkin(instance)

        return _ctx()

    def stats(self) -> Dict[str, int]:
        """池状态：{'size': 上限, 'created': 已创建, 'idle': 当前空闲}。"""
        with self._available:
            return {"size": self._size, "created": self._created, "idle": len(self._idle)}
