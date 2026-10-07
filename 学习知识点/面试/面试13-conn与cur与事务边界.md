# 面试13：conn 与 cur 各管什么？哪一步才是真正的"保存"？

> 配套代码：`modules/memory/pg_manager.py:129`（autocommit=False）、`pg_manager.py:131`（全类共用一个游标）、`pg_manager.py:416`（_reconnect_if_needed 心跳探活）、`pg_manager.py:647`（save_history 写后 commit）、`pg_manager.py:789-807`（游标与 fetch 语义的大段实战注释，在 search_memory_by_vector 里）
> 一句话结论：**conn 是"一根到数据库的电话线"（事务开在它身上），cur 是"握在手里的记事本"（结果集暂存在它那里）；autocommit=False 时 execute 只是"报了个数"，`conn.commit()` 才是按保存键——忘了 commit，数据"看似写入实则丢失"且不报任何错。**

---

## 〇、先把术语说人话

| 术语 | 大白话解释（生活比喻） |
|---|---|
| 连接（conn，Connection） | 你打给银行客服的**那通电话**。电话一接通就占着一条线路，所有对话都从这条线上走；电话挂断，线上说过但没"确认入账"的事全部作废 |
| 游标（cur，Cursor） | 通话时你手里拿的**记事本**。客服每答复一句（数据库返回一批结果），先记在本子上，你再从本子上抄走（fetch） |
| 事务（Transaction） | "这次通话里说的几件事，**要么全部生效，要么全部不算数**"。像转账：扣你的钱和加对方的钱必须一起成，不能只成一半 |
| autocommit=False | "先别急着入账"模式。每件事先记着，等你说"确认"（commit）才一起入账；说"算了"（rollback）就一起撕掉 |
| commit() | 按**保存键/确认入账**：事务里攒的所有改动一次性写进数据库，别的连接立刻能查到 |
| rollback() | **撕掉本页**：事务里攒的改动全部作废，回到事务开始前的样子 |
| RETURNING id | INSERT 的同时让数据库**顺手把新行的主键递回来**，省得再查一次"刚才那行的 id 是几" |

---

## 一、先说结论

| 面试问 | 一句话答 |
|---|---|
| conn 管什么？ | 连接 + 事务边界：commit / rollback 都调它的方法 |
| cur 管什么？ | 语句执行 + 结果集暂存：execute 发 SQL，fetch 系列搬结果 |
| 谁才是"保存"？ | `self._conn.commit()`（pg_manager.py:647）——execute 之后 commit 之前，数据只在事务里，别人看不见，断连即回滚 |
| 游标能复用吗？ | 本项目全类共用一个 `self._cur`（单线程模型，pg_manager.py:131）；execute 下一条 SQL 时旧结果集被**静默丢弃** |
| fetchall 能取两次吗？ | 不能。官方语义是"返回**剩余**行"：取过的不再给，第二次 fetchall 得 `[]` |

---

## 二、大白话拆解（全部摘自真实代码）

```python
# pg_manager.py:122-131（连接建立时的两个关键设置）
self._conn = psycopg2.connect(host=..., ...)   # 电话线接通
self._conn.autocommit = False                  # 关自动提交：每条写操作后由代码显式 commit
self._cur = self._conn.cursor()                # 从电话线上拿一个记事本（全类共用这一个）

# pg_manager.py:624-647（写路径的完整生命周期，以 save_history 为例）
self._cur.execute(
    """INSERT INTO chat_history (...) VALUES (...) RETURNING id""",
    (session_id, user_id, ...),
)
#   ↑ 只是把 SQL 发出去并执行，结果集挂到 _cur 上；数据在【未提交事务】里
row = self._cur.fetchone()                     # pg_manager.py:636，从结果集搬回第一行（拿新主键）
self._cur.execute(                             # pg_manager.py:643，同事务再刷会话封面时间
    "UPDATE chat_session SET updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
    (session_id,),
)
self._conn.commit()                            # pg_manager.py:647 ← 真正的"按保存键"，这一刻才落盘可见

# pg_manager.py:695-703（读方法为什么不 commit，以 get_short_term_history 为例）
self._reconnect_if_needed()
self._cur.execute("SELECT ... ORDER BY created_at DESC LIMIT %s", (...))
rows = self._cur.fetchall()
#   execute → fetchall → return，全程没有 commit —— SELECT 不改数据，无需提交
```

| 部分 | 大白话解释 |
|---|---|
| `conn` | 电话线：所有 SQL 都从这根线上走；事务（"这次通话里说的都算数才算数"）挂在线上 |
| `cur` | 记事本：execute 后数据库的应答记在这本上，fetch 是翻开本子抄下来 |
| `autocommit=False` | 每句话说完先别入账，攒着；等我说 commit 才一起入账——要么全成要么全不成 |
| `commit()` | 按保存键：事务里攒的所有改动一次性落盘，对其他连接可见 |
| `_reconnect_if_needed`（pg_manager.py:416-423） | 每个方法开头 `SELECT 1` 心跳探活：PG 掐断空闲连接后第一次操作不报错，先探再补 |

---

## 三、写路径流程图（以 save_history 为例）

```
代码动作                              数据此刻在哪？           别的连接能看到吗？
─────────────────────────────────────────────────────────────────────
self._cur.execute(INSERT ... )   →   未提交事务缓冲区          ✗ 看不到
row = self._cur.fetchone()       →   主键 (42,) 回到 Python     ✗ 看不到
self._cur.execute(UPDATE ...)    →   同一事务里又攒一笔         ✗ 看不到
self._conn.commit()              →   两笔一起落盘              ✓ 能看到了
───────────── 如果 commit 之前进程崩溃 / 连接被掐 ─────────────
（什么都没发生）                   →   PG 自动 rollback         这行数据彻底消失，且无报错
```

读路径则短得多：

```
_reconnect_if_needed()   →   execute(SELECT)   →   fetchall()   →   return
     （探活）                 （结果集挂上 _cur）    （抄回 Python）    （无 commit）
```

---

## 四、输入 → 处理 → 输出对照

```python
# 场景1：写了但没 commit（autocommit=False 的经典事故）
cur.execute("INSERT INTO chat_history (...) VALUES (...)")   # rowcount=1，"成功"
# 同一时刻另开一个连接查这张表 → 查不到这行（未提交，对别人不可见）
# 进程此刻崩溃 / 连接被掐 → 这行数据彻底消失，且没有任何报错日志

# 场景2：写了并 commit（pg_manager.py:624-647 的正确姿势）
cur.execute("INSERT ... RETURNING id", (...))
row = cur.fetchone()          # 拿到 (42,)
conn.commit()                 # 落盘；另一个连接现在能查到 id=42

# 场景3：游标结果集被顶掉（共用 _cur 的边界）
cur.execute("SELECT ...")     # 结果集 A 挂上
cur.execute("SELECT ...")     # 结果集 B 顶掉 A —— A 静默丢弃，不报错
cur.fetchall()                # 拿到的是 B 的行；A 的数据再也拿不回来

# 场景4：fetchall 只能取一次
cur.execute("SELECT ... LIMIT 3")
rows1 = cur.fetchall()        # [r1, r2, r3]   ← "返回剩余全部行"
rows2 = cur.fetchall()        # []              ← 剩余为空，二次取必空
```

一个带真实数据的端到端例子（save_history 保存一轮"3*7 等于多少"）：

| 阶段 | 内容 |
|---|---|
| 输入 | `session_id="abc123"`, `user_query="3*7 等于多少"`, `llm_full_answer="3*7 等于 21..."`, `summary_embedding=[0.12, -0.34, ...]`（1024 维） |
| 处理 | 向量 list → `np.array(dtype=float32)`（pgvector 只认 numpy）；`tool_call_log`/`rag_sources` → JSON 字符串存 JSONB 列；INSERT + UPDATE 攒进同一事务 |
| 输出 | `commit()` 后返回新主键 `42`；此刻另一个连接 `SELECT` 能查到 id=42 这行，会话列表的 updated_at 也同步刷新 |

---

## 五、为什么要这么写

| 做法 | 原因（不这么写会怎样） |
|---|---|
| autocommit=False + 显式 commit | 开 autocommit：多语句操作（先 INSERT 再 UPDATE，见 pg_manager.py:624-643）中途失败时前一半已落盘，出现半截数据；关掉后一个事务整体成功或整体回滚 |
| 读方法不 commit | SELECT 无改动可提交；多调一次 commit 只是白往返一次 |
| 每方法开头 `_reconnect_if_needed` 探活（pg_manager.py:416-423） | PG 默认掐空闲连接：不探活，隔夜后第一个请求直接 OperationalError，用户看到的就是"每天早上第一次必失败" |
| execute 后立刻 fetch（RETURNING id，pg_manager.py:636） | 不立刻取就执行下一条 SQL：结果集被顶掉（场景3），`rowcount` 只有行数没有主键 |
| DDL（建扩展/建表）单独 commit（pg_manager.py:170） | `_connect` 里装扩展与建表拆成独立事务：一条 DDL 失败只回滚自己，不连累已执行的部分；且未结束的事务持锁，会堵住第二个启动的进程 |
| 全类共用一个游标（仅限单线程，pg_manager.py:131） | 多线程共用：方法 B 的 execute 顶掉方法 A 的结果集（场景3），A 的 fetchall 拿到 B 的数据且不报错——所以本项目自注释"单线程模型"，多线程场景要每线程各开游标 |

---

## 六、追问清单（提前准备好答案）

**Q1：为什么不每 execute 一条就 commit 一次？**
答：save_history 里是"INSERT 历史 + UPDATE 会话封面"两笔（pg_manager.py:624-643），它们要么一起成要么一起不成——若各 commit 一次，中途崩溃会留下"历史存了但封面时间没刷"的半截状态。一个事务包住，崩溃则 PG 自动 rollback，两笔都不留。另外每条 commit 多一次网络往返，写密集时白白拖慢。

**Q2：rollback 在哪调？你们代码里怎么没见到？**
答：正常路径不手写 rollback——`@handle_exception` 捕获异常后上抛，连接交还时未提交事务由 psycopg2/PG 自动回滚。rollback 的语义价值在"同一连接还要继续用时，先把烂摊子撕掉"：本项目每方法开头 `_reconnect_if_needed` 探活（pg_manager.py:419），探活失败直接整条重连（`_connect()`，pg_manager.py:423），比重用脏连接更彻底，所以不需要显式 rollback。

**Q3：为什么不用连接池？**
答：本项目是单进程单线程模型（pg_manager.py:131 注释"单线程模型"），一根连接加一个游标足够，连接池解决的是"并发请求抢连接"的问题，没有并发就没有池化收益。真要上多线程，第一步就是每线程各开游标——因为共用游标会出现场景3的结果集互顶。

**Q4：RETURNING id 是什么？为什么不 INSERT 完再 SELECT 查一次？**
答：PostgreSQL 的扩展语法，INSERT 执行完顺手把新行的指定列（这里是自增主键 id）塞进结果集返回（pg_manager.py:626 的 SQL 尾部）。替代方案是再发一条 `SELECT lastval()` 之类的查询，多一次往返还要处理并发下"查到的未必是我插的那行"的问题。注意取了必须立刻 fetch（pg_manager.py:636），否则下一条 execute 把结果集顶掉，主键就丢了。

**Q5：fetchone / fetchmany / fetchall 取的是三份数据吗？**
答：不是，是同一份结果集上的三种"抄法"（pg_manager.py:798 注释明说）：fetchone 抄一行、fetchmany(n) 抄 n 行、fetchall 抄**剩余全部**。混着用可以，比如先 fetchone 看第一行再 fetchall 拿剩下；但任何一种取过的行都不会再出现，所以重复 fetchall 得 `[]`。

**Q6：INSERT 之后不 commit 直接 fetchall 会怎样？**
答：分两种：普通 INSERT 没有结果集，fetchall 直接抛 `ProgrammingError`（不是返回 `[]`，pg_manager.py:795 注释专门标了这个坑）；带 RETURNING 的 INSERT 有结果集，必须先 fetch 再 commit，顺序反了主键拿不到。

---

## 七、简单总结

1. 一句话记忆：**conn 是线（事务在线上），cur 是本（结果在本上），commit 是按保存键**——`execute` 只是报数，`self._conn.commit()`（pg_manager.py:647）才落盘
2. 三个必背边界：autocommit=False 时**不 commit = 静默丢数据**；`fetchall` 二次调用必得 `[]`（"剩余行"语义）；`execute` 下一条会**静默顶掉**上一条的结果集
3. 心跳探活 `_reconnect_if_needed`（pg_manager.py:416-423）治的是"空闲连接被 PG 掐断后第一次操作必报错"
