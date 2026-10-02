# 面试13：conn 与 cur 各管什么？哪一步才是真正的"保存"？

> 配套代码：`modules/memory/pg_manager.py:129`（autocommit=False）、`pg_manager.py:416`（_reconnect_if_needed）、`pg_manager.py:574`（写后 commit）、`pg_manager.py:705-724`（游标与 fetch）
> 一句话结论：**conn 是"一根到数据库的电话线"（事务开在它身上），cur 是"握在手里的记事本"（结果集暂存在它那里）；autocommit=False 时 execute 只是"报了个数"，`conn.commit()` 才是按保存键——忘了 commit，数据"看似写入实则丢失"且不报任何错。**

---

## 一、先说结论

| 面试问 | 一句话答 |
|---|---|
| conn 管什么？ | 连接 + 事务边界：commit / rollback 都调它的方法 |
| cur 管什么？ | 语句执行 + 结果集游标：execute 发 SQL，fetch 系列搬结果 |
| 谁才是"保存"？ | `self._conn.commit()`（pg_manager.py:574）——execute 之后 commit 之前，数据只在事务里，别人看不见，断连即回滚 |
| 游标能复用吗？ | 本项目全类共用一个 `self._cur`（单线程模型）；execute 下一条 SQL 时旧结果集被**静默丢弃** |
| fetchall 能取两次吗？ | 不能。官方语义是"返回**剩余**行"：取过的不再给，第二次 fetchall 得 `[]` |

---

## 二、大白话拆解

```python
# pg_manager.py:122-131（连接建立时的两个关键设置）
self._conn = psycopg2.connect(host=..., ...)   # 电话线接通
self._conn.autocommit = False                  # 关自动提交：每条写操作后由代码显式 commit
self._cur = self._conn.cursor()                # 从电话线上拿一个记事本（全类共用这一个）

# pg_manager.py:561-574（写路径的完整生命周期，以 save_history 为例）
self._cur.execute("INSERT INTO chat_history ... RETURNING id", (...))
#   ↑ 只是把 SQL 发出去并执行，结果集挂到 _cur 上；数据在【未提交事务】里
row = self._cur.fetchone()                     # 从结果集搬回第一行（拿新主键）
self._conn.commit()                            # ← 真正的"按保存键"，这一刻数据才落盘可见

# pg_manager.py:416（读方法为什么不 commit）
# get_short_term_history / search_memory_by_vector:
#   execute → fetchall → return      （SELECT 不改数据，无需 commit）
```

| 部分 | 大白话解释 |
|---|---|
| `conn` | 电话线：所有 SQL 都从这根线上走；事务（"这次通话里说的都算数才算数"）挂在线上 |
| `cur` | 记事本：execute 后数据库的应答记在这本上，fetch 是翻开本子抄下来 |
| `autocommit=False` | 每句话说完先别存档，攒着；等我说 commit 才一起存——要么全成要么全不成 |
| `commit()` | 按保存键：事务里攒的所有改动一次性落盘，对其他连接可见 |
| `_reconnect_if_needed` | 每个方法开头 `SELECT 1` 心跳探活：PG 掐断空闲连接后第一次操作不报错，先探再补 |

---

## 三、输入 → 输出对照

```python
# 场景1：写了但没 commit（autocommit=False 的经典事故）
cur.execute("INSERT INTO chat_history (...) VALUES (...)")   # rowcount=1，"成功"
# 同一时刻另开一个连接查这张表 → 查不到这行（未提交，对别人不可见）
# 进程此刻崩溃 / 连接被掐 → 这行数据彻底消失，且没有任何报错日志

# 场景2：写了并 commit（pg_manager.py 的正确姿势）
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

---

## 四、为什么要这么写

| 做法 | 原因（不这么写会怎样） |
|---|---|
| autocommit=False + 显式 commit | 开 autocommit：多语句操作（先 INSERT 再 UPDATE）中途失败时前一半已落盘，出现半截数据；关掉后一个事务整体成功或整体回滚 |
| 读方法不 commit | SELECT 无改动可提交；多调一次 commit 只是白往返一次 |
| 每方法开头 `_reconnect_if_needed` 探活 | PG 默认掐空闲连接：不探活，隔夜后第一个请求直接 OperationalError，用户看到的就是"每天早上第一次必失败" |
| execute 后立刻 fetch（RETURNING id） | 不立刻取就执行下一条 SQL：结果集被顶掉，`rowcount` 只有行数没有主键 |
| DDL（建表/加列）单独 commit | `_connect` 里装扩展与建表拆成独立事务：一条 DDL 失败只回滚自己，不连累已执行的部分 |
| 全类共用一个游标（仅限单线程） | 多线程共用：方法 B 的 execute 顶掉方法 A 的结果集（场景3），A 的 fetchall 拿到 B 的数据且不报错——所以本项目自注释"单线程模型"，多线程场景要每线程各开游标 |

---

## 五、简单总结

1. 一句话记忆：**conn 是线（事务在线上），cur 是本（结果在本上），commit 是按保存键**——`execute` 只是报数，`self._conn.commit()`（pg_manager.py:574）才落盘
2. 三个必背边界：autocommit=False 时**不 commit = 静默丢数据**；`fetchall` 二次调用必得 `[]`（"剩余行"语义）；`execute` 下一条会**静默顶掉**上一条的结果集
3. 心跳探活 `_reconnect_if_needed`（pg_manager.py:416）治的是"空闲连接被 PG 掐断后第一次操作必报错"
