# 05 `_conn` 与 `_cur` 分不清？一张表搞定

> 对应代码：`modules/memory/pg_manager.py:107-147`

## 一、一句话区分

| | 是什么 | 管什么 |
|---|---|---|
| **`_conn`** | 到 PostgreSQL 的**一条连接**（TCP） | **事务**：`commit()` / `rollback()` |
| **`_cur`** | 连接上的**一个游标**（操作手柄） | **执行与取结果**：`execute()` / `fetchone()` |

**口诀：`conn` 管提交，`cur` 管干活。**

## 二、类比：银行柜台

```
_conn  = 你在这个柜台办理业务的「整个过程」
         所有事办完后统一签字确认（commit），或整单作废（rollback）

_cur   = 你递进去的「每一张单据」和拿回来的「回执」
         递单（execute）→ 收回执（fetchone）
```

关键点：**签字是对「整个过程」签的，不是对某一张单子签的**——所以 `commit()` 挂在 `_conn` 上，不挂在 `_cur` 上。

另一个类比：`_conn` 是一次电话通话，`_cur` 是通话里的每一句问答。

## 三、为什么非要分两个？

因为**一个事务里要跑多条 SQL**，但结果是逐条取回的：

```python
self._cur.execute("INSERT ... RETURNING id")   # 第1条 SQL
row = self._cur.fetchone()                     # 取第1条的结果
self._cur.execute("UPDATE ...")                # 第2条 SQL
self._conn.commit()                            # 两条一起生效 ← 作用在连接上
```

如果只有一个对象，"提交"就不知道该算到哪条 SQL 头上。拆开后职责清晰：
- **连接**决定"这堆操作算不算数"（事务边界）
- **游标**决定"每条 SQL 怎么执行、结果怎么拿"

## 四、常用方法对照

| `_conn`（连接） | 作用 |
|---|---|
| `.commit()` | 提交事务，改动真正落盘 |
| `.rollback()` | 回滚事务，全撤销 |
| `.close()` | 关闭连接 |
| `.cursor()` | **造一个游标**（`_cur` 就是这么来的） |
| `.autocommit` | 是否自动提交（本项目设 `False`） |

| `_cur`（游标） | 作用 |
|---|---|
| `.execute(sql, params)` | 发 SQL 给数据库执行 |
| `.fetchone()` | 取 1 行（没有则返回 `None`） |
| `.fetchall()` | 取全部行（没有则返回 `[]`） |
| `.rowcount` | 受影响的行数 |
| `.close()` | 关闭游标 |

⚠ 最容易记混的两点：
1. **`commit()` 是 `conn` 的**，不是 `cur` 的
2. **`fetchone()` 是 `cur` 的**，不是 `conn` 的

## 五、本项目里的生命周期

```107:113:modules/memory/pg_manager.py
    def _connect(self):
        """连接 PG 并注册 pgvector 扩展。"""
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            s = get_settings()
            self._conn = psycopg2.connect(
```

```147:147:modules/memory/pg_manager.py
            self._cur = self._conn.cursor()
```

| 阶段 | 发生了什么 |
|---|---|
| 首次 `PGManager()` | `__init__` → `_connect()` → **同时创建 `_conn` 和 `_cur`** |
| 每次调用方法 | 先 `_reconnect_if_needed()` 探活，两者一起检查 |
| 连接断了 | `_connected = False` → 重跑 `_connect()` → **两者一起重建**（旧游标随之失效） |
| 全项目 | **只有一个 `conn`、一个 `cur`**，全局复用（`_instance` 单例） |

> 注意 `_cur` 是从 `_conn` 里"长"出来的（`self._conn.cursor()`）。**连接没了，游标必然失效**，所以重连必须两个都重建——这正是 `_reconnect_if_needed` 里直接调 `_connect()` 整个重来的原因，而不是只重连 `_conn`。

## 六、一次完整调用里它们各自干了啥

以 `save_turn` 为例（`pg_manager.py:523-543`）：

```
_reconnect_if_needed()          → 用 _cur 执行 SELECT 1 探活
_cur.execute(INSERT ... RETURNING id)   ← 游标：发 SQL
_cur.fetchone()                         ← 游标：取回 id
_conn.commit()                          ← 连接：提交事务
```

`create_session` / `update_session_state` / `get_session_state` 全是这个套路：
**`_cur` 干活 → `_conn` 收尾。**

只有纯读操作（如 `get_short_term_history`）不需要 `commit()`——没改数据，提交了也没意义（不过本项目部分读方法也调了 commit，无害）。

## 七、几个细节

**Q：可以有多个 cur 吗？**
可以。`conn.cursor()` 想开几个开几个，各自独立执行、互不影响，但**共享同一个事务**——任何一个 `commit()` 都会把所有游标的改动一起提交。本项目只开一个就够（单线程顺序执行）。

**Q：`_conn` / `_cur` 的下划线啥意思？**
Python 约定：`_` 前缀表示"内部私有，外部别直接碰"。不强制，纯粹是提示——你要用 `pg._cur` 语法上拦不住。

**Q：为什么 `autocommit = False`？（142 行）**
默认 `True` 时每条 SQL 立刻落盘、无法回滚。本项目一次落库要写多个字段，任何一步失败都要整体撤销，否则会出现"有原文没向量"的脏数据。**事务状态属于连接**，所以这个开关在 `_conn` 上。

## 八、速查表

```python
conn = psycopg2.connect(...)   # 建连接
cur  = conn.cursor()           # 从连接里开游标（cur 依赖 conn）

cur.execute(sql, params)       # 干活用 cur
cur.fetchone() / fetchall()    # 取结果用 cur
conn.commit() / rollback()     # 提交/回滚用 conn ← 别写反

conn.close()                   # 最后关连接（cur 随之失效）
```

## 九、一句话带走

> **连接是"会话"，游标是"会话里的每次操作"。**
> 干活找 `_cur`，收尾（commit）找 `_conn`；重连时两个一起重建。
