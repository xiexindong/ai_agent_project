# 知识点 04：为什么 `execute` 之后还要 `fetchone`

> 对应代码：`modules/memory/pg_manager.py:454-459`

## 一、被问到的代码

```454:459:modules/memory/pg_manager.py
        self._cur.execute(
            "SELECT agent_state FROM chat_session WHERE session_id = %s AND user_id = %s",
            (session_id, user_id),
        )
        row = self._cur.fetchone()
        return row[0] if row and row[0] else {}
```

## 二、一句话

**`execute()` 是"下单"，`fetchone()` 才是"取餐"。**

```python
cur.execute(...)    # 把 SQL 发给数据库，数据库算完了，结果还留在数据库那边
row = cur.fetchone()  # 把结果从数据库搬回 Python，变成内存里的一个元组
```

少了 `fetchone()`，你什么也拿不到——`row` 根本不存在。

## 三、为什么要分两步？（不能一步到位？）

设计成两步是有道理的：

| 好处 | 说明 |
|---|---|
| **省内存** | 查到 100 万行时，可以先只取 1 行看看，不必把 100 万行全搬进 Python 内存 |
| **可控制** | 想取 1 行 / 10 行 / 全部，由你决定（`fetchone` / `fetchmany(10)` / `fetchall()`） |
| **流式处理** | 可以循环 `while row := cur.fetchone():` 一行行处理大结果集 |

类比：你在电商下了单，**商品不会瞬间出现在你手上**，得等快递（fetch）送过来。

## 四、`self._cur` 是什么？游标（Cursor）

```
Python              ←── 网络 ──→     PostgreSQL
  conn（连接）                        数据库
   └─ cur（游标）                       └─ 结果集（一堆行）
        ↑
   指向"当前读到第几行"的指针
```

- **连接 `conn`**：一条电话线
- **游标 `cur`**：电话线上的一次会话，用来发 SQL、接结果

一个连接可以开多个游标，但通常一个就够。

## 五、`fetch` 三兄弟怎么选

| 方法 | 取多少 | 没数据时返回 | 本项目用在哪 |
|---|---|---|---|
| `fetchone()` | **1 行** | `None` | 查单行（`get_session_state`、`get_history`） |
| `fetchmany(n)` | n 行 | `[]` | 分页取 |
| `fetchall()` | **全部** | `[]` | 查列表（`search_memory_by_vector`、`get_short_term_history`） |

**这里为什么用 `fetchone`？** 因为 `session_id` 是主键，最多只可能有 1 行，取 1 行就够了。用 `fetchall` 也不会错，但拿到的是 `[(state,)]` 这种包了一层的列表，还得 `rows[0][0]`，多此一举。

## 六、`row` 长什么样

数据库返回的**每一行都是元组**，列顺序对应 SELECT 的顺序：

```python
# SELECT agent_state FROM ... 只查了 1 列
row = ({'iteration': 2, 'user_query': '3*7'},)
#       ↑ 第0列                              ↑ 注意逗号，这是元组
```

所以 `row[0]` 就是取第 0 列 → `agent_state` 的值。

如果查多列：`SELECT agent_state, updated_at FROM ...`
```python
row = ({...}, datetime(2026, 9, 15, 10, 30))
row[0]  # agent_state
row[1]  # updated_at
```

本项目有个更可读的写法别处也在用：**字典游标**（`RealDictCursor`），那时 `row["agent_state"]` 就能按列名取，不用记下标。

## 七、第 459 行那个三元表达式

```python
return row[0] if row and row[0] else {}
```

拆开等价写法：

```python
if row is None:        # 没查到这行
    return {}
if not row[0]:         # 查到了，但 agent_state 是空的（{} 或 NULL）
    return {}
return row[0]
```

为什么要判两次？两种"空"都算没查到：

| 情况 | `row` | 含义 |
|---|---|---|
| 没有这个会话 / 不属于这个 user | `None` | 压根没这条记录 |
| 记录存在但状态是空的 | `({},)` | agent_state 是 `{}`（**空字典是假值**） |

统一返回 `{}`，调用方就不用判空了——这正是 `server.py:225` 能直接写 `if not pg.get_session_state(...)` 的原因。

⚠️ 注意 `fetchone()` 查不到时**返回 `None`，不抛异常**。所以必须判空，否则 `row[0]` 直接 `TypeError: 'NoneType' object is not subscriptable`。

## 八、⚠️ 一个连带的知识点：取出来是 dict，不是字符串

存的时候（知识点 02）要手动 `json.dumps`：

```python
json.dumps({"iteration": 2})   # dict → '{"iteration": 2}'
```

但**取出来的时候不用 `json.loads`**：

```python
row[0]  # 直接就是 {'iteration': 2}  ← dict！
```

为什么？PostgreSQL 的 `agent_state` 列是 **JSONB 类型**，psycopg2 看到列类型是 json/jsonb，会**自动帮你 parse 成 dict/list**。

| 方向 | 谁负责转换 |
|---|---|
| 存（Python → PG） | **你**：`json.dumps()`，因为 psycopg2 不认识 dict |
| 取（PG → Python） | **驱动**：自动 parse，因为 PG 告诉它这是 jsonb |

> 这就是"类型信息"的价值——数据库知道这一列是 JSONB，驱动就能自动处理。如果这列是 `TEXT`，取回来就真是一个字符串，你得自己 `json.loads`。

## 九、两个常见坑

**坑 1：不 fetch 就复用游标**

```python
cur.execute("SELECT a FROM t")
cur.execute("SELECT b FROM t")   # 上一个结果集被丢弃，白查了
```

**坑 2：该用 fetchone 却用了 fetchall 去接单行**

```python
row = cur.fetchall()   # 得到 [({...},)]
row[0]                 # ({...},)  ← 是元组不是 dict！
row[0][0]              # {...}     ← 得写两层
```

## 十、速查表

```python
cur.execute(sql, params)   # 只发命令，不返回结果
cur.fetchone()             # 取1行 → 元组；没数据返回 None（不报错）
cur.fetchall()             # 取全部 → [元组, 元组...]；没数据返回 []
row[0]                     # 取第0列
row[0] if row and row[0] else {}   # 两种空都兜底成 {}
JSONB 列                   # 存要 json.dumps，取是 dict（驱动自动 parse）
```

## 十一、一句话带走

> `execute` 是下单，`fetch` 是取餐，**不 fetch 就什么都没有**。
> 写（UPDATE/INSERT）不用 fetch，读（SELECT）必须 fetch。
