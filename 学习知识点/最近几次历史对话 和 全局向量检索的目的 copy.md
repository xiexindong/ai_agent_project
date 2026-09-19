# 整段代码拆解（PostgreSQL）
```python
self._reconnect_if_needed()
self._cur.execute(
    """SELECT user_query, llm_full_answer, llm_summary, created_at
       FROM chat_history
       WHERE session_id = %s AND user_id = %s
       ORDER BY created_at DESC LIMIT %s""",
    (session_id, user_id, rounds),
)
```
> 这是**从数据库 `chat_history` 聊天记录表读取历史对话，当作短期记忆**。

## 1. 逐行解释
1. `self._reconnect_if_needed()`
    检查数据库连接，如果断开了，自动重连PG库，防止查询时报连接异常。

2. SQL语句
```sql
SELECT user_query, llm_full_answer, llm_summary, created_at
FROM chat_history
WHERE session_id = %s AND user_id = %s
ORDER BY created_at DESC
LIMIT %s
```
- `FROM chat_history`：聊天历史表，**每一行代表一轮对话**
  - `user_query`：用户问句
  - `llm_full_answer`：大模型完整原始回答
  - `llm_summary`：回答摘要（做压缩记忆用，可选）
  - `created_at`：这条记录的创建时间
- `WHERE session_id = %s AND user_id = %s`
  筛选条件：**同一个用户 + 同一个对话会话**
  > `session_id`：会话ID，一个聊天窗口对应一个session_id
  > `user_id`：用户账号ID，区分不同用户
- `ORDER BY created_at DESC`
  **按时间倒序，最新的对话排在最前面**
- `LIMIT %s`：取多少轮历史，参数就是变量 `rounds`，比如 `rounds=5` 就是拿最近5轮对话

3. 参数 `(session_id, user_id, rounds)`
`%s` 是psycopg2的参数占位符，防止SQL注入，依次填入：
`session_id` → `user_id` → `rounds`

## 2. 拿到数据之后，你还要自己反转顺序！（重点坑）
因为 `ORDER BY created_at DESC` 拿到的结果顺序是：
> 最新对话 → 次新 → ... → 最早对话

而给大模型的上下文，**必须是【旧对话在前，新对话在后】**。

示例：
数据库返回行顺序：
1. 第5轮（最新）
2. 第4轮
3. 第3轮

你需要 `histories = list(reversed(rows))` 反转，变成：
> 第3轮 → 第4轮 → 第5轮（交给LLM的正确顺序）

```python
rows = self._cur.fetchall()
# rows: [(user_q5, ans5, summary5, time5), (user_q4, ans4, summary4, time4), ...]
histories = list(reversed(rows))
```

## 3. 这个短期记忆完整流程
```
用户发新消息
    ↓
执行这条SQL，根据user_id+session_id，取出最近N轮聊天记录
    ↓
反转顺序，组装成对话messages列表
    ↓
把历史 + 当前用户新query一起塞进prompt，传给大模型
```

## 4. 两种记忆策略，对应你表里的两个字段
1. **完整记忆（直接用 llm_full_answer）**
直接把原始问答全部塞给prompt；优点信息完整；缺点token消耗大，轮次一多容易超长。
2. **压缩记忆（用 llm_summary）**
每一轮回答后，单独生成摘要存入`llm_summary`。读取历史时，**用摘要代替原始长回答**，大幅减少token。
> 这就是很多Agent短期记忆的优化手段。

## 5. 和向量长期记忆的区别（你项目里刚好两种都有）
- ✅ **chat_history表（当前这段SQL）= 短期记忆**
  读取最近N轮对话，**直接拼进prompt上下文窗口**，没有向量检索。只拿当前session的最近对话。
- ✅ **向量库 = 长期记忆**
  当对话很久，超出窗口限制，把旧对话向量化存入向量库；当用户提问相关内容，用问句检索向量库，召回很久以前的信息，**不是固定取最近N轮**。

一句话总结：
> 根据`user_id + session_id`，从聊天历史表里，**倒序取出最近 rounds 轮对话记录**，作为短期上下文，读取后反转顺序，拼入prompt。

## 6. 常见后续代码示例
```python
# 执行查询后读取结果
rows = self._cur.fetchall()
# 反转，变成旧对话在前
rows = list(reversed(rows))
messages = []
for q, a, summary, t in rows:
    messages.append({"role":"user", "content":q})
    messages.append({"role":"assistant", "content":a})
# 再追加当前用户最新query
messages.append({"role":"user", "content":new_query})
```

要不要我顺便讲下：什么时候只用`llm_summary`摘要来减少token？