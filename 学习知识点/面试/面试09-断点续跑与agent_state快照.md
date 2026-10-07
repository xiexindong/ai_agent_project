# 面试09：断点续跑——整段覆盖式存档为什么反而安全？

> 配套代码：`modules/agent/graph.py:244-289`（_save_node 快照落库）、`modules/memory/pg_manager.py:498`（update_session_state）、`pg_manager.py:533`（get_session_state）、`modules/agent/state.py:71-219`（AgentState）、`init_db.sql:20`（agent_state JSONB 列）
> 一句话结论：**断点续跑 = 每轮对话结束把整个 AgentState 序列化成 JSON 存进 `chat_session.agent_state`（JSONB），整段覆盖不合并；覆盖安全的前提是快照永远由"单一数据源"全量生成，合并反而会引入新旧状态交织的一致性问题。**

---

## 〇、名词先扫盲（大白话版）

| 术语 | 大白话解释 |
|---|---|
| **断点续跑** | 像打游戏存档：游戏闪退了，重进游戏读档，接着上次的关卡打，而不是从第一关重来。Agent 进程崩了 / 重启了，读回上次的状态接着算 |
| **AgentState** | Agent 的"全部身家"打包成一个对象：问了什么、干到第几步、调过哪些工具、拿到了什么结果 |
| **快照（snapshot）** | 给 Agent 此刻的"脑子"拍张照片存进数据库，以后照着照片还原 |
| **JSONB** | PostgreSQL 里的一种列类型：一列就能装下整个 JSON 对象，还能按里面的键查询。相当于数据库表里嵌了一个"小文件柜" |
| **整段覆盖 vs 字段级合并** | 覆盖 = 把旧文件整个删掉换上新文件；合并 = 打开旧文件逐行对比着改。前者简单粗暴但不会出"半新半旧"的怪文件 |
| **dataclass / TypedDict** | Python 里两种"规定好有哪些字段"的结构：dataclass 给业务代码用（点属性访问，写错字段名 IDE 直接报错），TypedDict 给 LangGraph 框架用（框架只认普通 dict） |
| **序列化 / 反序列化** | 序列化 = 把内存对象"打包"成可存储的 JSON 字符串；反序列化 = "拆包"还原成对象。本项目里就是 `to_dict()` / `from_dict()` 这对方法 |

---

## 一、先说结论

| 面试问 | 一句话答 |
|---|---|
| 存什么？ | AgentState 全量约 30 个字段（任务、待办、已完成、关键结果、循环轮次、工具流水等） |
| 存哪？ | 会话主表 `chat_session.agent_state`，JSONB 列，一行会话一份快照（init_db.sql:20） |
| 什么时候存？ | LangGraph 的 save 节点（answer 之后、END 之前），每轮必跑，不存在旁路 |
| 覆盖还是合并？ | **整段覆盖**（`UPDATE ... SET agent_state = %s`），不做字段级合并 |
| 怎么恢复？ | 新一轮 `get_session_state(session_id, user_id)` 读出 dict → `AgentState.from_dict(d)` 还原现场 |

---

## 二、大白话拆解：存档-读档完整链路

```
graph.py:434  _graph.invoke(initial_state)            图开始跑（新任务入口）
   ↓ prepare → (plan) → reason → act（循环）→ answer
   ↓
graph.py:244  _save_node（answer 之后、END 之前，每轮必经）
   ├─ graph.py:262  save_turn(...)        先存本轮对话（原文+摘要+向量）
   ├─ graph.py:278  agent_state = AgentState.from_dict(state)   # dict → dataclass
   ├─ graph.py:279  snapshot = agent_state.to_dict()            # dataclass → dict
   ├─ graph.py:282  snapshot.pop("principal", None)             # 活体权限对象没法存 JSON，剔除
   └─ graph.py:283  pg.update_session_state(session_id, snapshot)
   ↓
pg_manager.py:523-528  UPDATE chat_session SET agent_state = %s, ...
                       WHERE session_id = %s        ← 整段覆盖，不是 MERGE
   ↓
数据库  chat_session.agent_state（JSONB 列，init_db.sql:20 默认 '{}'）

—— 会话中断 / 进程重启 ——

新一轮进来 → pg_manager.py:533  get_session_state(session_id, user_id)   # 读出 JSONB → dict
          → state.py:171  AgentState.from_dict(d)                        # dict → dataclass，现场还原
```

核心代码片段（存档端，modules/agent/graph.py:277-283）：

```python
# graph.py:277-283 —— save 节点里的快照落库
from modules.agent.state import AgentState
agent_state = AgentState.from_dict(state)
snapshot = agent_state.to_dict()
# principal 是 Principal 活体对象，json.dumps 无法序列化，会让整段落库抛异常
# 且快照里也不需要它（续跑时由请求方重新解析传入），落库前剔除
snapshot.pop("principal", None)
self.agent.memory.pg.update_session_state(session_id, snapshot)
```

核心代码片段（落库 SQL，modules/memory/pg_manager.py:522-528）：

```python
# pg_manager.py:522-528 —— 整段覆盖，注意没有字段级 MERGE
self._cur.execute(
    "UPDATE chat_session SET agent_state = %s, updated_at = CURRENT_TIMESTAMP "
    "WHERE session_id = %s",
    (json.dumps(agent_state, ensure_ascii=False), session_id),  # dict 转 JSON 字符串才能进 JSONB 列
)
self._conn.commit()
```

| 部分 | 大白话解释 |
|---|---|
| `AgentState`（dataclass） | Agent 的"全部身家"：干了什么、干到哪、下一步是什么（state.py:71-114 共约 30 个字段） |
| `to_dict()` / `from_dict()` | 存档时的"打包/拆包"：LangGraph 节点间只传 dict，业务代码用 dataclass 点属性（state.py:118 / 171） |
| JSONB 列 | PG 里能直接按 JSON 键查询的二进制 JSON，一列装下整个状态对象 |
| save 节点位置 | answer 之后 END 之前——**每轮必经**，不存在"某轮忘了存"的旁路 |
| `snapshot.pop("principal")` | 权限对象是"活体"（连着当前请求上下文），像门禁卡一样不能复印存档；下次进门重新发卡 |

---

## 三、输入 → 输出对照

```python
# 存档（graph.py:283 → pg_manager.py:498 update_session_state）
# 输入: session_id="abc123",
#        agent_state = {"user_query": "整理报销单", "iteration": 3,
#                       "sub_tasks": [...], "current_task_index": 1,
#                       "tool_call_logs": [...], ...（全量约 30 字段）}
# SQL: UPDATE chat_session SET agent_state = '{"user_query": "整理报销单", ...}',
#       updated_at = CURRENT_TIMESTAMP WHERE session_id = 'abc123'

# 读档（pg_manager.py:533 get_session_state）
# 输入: session_id="abc123", user_id="cli-user"
# 输出: {"user_query": "整理报销单", "iteration": 3, ...}
#       没有记录 / 状态为空 → {}（pg_manager.py:560 兜底；
#       新建会话 create_session 只插三个基本字段，agent_state 走列默认值 '{}'，init_db.sql:20）

# 恢复（state.py:171 from_dict）
state = AgentState.from_dict({"user_query": "整理报销单", "iteration": 3, ...})
# → state.user_query == "整理报销单"   ← Agent 接着上次的进度继续
# → 快照里缺的字段自动用默认值补齐（d.get(key, 默认值)，state.py:184-219）
```

**一个容易考的细节**：`get_session_state` 的 SQL 里带了 `user_id`（pg_manager.py:555），所以它同时兼职"越权校验"——用户 A 拿用户 B 的 session_id 来读，查不到东西返回 `{}`，`/chat` 接口直接拿它判 403。

---

## 四、关键设计对照表

| 设计选择 | 本项目做法 | 备选方案 | 为什么不选备选 |
|---|---|---|---|
| 写入方式 | **整段覆盖** | 字段级 MERGE | 合并要回答"新快照缺某字段时，保留旧值还是删掉"的语义黑洞；两边判断不一致就出现"半新半旧"的缝合状态 |
| 存储位置 | 会话主表一行一份 | 历史表每轮插一行 | 续跑只要"最新一份"；历史表要 ORDER BY 取最新 + 定期清旧行 |
| 存储格式 | JSONB 一列 | 拆 30 个独立列 | 状态结构随迭代加字段，拆列每次都要 ALTER TABLE；JSONB 加字段零成本 |
| 存档时机 | 图里固定的 save 节点 | 业务代码里散落存档点 | 散落必有分支漏存，恢复到的是上上轮的旧进度 |
| 状态载体 | dataclass ↔ dict 成对转换 | 全程裸 dict | 裸 dict 全是 `state["task"]` 字符串索引，拼错键名运行时才炸 |

---

## 五、为什么要这么写（深入版）

| 做法 | 原因（不这么写会怎样） |
|---|---|
| **整段覆盖**不合并 | 快照由当前内存 state 全量生成，单一数据源；字段级 MERGE 要处理"新值缺字段时取旧值还是删掉"的语义黑洞，两边判断不一致就出现"半新半旧"的缝合状态 |
| 存会话主表不存历史表 | 断点续跑要的是"最新一份快照"，历史表每轮插一行：查最新要排序、旧行还得清理 |
| JSONB 不拆成 30 个列 | 状态结构随迭代会加字段：拆列每次都要 ALTER TABLE；JSONB 加字段零成本（`from_dict` 里 `d.get(键, 默认)` 兜底，state.py:184-219） |
| save 放图的固定节点 | 在业务代码里散落存档点：某个分支忘了存，恢复到的是上上轮的旧进度 |
| dataclass 与 dict 成对转换 | 只存 dict：业务代码全是 `state["task"]` 字符串索引，拼错键名运行时才炸；dataclass 点属性 IDE 直接查错 |
| 落库前剔除 principal | principal 是连着当前请求的活体权限对象，`json.dumps` 序列化它直接抛异常，整段快照落库失败；而且它本就不该"存档"（续跑时重新解析） |

---

## 六、等价写法：LangGraph 原生 checkpoint

LangGraph 自带 checkpointer（`config.py:101` 有 `checkpoint_enabled` 开关，graph.py:125-130 构建时尝试挂 `MemorySaver`），把每一步 state 持久化到线程。本项目的 `agent_state` 快照是**业务级存档**（一轮一份、人可读、含业务语义），checkpoint 是**框架级存档**（一步一份、为图恢复服务）——两者粒度与用途不同。graph.py:416-424 的 `resume` 逻辑用的就是 checkpointer：发现上次的图没跑完（`snapshot.next` 非空），就从断点节点继续 `invoke(None, config)`。

| 对比项 | 业务级快照（agent_state 列） | 框架级 checkpoint |
|---|---|---|
| 粒度 | 一轮对话一份 | 图里每跳一步一份 |
| 存哪 | PG 的 chat_session 表 | checkpointer 后端（本项目是内存 MemorySaver） |
| 给谁看 | 人可读、含业务语义（任务进度等） | 框架自己恢复用 |
| 进程重启后 | 还在（PG 持久化） | 内存版进程一停就没 |

---

## 七、可能的追问与回答要点

**Q1：整段覆盖，两个请求同时写同一会话怎么办（并发写）？**
答：① 业务上一个 session_id 同一时刻只有一个请求在处理（会话级串行）；② 即使并发，PG 的 UPDATE 有行锁，后到者覆盖先到者——这正是"整段覆盖"的语义：最后一份快照就是真相，不需要合并；③ 如果换成字段级合并，并发下两个请求各改一半字段，才会真的交织出脏状态。

**Q2：快照越来越大怎么办？**
答：① 字段大头是 tool_call_logs / messages 这类流水，生产上可以在 `to_dict()` 后做瘦身（比如日志只留最近 N 条）；② JSONB 在 PG 里有 TOAST 压缩，几十 KB 级的快照完全不是事；③ 真到量级问题，再考虑把流水字段拆出去只留"恢复所必需"的字段子集。

**Q3：为什么不用 Redis 存快照？**
答：① 快照与会话记录强绑定（一行会话一份状态），放同一张表事务内一起更新，不存在两边不一致；② Redis 是缓存语义（可淘汰、会过期），断点续跑要的是"肯定在"的持久化；③ 读路径上 `get_session_state` 顺带做了 user_id 越权校验，换 Redis 这套校验要重写。

**Q4：`from_dict` 遇到旧版快照缺新加的字段会怎样？**
答：不会炸。state.py:184-219 每个字段都是 `d.get(键, 默认值)`，缺啥补默认值。这就是为什么"加字段零成本"——旧快照永远能被新版代码读出来，向后兼容天然成立。

**Q5：save 节点抛异常会怎样？**
答：graph.py:287-288 有 try/except 兜底，落库失败只记 error 日志，不影响本轮答案返回给用户——答案已经在 answer 节点生成完了，快照失败只是"这轮没存上"，下轮再存。注意 `update_session_state` 本身是 `reraise=True`（pg_manager.py:497），异常会抛给 save 节点的 except 捕获记录。

---

## 八、简单总结

1. 链路四步：answer → save 节点（graph.py:244）→ `update_session_state`（pg_manager.py:498，JSONB 整段覆盖）→ 新一轮 `get_session_state`（pg_manager.py:533）+ `from_dict`（state.py:171）还原
2. 覆盖安全的前提：快照**永远由内存全量生成**（单一数据源）；合并式更新才会出现新旧交织
3. 两个兜底：新会话天然 `{}`（init_db.sql:20 列默认值）；`from_dict` 用 `d.get(键, 默认值)` 容忍结构演进（state.py:184-219）
