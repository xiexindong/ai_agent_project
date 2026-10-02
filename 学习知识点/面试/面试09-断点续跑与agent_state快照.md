# 面试09：断点续跑——整段覆盖式存档为什么反而安全？

> 配套代码：`modules/agent/graph.py:268`（save 节点）、`modules/memory/pg_manager.py:456`（update_session_state）、`modules/agent/state.py:42-149`、`init_db.sql:20`（agent_state JSONB 列）
> 一句话结论：**断点续跑 = 每轮对话结束把整个 AgentState 序列化成 JSON 存进 `chat_session.agent_state`（JSONB），整段覆盖不合并；覆盖安全的前提是快照永远由"单一数据源"全量生成，合并反而会引入新旧状态交织的一致性问题。**

---

## 一、先说结论

| 面试问 | 一句话答 |
|---|---|
| 存什么？ | AgentState 全量 30 个字段（任务、待办、已完成、关键结果、循环轮次等） |
| 存哪？ | 会话主表 `chat_session.agent_state`，JSONB 列，一行会话一份快照 |
| 什么时候存？ | LangGraph 的 save 节点（answer 之后、END 之前），每轮必跑 |
| 覆盖还是合并？ | **整段覆盖**（`SET agent_state = %s`），不做字段级合并 |
| 怎么恢复？ | 新一轮 `get_session_state(session_id)` 读出来 `AgentState.from_dict()` 还原现场 |

---

## 二、大白话拆解：存档-读档完整链路

```
graph.py:271  _graph.invoke(initial_state)            图开始跑
   ↓ prepare → plan → reason → act（循环）→ answer
   ↓
graph.py:268  save 节点:  agent_state = AgentState.from_dict(state)   # dict → dataclass
                self.agent.memory.pg.update_session_state(
                    session_id, agent_state.to_dict())               # dataclass → dict
   ↓
pg_manager.py:456  UPDATE chat_session SET agent_state = %s
                   WHERE session_id = %s          ← 整段覆盖，不是 MERGE
   ↓
数据库  chat_session.agent_state（JSONB）

—— 会话中断 / 进程重启 ——

新一轮进来 → pg.get_session_state(session_id)   # 读出 JSONB → dict
          → AgentState.from_dict(d)              # dict → dataclass，现场还原
```

| 部分 | 大白话解释 |
|---|---|
| `AgentState`（dataclass） | Agent 的"全部身家"：干了什么、干到哪、下一步是什么 |
| `to_dict()` / `from_dict()` | 存档时的"打包/拆包"：LangGraph 节点间只传 dict，业务代码用 dataclass 点属性 |
| JSONB 列 | PG 里能直接按 JSON 键查询的二进制 JSON，一列装下整个状态对象 |
| save 节点位置 | answer 之后 END 之前——**每轮必经**，不存在"某轮忘了存"的旁路 |

---

## 三、输入 → 输出对照

```python
# 存档（graph.py:268 → pg_manager.py:456）
# 输入: session_id="abc123",
#        agent_state = {"task": "整理报销单", "todos": ["收发票", "填表单"],
#                       "completed": ["收发票"], "iteration": 3, ...（全量 30 字段）}
# SQL: UPDATE chat_session SET agent_state = '{"task": "整理报销单", ...}',
#       updated_at = NOW() WHERE session_id = 'abc123'

# 读档（pg_manager.py:491 get_session_state）
# 输入: session_id="abc123"
# 输出: {"task": "整理报销单", "todos": ["填表单"], "completed": ["收发票"], ...}
#       没有记录 / 状态为空 → {}（新建会话 create_session 只插三个基本字段，天然是 {}）

# 恢复
state = AgentState.from_dict({"task": "整理报销单", ...})
# → state.task == "整理报销单"   ← Agent 接着上次的进度继续
```

---

## 四、为什么要这么写

| 做法 | 原因（不这么写会怎样） |
|---|---|
| **整段覆盖**不合并 | 快照由当前内存 state 全量生成，单一数据源；字段级 MERGE 要处理"新值缺字段时取旧值还是删掉"的语义黑洞，两边判断不一致就出现"半新半旧"的缝合状态 |
| 存会话主表不存历史表 | 断点续跑要的是"最新一份快照"，历史表每轮插一行：查最新要排序、旧行还得清理 |
| JSONB 不拆成 30 个列 | 状态结构随迭代会加字段：拆列每次都要 ALTER TABLE；JSONB 加字段零成本（`from_dict` 里 `d.get(键, 默认)` 兜底，state.py:202） |
| save 放图的固定节点 | 在业务代码里散落存档点：某个分支忘了存，恢复到的是上上轮的旧进度 |
| dataclass 与 dict 成对转换 | 只存 dict：业务代码全是 `state["task"]` 字符串索引，拼错键名运行时才炸；dataclass 点属性 IDE 直接查错 |

---

## 五、等价写法：LangGraph 原生 checkpoint

LangGraph 自带 checkpointer（`config.py` 里有 `checkpoint_enabled` 开关），把每一步 state 持久化到线程。本项目的 `agent_state` 快照是**业务级存档**（一轮一份、人可读、含业务语义），checkpoint 是**框架级存档**（一步一份、为图恢复服务）——两者粒度与用途不同，本项目业务侧选择自己存。

---

## 六、简单总结

1. 链路四步：answer → save 节点 → `update_session_state`（JSONB 整段覆盖）→ 新一轮 `get_session_state` + `from_dict` 还原
2. 覆盖安全的前提：快照**永远由内存全量生成**（单一数据源）；合并式更新才会出现新旧交织
3. 两个兜底：新会话天然 `{}`（create_session 不写该列）；`from_dict` 用 `d.get(键, 默认值)` 容忍结构演进（state.py:202）
