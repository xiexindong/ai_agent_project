# Planning 全流程：从一句话到一份计划、再到最终答案（详细版）

> 本文回答一个问题：**用户敲下一句复杂的话之后，系统到底怎么把它拆成计划、一步步做完、最后给出答案？**
> 全程用同一句 query 贯穿，每一步都标了代码位置；标 **【实测】** 的输出来自真实运行，标 **【示意】** 的是按代码里的格式串推出来的样例值。

**示例 query（本文全程用它）**

```
帮我统计 2024 年 Q4 各区域销售额，然后和 Q3 对比，最后整理成一页分析报告
```

**三条阅读路线**

| 时间 | 路线 |
|---|---|
| 10 分钟 | 第二章全景图 + 第四章十阶段速览表 |
| 40 分钟 | 第五章起逐阶段读：每章都拆成「做什么 / 代码在哪 / 输入 → 输出 / 为什么这么设计」 |
| 15 分钟动手 | 第十八章改配置复现开关与边界；第二十章自测 |

**相关文档**：设计速查见 [`Planning与执行.md`](./Planning与执行.md)；
工具侧治理（act 那一步内部）见 [`Harness执行流程.md`](./Harness执行流程.md) 与 [`工具Harness.md`](./工具Harness.md)；
多 Agent 的规划见 [`多Agent协作全景.md`](./多Agent协作全景.md)。

---

## 一、先看清：Planning 不是一条独立链路，而是 ReAct 图里的一个节点

图结构（`modules/agent/graph.py:63-137`）：

```
prepare ──条件边──┬──▶ plan ──▶ reason ◀──┐
                  ├──▶ reason ◀──────────┐│
                  └──▶ answer            ││
                            │            ││
                            ▼            ││
                           act ──────────┘│   act → reason 构成 ReAct 循环
                                          │
                       answer ──▶ save ──▶ END
```

对应的建图代码：

```python
# graph.py:103-123
add_conditional_edges("prepare", _after_prepare, {"plan": "plan", "reason": "reason", "answer": "answer"})
add_edge("plan", "reason")        # 规划完必进推理
add_conditional_edges("reason", _should_continue, {"act": "act", "answer": "answer"})
add_edge("act", "reason")         # 工具结果回到思考
add_edge("answer", "save"); add_edge("save", END)
```

三个必须先搞清的事实：

1. **plan 只跑一次**：`add_edge("plan", "reason")`（graph.py:112）。`act → reason` 的循环**不再回 plan**——计划是循环开始前定好的，不是每轮重拆。
2. **plan 是可选节点**：`_after_prepare` 判定不复杂就直接 `reason`，规划那次 LLM 调用根本不会发生。
3. **计划存在 state 里，不在 prompt 里**：`sub_tasks / current_task_index / task_results / plan_rationale / plan_replan_count`（state.py:53-57）。这让它能随 state 序列化、断点续跑、被审计看到。

---

## 二、全景图：一句话的完整旅程

```
①  入口  AgentGraph.run(query, session_id, user_id, principal)      graph.py:373
         初始 state 只有 5 个字段 + recursion_limit=30
              │
②  prepare  开预算账 → 注入检测 → 清洗 → 技能 → 记忆 → RAG → 标准答案   react_agent.py:118
              │  （六件事各自 try/except，失败只降级不中断）
③  路由     _after_prepare                                          graph.py:326
             error/已 finished → answer；否则 should_plan() → plan | reason
              │
④  判定     needs_planning：复杂度打分                                planner.py:262
             【实测】score=4/2 → 规划 | ['含序列词','含汇总/对比词','长问句']
              │   ← 不复杂的话这里就拐去 reason，本文故事到此结束
⑤  plan 节点 LLM 拆步骤 → JSON → 解析 → 按 depends_on 拓扑排序          planner.py:299
             【实测】3 步：s1(查Q4) → s2(查Q3) → s3(对比成文)
              │  写进 state.sub_tasks，指针指向第一个 pending
              ▼
⑥  reason #1  预算预检 → L0 工具路由 → 拼 prompt → 注入计划进度 → 问 LLM   react_agent.py:420
              │  【实测】注入文本："## 执行计划（共 3 步，已完成 0）…当前步骤：s1"
              │  LLM → need_tool_call=True, action=data_query
              ▼
⑦  act #1     参数校验 → Harness 准入(三关) → 沙箱执行 → 结果过滤/记账/审计  react_agent.py:621
              │  observation="Q4: 华东1200万/华北980万/华南860万"
⑧  推进       _advance_plan：s1 标 done，结果入账，指针移到 s2              react_agent.py:396
              │  （失败则 replan，只重排剩余步骤）
              └──▶ 回到 ⑥ reason #2：prompt 里的计划进度已变成"已完成 1"
                        …… 循环直到所有步骤 closed
              │
⑨  收敛       is_complete → progress_text 提示"所有步骤已完成，请汇总作答"   planner.py:203
              │
⑩  answer     摘要 → save 落库 → 自动初审入队                            graph.py:218/233
```

---

## 三、逐轮时序图：state 是怎么一轮轮长的

以本例（3 步，s2 失败一次后重排为 r1/r2）为例：

| 轮次 | 进入时状态 | 本轮动作 | 出来时 `sub_tasks` |
|---|---|---|---|
| prepare | 空 | 召回 + 清洗 | `[]` |
| plan | 无计划 | 生成 3 步 | `[s1○ s2○ s3○]` |
| reason#1 | 当前步 s1 | 决定调 `data_query` | 不变 |
| act#1 | — | 执行成功 | `[s1✓ s2○ s3○]` |
| reason#2 | 当前步 s2 | 决定调 `data_query` | 不变 |
| act#2 | — | **执行失败** | `[s1✓ s2✗ s3○]` |
| （act#2 内） | — | **replan** | `[s1✓ s2✗ r1○ r2○]` |
| reason#3 | 当前步 r1 | 决定调 `data_query` | 不变 |
| act#3 | — | 执行成功 | `[s1✓ s2✗ r1✓ r2○]` |
| reason#4 | 当前步 r2 | 决定调 `report_write` | 不变 |
| act#4 | — | 执行成功 | `[s1✓ s2✗ r1✓ r2✓]` |
| reason#5 | 全部 closed | 看到"请汇总作答" → 不再调工具 | 不变 |
| answer/save | finished | 出答案 + 落库 | 不变 |

（○ pending、✓ done、✗ failed）

**总共 5 轮 reason、4 次工具调用、1 次规划、1 次重规划**——这就是"先规划再执行"与裸 ReAct 的差别：轮次一样，但每一轮都有全局坐标。

---

## 四、十阶段速览

| # | 阶段 | 代码 | 本例实测 |
|---|---|---|---|
| ① | 入口建 state | `graph.py:373` | 5 字段 + `recursion_limit=30` |
| ② | prepare 召回 | `react_agent.py:118` | 开预算账；注入检测未命中；技能/记忆/RAG 各自降级保护 |
| ③ | 路由 | `graph.py:326` | 返回 `"plan"` |
| ④ | 复杂度打分 | `planner.py:262` | `score=4/2 → 规划` |
| ⑤ | 生成计划 | `planner.py:299` | `3 步 \| rationale=先取两个季度的事实数据，再对比，最后成文` |
| ⑥ | reason | `react_agent.py:420` | 注入计划进度后问 LLM → 决定调工具 |
| ⑦ | act | `react_agent.py:621` | observation 写入工具返回 |
| ⑧ | 推进 / 重排 | `react_agent.py:396`、`planner.py:363` | `已重规划: 保留 1 步完成项，重排 2 步（第 1 次）` |
| ⑨ | 收敛 | `planner.py:203` | `状态：所有步骤已完成，请基于下面的结果汇总出最终答案` |
| ⑩ | answer + save | `graph.py:218/233` | 答案 + 摘要入库；因有工具失败，挂一条 `tool_fail` 复核 |

---

## 五、① 入口：初始 state 只有 5 个字段

```python
# graph.py:393-401
initial_state = {
    "session_id": session_id,   # 兼作 thread_id（断点续跑的钥匙）
    "user_id": user_id,
    "user_query": query,        # 原始提问；prepare 里才清洗
    "principal": principal,     # 权限载体；None = 未启用权限
    "prepared": False,          # 防 prepare 重复召回
}
config = {"configurable": {"thread_id": session_id}, "recursion_limit": 30}
```

| 设计点 | 为什么 |
|---|---|
| `session_id` 兼 `thread_id` | 断点续跑时用它取回快照；也是预算账本的 key |
| `recursion_limit=30` | 一轮 ReAct 至少 2 跳（reason→act→reason），`max_iterations=10` 约 20 跳，30 留余量；超了 LangGraph 抛 `GraphRecursionError`，是框架级防死循环 |
| `prepared=False` | LangGraph 可能重复进入同一节点，用它保证召回只做一次 |

**断点续跑**（graph.py:403-410）：开启 `checkpoint_enabled` 且检测到未完成快照 → `invoke(None, config)` 接着跑。计划进度存在 state 里，所以"做到第几步"能原样恢复。

---

## 六、② prepare：一次性把上下文备齐

`react_agent.py:118-215`。六件事**按固定顺序**做，每件都包 try/except——**任何一件失败都只降级，不中断**：

| # | 做什么 | 代码 | 产物（state 字段） | 失败时 |
|---|---|---|---|---|
| 1 | 开预算账 | 137-138 `begin_run(session_id)` | `RunBudget`（不在 state，挂在 Harness） | Harness 为 None 则整段跳过 |
| 2 | Prompt 注入检测 | 140-149 | `error="prompt_injection_blocked"` + `finished=True` | 命中即返回，不再往下 |
| 3 | 输入清洗 | 151 `sanitize_input` | `user_query` | — |
| 4 | 技能 SOP 匹配 | 160-169 | `active_skill` / `skill_context` | 降级：按通用助手流程 |
| 5 | 记忆召回 | 171-187 | `memory_context` / `long_term_memory` / `short_term_memory` / `memory_compressed` / `memory_low_confidence` | 降级：关闭长期记忆 |
| 6 | RAG 召回 | 189-214 | `rag_context` / `rag_sources`；命中标准答案则前置 | 降级：rag_context 置空 |

### 6.1 注入检测为什么排第一

```python
# react_agent.py:140-149
if check_prompt_injection(query):
    return AgentState(..., finished=True,
                      final_answer="检测到潜在的指令注入，已拦截。请使用正常方式提问。",
                      error="prompt_injection_blocked")
```

它直接构造一个"已结束"的 state。`_after_prepare`（graph.py:332）看到 `error` 就返回 `"answer"`，**根本不进 reason**——否则 LLM 会把拦截话术覆盖掉，守卫形同虚设。

### 6.2 prepare 与本主题的接缝：技能 SOP 会影响打分

`skill_context` 会一路传给规划器（react_agent.py:332），而 SOP 本身是多步的（同时含 `"1."` 与 `"2."`）时，复杂度打分 **+1**（planner.py:286-288）。也就是说：**命中流程型技能，本身就会把这条 query 推向"该规划"**。

### 6.3 prepare 结束时 state 大概长这样【示意，字段结构来自 state.py】

```python
{
  "user_query": "帮我统计 2024 年 Q4 各区域销售额，然后和 Q3 对比，最后整理成一页分析报告",
  "active_skill": "销售分析",                 # 若命中
  "skill_context": "1. 取数\n2. 对比\n3. 成文",  # 多步 SOP → 打分 +1
  "rag_context": "【人工审核确认的标准答案】…",
  "memory_context": "上一轮你问过 Q3 销售额…",
  "sub_tasks": [], "current_task_index": 0, "task_results": [],
  "plan_rationale": "", "plan_replan_count": 0,
  "prepared": True, "iteration": 0, "finished": False,
}
```

---

## 七、③ 路由：进不进 plan 节点

```python
# graph.py:326-338
def _after_prepare(self, state: dict) -> str:
    if state.get("error") or (state.get("finished") and state.get("final_answer")):
        return "answer"                                  # 拦截/异常：直接发走
    if self.agent.should_plan(AgentState.from_dict(state)):
        return "plan"
    return "reason"
```

`should_plan`（react_agent.py:317-321）的两道前置闸：

```python
if self.planner is None or state.sub_tasks:
    return False        # 没开规划器 / 已经有计划 → 不规划
return self.planner.needs_planning(state.user_query, state.skill_context)
```

**"已有计划就不规划"** 是关键：重规划只在步骤失败时由 `_advance_plan` 触发，不会每轮反复发生。这保证了**一次运行最多 1 次规划 + `MAX_REPLANS` 次重规划**，成本可预算。

---

## 八、④ 判定：这句话够不够复杂

`planner.py:262-292`。打分制，累计 ≥ `AGENT_PLANNING_MIN_SCORE`（默认 2）才规划。

### 8.1 完整词表

| 信号 | 分 | 词表（planner.py:61-66） | 本例 |
|---|---|---|---|
| 序列词 | **+2** | 然后、接着、之后、随后、最后、接下来、其次、再、先 | ✅「**然后**」「**最后**」 |
| 步骤标记 | **+2** | 第一步、第二步、第三步、1.、2.、3.、①、②、③ | — |
| 汇总对比词 | +1 | 分别、汇总、对比、比较、统计、整理、总结、报告、调研、方案、计划、流程、评估、分析、列出、一步步、逐步 | ✅「**统计**」「**对比**」「**整理**」 |
| 并列项 ≥2 处 | +1 | ；/ ; / 、 / ，然后 / 以及 / 并且 | —（只有 1 处「，然后」，需 ≥2） |
| 长度 ≥ 40 字 | +1 | `len(q) >= 40` | ✅（本句 41 字） |
| 技能 SOP 多步 | +1 | `skill_context` 同时含 "1." 与 "2." | —（本例未命中技能） |

### 8.2 两条短路

```python
if not self.enabled: return False      # 开关关着
if len(q) < 8:       return False      # "你好"、"3*7" 这类
```

### 8.3 【实测】日志

```
规划判定: score=4/2 → 规划 | ['含序列词', '含汇总/对比词', '长问句']
```

**为什么是"打分"而不是"让 LLM 判断"**：判定本身若再调一次 LLM，就等于为了省一次 LLM 而多花一次 LLM。关键词打分零成本、可解释（日志直接列出命中原因）、可配置（`MIN_SCORE`）。

---

## 九、⑤ plan 节点：LLM 把一句话拆成有序步骤

### 9.1 给模型的输出契约（精简 Schema，省 token）

```python
# planner.py:69-70
_PLAN_SCHEMA = ('{"rationale": "string", "steps": [{"id": "string", "instruction": "string", '
                '"tool_hint": "string", "depends_on": ["string"]}]}')
```

只给"字段名→类型"，不给示例——token 省一半以上（与多 Agent 的 prompt 同做法）。

### 9.2 提示词全文结构（planner.py:313-327）

```
你是一个任务规划器。请把下面这个请求拆成有序的执行步骤。

## 用户请求
{query}

## 可用工具
{tools_desc}            ← _tools_overview()：react_agent.py:386，"- 工具名: 描述" 逐行

## 作业流程（可选）
{skill_context}         ← 技能 SOP

## 输出要求
只输出一段 JSON，不要任何解释文字，格式如下：{_PLAN_SCHEMA}
约束：
1. steps 最多 6 步，能一步做完就不要拆
2. instruction 是一句话的具体动作，不要写'然后'这类连接词
3. tool_hint 填工具名，用不上工具就填空字符串
4. depends_on 填它依赖的步骤 id（如 s2 要用 s1 的结果就填 ['s1']），无依赖填空数组
5. 必须按依赖顺序排列：被依赖的步骤排在前面
6. rationale 用一句话说明为什么这么拆
```

system prompt：`"你是严谨的任务规划器，只输出 JSON。"`

### 9.3 解析：三次机会

`_parse_json`（planner.py:443-459）：

```
① json.loads(text)                     直接解析
② 有 ``` 就剥掉代码块外壳再解析        模型爱用代码块包 JSON
③ repair_json(text)                    修"近似 JSON"（尾逗号/缺括号）
④ 都失败 → 返回 None → 降级单步计划
```

### 9.4 拓扑排序：把顺序纠正过来

`_sort_steps`（planner.py:463-485）按 `depends_on` 做 Kahn 排序。

**【实测】**：模型返回顺序是 `s1, s3, s2`（s3 依赖 s1、s2），排序后：

```
s1 | 查询 2024 Q4 各区域销售额   | hint=data_query | deps=[]         | pending
s2 | 查询 2024 Q3 各区域销售额   | hint=data_query | deps=[]         | pending
s3 | 对比两季度数据并整理成一页报告 | hint=          | deps=['s1','s2'] | pending
```

**不排序会怎样**：模型第一轮就可能去做还没数据可对比的 s3。

**防死循环**（470 行）：最多循环 `len(steps)` 轮；依赖成环时排不完，剩余按原顺序补在最后（482-484），不会卡住。引用了不存在的 id 直接忽略（475 行）。

### 9.5 写回 state，指针指向第一个待做步骤

```python
# react_agent.py:352-358
state.sub_tasks = [s.to_dict() for s in plan.steps]
state.plan_rationale = plan.rationale
state.plan_replan_count = plan.replan_count
state.current_task_index = next((i for i, s in enumerate(plan.steps) if s.is_open), len(plan.steps))
```

`_plan_node`（graph.py:160-172）把 5 个字段回写给图状态——**不回写 reason 看到的就是空计划**。

**【示意】日志**

```
计划已生成: 3 步 | rationale=先取两个季度的事实数据，再对比，最后成文
计划已写入 state: 3 步 | 思路=先取两个季度的事实数据，再对比，最后成文
```

---

## 十、⑥ reason 一轮内部：七步走完才问模型

`reason_step`（react_agent.py:420-597）。一次调用改 4 个字段：`reasoning / action / need_tool_call / final_answer`。

### 10.1 步 1：预算预检（461-472）

```python
budget = self.tool_harness.budget(state.session_id)
if budget.expired or budget.tokens >= budget.spec.token_budget:
    state.need_tool_call = False; state.action = {}; state.finished = True
    state.final_answer = state.final_answer or "已达到本次运行的工具调用预算（时间或上下文额度用尽），以下基于已有信息作答。"
    return state          # 提前返回：不再花这次 LLM 调用
```

| 闸 | 判据 | 默认 |
|---|---|---|
| 墙钟超时 | `deadline = 建账时刻 + timeout_seconds` | 120 秒 |
| 上下文 token | `tokens >= token_budget` | 6000 |

**这里不查次数**：次数超限由 `BudgetMiddleware` 在工具调用前拦（middleware.py:274）。本处管的是"还值不值得再问一次模型"——时间和上下文都没了，问了也只会再要一次工具。

### 10.2 步 2：L0 工具路由（474-504）

```python
if not state.routed_tools:                      # 只在第 1 轮算
    decision = self.router.route(state.user_query, registry=..., skill_name=state.active_skill)
    state.routed_tools = list(decision.tool_names)
selected = [t for t in self.tool_registry.list_tools() if t.name in state.routed_tools]
```

- **只算一次**：每轮重算要再调一次 embedding；且名单中途变化会让模型困惑（上一轮还在的工具这轮没了）。
- **这一步之后远程工具才真正就位**（486-487 行注释）：网关目录里的工具此刻才连 server、注册进 registry；没被召回的远程工具不会起进程、不会进 prompt。

**【示意】日志**：`本轮工具名单 3/3 个（候选池 12 个）: data_query、report_write、calculator`

### 10.3 步 3：权限过滤 + 生成工具 schema（505-513）

```python
allowed = self._allowed_tools(state.principal, selected)        # 按角色/等级过滤
tools_schema = self.tool_registry.get_openai_tools_schema(user_level=..., allowed=allowed)
tools_desc = "\n".join(f"- {t['function']['name']}: {t['function']['description']}" ...)
```

模型看到的工具清单 = 路由召回 ∩ 权限允许。**这份 `tools_desc` 同时也是传给规划器的工具清单**（规划在 reason 之前，用的是 `_tools_overview()`，react_agent.py:386）。

### 10.4 步 4：拼 prompt（八个槽位，529-538）

```python
rec = self._builder.build_record(
    state.user_query,                                   # 槽位 1
    permission_context=self._permission_context(...),    # 2
    skill_context=state.skill_context or "（未匹配到专门技能…）",   # 3
    rag_context=state.rag_context or "（知识库无相关内容）",        # 4
    memory_context=state.memory_context or "（无历史记忆）",        # 5
    tool_history=tool_history,                          # 6（本轮已调过的工具及结果）
    observation=state.observation or "（初始状态，暂无工具观察）",   # 7
    tools_desc=tools_desc,                              # 8
)
```

好处（521-527 行注释）：变量表由模板自己算，少传一个就抛 `PromptError`（不会出现"槽位空着照样发请求"的静默降质）；上下文按 token 预算裁剪；返回值带 `prompt_tokens` 与 `trimmed`。

裁剪发生时：`提示词上下文超预算已裁剪: {...}`（540 行）。

### 10.5 步 5：注入计划进度（543-547）

```python
plan_text = self._plan_context(state)      # → plan.progress_text()
if plan_text:
    rec = self._with_plan(rec, plan_text)  # 追加到最后一条 user 消息末尾
```

**为什么不走槽位而是直接拼在 user 消息里**：模板槽位固定 8 个，多传会被丢弃；直接拼则"没有计划时 messages 一字不动"——保证关闭规划时行为与改造前逐字一致。

**【实测】reason#1 注入的内容**

```
## 执行计划（共 3 步，已完成 0）
总体思路：先取两个季度的事实数据，再对比，最后成文
当前步骤：s1 —— 请只围绕这一步决定要不要调工具、调哪个工具。
- [ ] s1: 查询 2024 Q4 各区域销售额（当前步骤）
- [ ] s2: 查询 2024 Q3 各区域销售额
- [ ] s3: 对比两季度数据并整理成一页报告
```

**【实测】act#1 之后（reason#2）注入的内容**

```
## 执行计划（共 3 步，已完成 1）
总体思路：先取两个季度的事实数据，再对比，最后成文
当前步骤：s2 —— 请只围绕这一步决定要不要调工具、调哪个工具。
- [x] s1: 查询 2024 Q4 各区域销售额 → Q4: 华东1200万/华北980万/华南860万
- [ ] s2: 查询 2024 Q3 各区域销售额（当前步骤）
- [ ] s3: 对比两季度数据并整理成一页报告
```

差别只有两处：`已完成 0 → 1`、`s1` 从 `[ ]` 变 `[x]` 且挂上了真实产出。**模型每轮都看着这张"进度表"决定下一步**——这就是 Plan-and-Execute 相对裸 ReAct 的全部增量。

### 10.6 步 6：问模型（549-552）

```python
result = self.llm.invoke_with_tools(messages=rec["messages"], tools=tools_schema)
```

（`invoke_with_tools` 内部会对模型返回的非法 tool_calls JSON 做 `repair_json` 修复——尾逗号/缺括号都能救回来。）

### 10.7 步 7：决策 + 两道刹车（557-595）

```python
if tool_calls and state.iteration < state.max_iterations:
    repeated = self._last_same_success_call(state, tool_calls[0])
    if repeated is not None:              # 刹车①：同名同参且成功过 → 不再执行
        state.need_tool_call = False
        state.observation = f"工具 {name} 已用相同参数执行过，结果如下（不要重复调用，直接据此作答）：\n{result}"
        return state
    state.need_tool_call = True; state.action = tool_calls[0]
else:
    state.need_tool_call = False
    state.final_answer = result["content"] or "抱歉，我暂时无法回答这个问题。"

if state.iteration >= state.max_iterations and state.need_tool_call:   # 刹车②
    state.need_tool_call = False
    state.final_answer = result["content"] or "已达到最大推理次数，基于当前信息给出最佳答案。"
```

| 刹车 | 判定 | 实测效果 |
|---|---|---|
| 重复调用 | 本轮动作与"最近一次成功调用"工具名+参数全等（612-617） | 同一句"3*7"从 9 次调用 / 14 秒压到 1 轮 |
| `max_iterations=10` | 轮数到顶仍要调工具 | 强制收尾，防无限循环 |

**与 Planning 的关系**：计划保证**不漏步**，两道刹车保证**不走偏、不打转**，互补而非替代。

---

## 十一、⑦ act 一步内部：六步流水线

`act_step`（react_agent.py:621-682）先做参数检查，再交给 `tool_registry.execute_tool`；后者的六步流水线见 `modules/tools/tool_registry.py:188-336`。

### 11.1 参数缺失 → 不硬调，改为追问（629-644）

```python
if not isinstance(arguments, dict) or not arguments:
    state.observation = "工具 X 缺少必要参数，本轮不执行。请直接向用户追问缺失的参数，不要用猜测的默认值调用。"
    state.action = {}
    return state
```

**为什么不猜默认值**：写库的场景里，猜出来的默认值会造成真实副作用。宁可多一轮对话。

### 11.2 六步流水线

```
① 取能力台账   capability_of(tool) → cap（风险/副作用/沙箱/权限/数据域）
② 准入判定     harness.before_call(req)  → 策略 → 五道中间件 → 人工审批
                不过 → record_intercepted() 留痕，返回中文原因给模型
③ registry 自己的账：单工具限次 + 同参去重（命中缓存直接回传，省一轮 LLM）
④ 执行         harness.execute(...) → 按 cap.sandbox 选工位（进程内/子进程/容器/远程）
⑤ 调用后       harness.after_call(...) → 裁剪 + 脱敏 + token 记账 + 审计
⑥ 成功结果进缓存（只缓存成功的；失败必须让模型看到错误才有机会重试）
```

**两本账并行**（tool_registry.py:201-204）：registry 的计数管"同一会话里别打转"，Harness 的 `RunBudget` 管"这一轮总共能花多少"；都过才放行，互不替代。

### 11.3 ② 准入三关（harness.py:230-276）

顺序固定，**按"拦住它的代价"从低到高排**：

| 关 | 干什么 | 比喻 |
|---|---|---|
| ① 策略裁决 | 查黑名单（禁止工具名 / 风险等级提级） | 查黑名单，最便宜 |
| ② 五道中间件 | schema 校验 → 权限 → 数据范围 → 限流 → 预算 | 过安检，中等 |
| ③ 人工审批 | 高风险工具找人签字，且人可以改参数 | 等人签字，最贵 |

顺序反过来会"先把人叫来，再发现工具在黑名单里"，白等一场。

三种典型结果（harness.py:247-256 注释中的实测）：

| 输入 | 输出 |
|---|---|
| `data_query`（被提级为 high，未开人工通道） | `allowed=False, code=approval_required` |
| `calculator`（low，无覆盖） | `allowed=True` → 直接放行 |
| `drop_table`（命中禁止名单） | `allowed=False, code=policy_denied, reason="工具 drop_table 已配置为禁止执行"` |

**被拦也算一次调用**：`record_intercepted`（harness.py:402-407）会 `budget.record_block()` 并落一条 `state=deny` 的审计——越权尝试正是审计最该看到的。

### 11.4 ④ 沙箱选择（harness.py:328-340）

```python
mode = cap.sandbox or self.default_sandbox          # 工具声明 > 全局默认
timeout = cap.timeout_seconds or self.default_timeout
return self._runner_for(mode).run(tool, params, timeout)
```

四种工位：`none`（进程内）/ `subprocess`（本机子进程）/ `docker`（容器）/ `remote`（远端服务）。**执行器按 mode 缓存**，同一 mode 只建一次（harness.py:305-326）。

### 11.5 ⑤ 调用后三步（harness.py:345-365）

```python
result = self.result_filter.filter(result, req.principal, max_chars=limit)  # 裁剪 + 脱敏
tokens = estimate_tokens(...)
self.budget(req.session_id).consume(req.tool_name, tokens)                  # token 记进预算账
if self.audit_enabled: self._audit(req, result, cap, approved=True)
```

**这段与 Planning 直接相关**：`consume` 累加的 token 就是 reason 里那道"上下文预算"闸的数据来源——**工具返回越长，预算烧得越快，越快触发强制收尾**。

### 11.6 act 收尾（662-682）

```python
log_entry = {"name":…, "arguments":…, "result": str(result.result)[:500] if success else error,
             "success":…, "duration_ms":…}
state.tool_calls_history.append(log_entry)   # 下一轮 prompt 的 tool_history + 重复检测
state.tool_call_logs.append(log_entry)       # save 节点落 tool_call_log 表
state.observation = str(result.result) if success else f"工具执行失败: {result.error}"
self._advance_plan(state, tool_name, result) # ← 与 Planning 的接缝
```

---

## 十二、⑧ 推进与重规划

### 12.1 推进（react_agent.py:396-416）

```python
ok = bool(getattr(result, "success", False))
brief = str(getattr(result, "result", ""))[:200] if ok else str(getattr(result, "error", ""))[:200]
plan.mark(step.id, STATUS_DONE if ok else STATUS_FAILED, brief)
state.task_results.append({"id":…, "instruction":…, "tool":…, "status":…, "result":brief})
if not ok and self.planner is not None:
    plan = self.planner.replan(plan, observation=state.observation, tools_desc=...)
self._save_plan(state, plan)
```

`plan.current()` 取的是**第一个 pending 步骤**（planner.py:147-151），所以"标记完成"天然等于"指针前移"，不需要单独维护下标——这也是为什么 `current_task_index` 只是"展示用"，真正的推进逻辑在 `status` 上。

结果摘要**只留前 200 字**（405 行）：它要进 prompt，太长会吃掉上下文预算。

### 12.2 重规划的四条规则（planner.py:363-436）

| 情况 | 行为 | 代码 |
|---|---|---|
| 关闭重规划 / 次数用尽 | 原样返回，按原计划继续 | 373-376 |
| LLM 不可用 / 返回非法 | 沿用原计划 | 398-407 |
| 哪些要重排 | `pending + failed`，**已完成的一步不动** | 384 |
| 旧步骤怎么处理 | pending → `skipped`；failed **保留 `[!]`** | 429-432 |

重规划提示词带四块上下文：原始请求、已完成步骤（含结果）、待重排步骤、最新观察、可用工具；约束里明确"不要重复已完成步骤，id 用 r1、r2 编号"。

### 12.3 【实测】s2 失败后重排

```
## 执行计划（共 4 步，已完成 1）
总体思路：先取两个季度的事实数据，再对比，最后成文
当前步骤：r1 —— 请只围绕这一步决定要不要调工具、调哪个工具。
- [x] s1: 查询 2024 Q4 各区域销售额 → Q4: 华东1200万/华北980万/华南860万
- [!] s2: 查询 2024 Q3 各区域销售额 → 工具 data_query 执行失败: 无 Q3 明细权限
- [ ] r1: 从年度汇总表取 Q3 各区域销售额（当前步骤）
- [ ] r2: 对比两季度并成文
replan_count: 1 | current(): r1
```

**【实测】日志**：`已重规划: 保留 1 步完成项，重排 2 步 （第 1 次）`

### 12.4 两个反直觉点，如实说明

1. **s3 从列表里消失了**：它原本是 pending，被标 `skipped` 后没有重新拼回 `plan.steps`（432 行 `plan.steps = done_steps + failed_steps + new_steps`）。这是有意的：pending 步骤从没执行过，留着只会让模型困惑。
2. **failed 的 s2 刻意保留**：它挂过这件事是排查时最关键的信息（428 行注释）。rationale 也不改写——避免把"为什么这么做"的历史抹掉。

### 12.5 步骤状态机

```
        ┌──────────────────────────────┐
        │         pending（待做）        │
        └──┬──────────┬───────────┬────┘
           │          │           │
      执行成功      执行失败    重规划时被取代
           ▼          ▼           ▼
         done       failed      skipped
           │          │
           └──────────┴──► 都算 closed（is_complete 通过）
```

`is_complete` 把 `failed` 也算"不再做"（planner.py:174-176）：失败已经写进 observation 告诉模型了；若仍算待办，计划永远完不成，模型会一直卡在"还有步骤没做"。

---

## 十三、⑨ 收敛 → ⑩ 出答案落库

### 13.1 收敛提示（planner.py:203-208）

```
状态：所有步骤已完成，请基于下面的结果汇总出最终答案，不要再调用新的工具。
```

**为什么必须显式提示**：不提示的话模型可能继续找工具调用，表现为"做完了还在调工具"，最后一轮被 `max_iterations` 硬截断，答案质量反而更差。

**【实测】最终进度**

```
## 执行计划（共 4 步，已完成 3）
状态：所有步骤已完成，请基于下面的结果汇总出最终答案，不要再调用新的工具。
- [x] s1: 查询 2024 Q4 各区域销售额 → Q4: 华东1200万/华北980万/华南860万
- [!] s2: 查询 2024 Q3 各区域销售额 → 工具 data_query 执行失败: 无 Q3 明细权限
- [x] r1: 从年度汇总表取 Q3 各区域销售额 → Q3: 华东1050万/华北900万/华南800万
- [x] r2: 对比两季度并成文 → 报告已生成：Q4 环比 +12.5%
is_complete: True
```

（"已完成 3"而不是 4：统计的是 `done()`，`[!]` failed 不计入。）

### 13.2 answer（react_agent.py:686+ / graph.py:218-229）

```python
if state.final_answer and not state.need_tool_call:
    pass                       # 已有答案且不再调工具 → 原样保留，不多花一次调用
else:
    prompt = 用户提问 + 知识库参考 + 工具观察结果 + 推理过程 → llm.invoke
```

随后 `generate_summary` 出一句话摘要，供向量入库。

### 13.3 save（graph.py:233-276）

| 动作 | 内容 |
|---|---|
| `save_turn` | 原文 + 完整答案 + 摘要 + 两个向量（query / summary）+ 工具流水 + RAG 来源 |
| `update_session_state` | 整个 state 序列化存 PG（断点续跑的数据源） |
| `_save_tool_logs` | 工具调用逐条落 `tool_call_log` 表（结果截 2000 字） |
| `_auto_review` | 命中风险则入人工复核队列 |

### 13.4 自动初审四类（graph.py:296-322），只入队优先级最高的一类

| 优先级 | 类型 | 触发 |
|---|---|---|
| 1 | `sensitive` | 注入命中 / 敏感词 |
| 2 | `tool_fail` | **本轮存在工具调用失败** ← 本例 s2 失败，会命中 |
| 3 | `low_sim` | 长期记忆召回低置信 |
| 4 | `no_rag` | 有 RAG 但没召回任何来源 |

---

## 十四、state 字段全生命周期对照

| 字段 | prepare 后 | plan 后 | act#1 后 | 重排后 | 结束时 |
|---|---|---|---|---|---|
| `user_query` | 清洗后的原文 | 不变 | 不变 | 不变 | 不变 |
| `skill_context` | SOP 文本（参与打分） | 不变 | 不变 | 不变 | 不变 |
| `rag_context` | 召回文本 | 不变 | 不变 | 不变 | 不变 |
| `sub_tasks` | `[]` | 3 步全 pending | `s1=done` | `s1 done / s2 failed / r1,r2 pending` | 全 closed |
| `current_task_index` | 0 | 0（指向 s1） | 1（指向 s2） | 2（指向 r1） | — |
| `task_results` | `[]` | `[]` | 1 条 | 2 条 | 4 条 |
| `plan_rationale` | `""` | 「先取两个季度…」 | 不变 | 不变（不改写） | 不变 |
| `plan_replan_count` | 0 | 0 | 0 | **1** | 1 |
| `iteration` | 0 | 0 | 1 | 2 | 5 |
| `observation` | 「初始状态，暂无工具观察」 | 不变 | Q4 数据 | 失败原因 | 最后一步结果 |
| `tool_calls_history` | `[]` | `[]` | 1 条 | 2 条 | 4 条 |
| `finished` | False | False | False | False | True |

（`current_task_index` 在 `_save_plan` 里重算为"第一个 pending 的下标"，重排后实测指向 r1。）

---

## 十五、一次运行的日志时间线

标注：**【实测】** = 真实运行输出；**【示意】** = 按代码中的格式串推出的样例。

```
【示意】Agent执行: session=s1 | user=u1 | 权限=已加载 | query='帮我统计 2024 年 Q4 各区域销售额…'
【示意】节点[prepare]: session=s1
【示意】技能匹配: 销售分析
【实测】规划判定: score=4/2 → 规划 | ['含序列词', '含汇总/对比词', '长问句']     ← planner.py:291
【示意】节点[plan]: 生成执行计划
【实测】计划已生成: 3 步 | rationale=先取两个季度的事实数据，再对比，最后成文      ← planner.py:355
【示意】计划已写入 state: 3 步 | 思路=先取两个季度的事实数据，再对比，最后成文
【示意】节点[reason]: iteration=1 query='帮我统计 2024 年 Q4 各区域销售额...'
【示意】本轮工具名单 3/3 个（候选池 12 个）: data_query、report_write、calculator
【示意】计划上下文已注入 prompt（186 字）
【示意】节点[act]: tool=data_query
【示意】工具调用: data_query | args={'quarter': '2024Q4'}
【示意】计划推进: s1 → 完成 | 进度 1/3
【示意】节点[reason]: iteration=2 …
【示意】节点[act]: tool=data_query
【示意】工具执行失败: data_query | 无 Q3 明细权限
【实测】已重规划: 保留 1 步完成项，重排 2 步 （第 1 次）                        ← planner.py:434
【示意】计划推进: s2 → 失败 | 进度 1/4
【示意】节点[reason]: iteration=3 …
         …… r1 成功、r2 成功 ……
【示意】计划推进: r2 → 完成 | 进度 3/4
【示意】节点[answer]: 生成最终答案
【示意】节点[save]: session=s1
【示意】对话记忆 + Agent状态 已持久化
【示意】自动初审入队: type=tool_fail history_id=…
【示意】Agent完成: iterations=5 | tool_calls=4 | finished=True
```

想自己看全量：把日志级别调到 INFO，直接跑一次带复杂 query 的会话即可。

---

## 十六、七条降级路径：Planning 坏了不影响主流程

| 环节 | 出问题 | 行为 |
|---|---|---|
| 规划器未装配 | `planner is None` | `should_plan` 恒 False，走原 ReAct |
| 开关关闭 | `AGENT_PLANNING_ENABLED=False`（默认） | 同上 |
| 打分不够 | `score < min_score` | 不规划，零额外开销 |
| LLM 不可用 | `create_plan` 抛异常 | 单步计划（`_fallback_plan`） |
| JSON 解析失败 | `_parse_json` 返回 None | 单步计划 |
| 步骤为空 | 解析出 0 步 | 单步计划 |
| 重规划失败/次数用尽 | `replan` 返回原计划 | 按原计划继续做 |

**单步计划长这样**（planner.py:490-494）：

```python
plan.rationale = "规划不可用，按单步执行"
plan.steps = [PlanStep(id="s1", instruction=plan.query or "回答用户的问题")]
```

它让后续所有逻辑（`current()` / `mark()` / `progress_text()`）**无需任何分支判断**——降级后代码路径与正常完全一致，这是"降级不失败"最省事的实现方式。

---

## 十七、异常与边界矩阵

| 场景 | 谁拦 | 模型看到什么 | 计划会怎样 |
|---|---|---|---|
| 工具参数缺失 | `act_step` 629-644 | "缺少必要参数…请向用户追问" | 当前步**不推进**（未调工具），下一步还是它 |
| 工具被策略禁止 | Harness 策略关 | "工具 X 已配置为禁止执行" | 当前步标 failed → 可能触发重排 |
| 高风险未开人工通道 | `_ask_human` 285-289 | "需人工批准但未启用审批通道，已拒绝" | 同上 |
| 工具执行失败 | act 675 | "工具执行失败: …" | 标 failed → 触发重排（若开启） |
| 限流拦截 | 限流中间件 | "工具调用过于频繁（被 sender:U007 限流）" | 同上（**注意：会触发重排**，见下） |
| 预算/超时 | reason 461-472 | 直接收尾，给"基于已有信息作答" | 计划停在当前步 |
| 达到 max_iterations | 592-595 | "已达到最大推理次数…" | 未完成步骤保持 pending |
| LLM 不可用 | 各 try/except | 兜底话术 | 单步计划 / 沿用原计划 |

**一个值得注意的交互**：被限流或被预算拦下时，工具结果是失败的 → `_advance_plan` 会把它标 failed → **若开了重规划，会浪费一次重排机会**去处理一个"等一下就好"的问题。这也是 `AGENT_PLANNING_REPLAN_ENABLED` 默认虽开、但 `MAX_REPLANS` 只给 1 次的原因之一。

---

## 十八、动手验证

### 18.1 开关（`.env`，config.py:104-108）

```bash
AGENT_PLANNING_ENABLED=true           # 总开关（默认 false）
AGENT_PLANNING_MIN_SCORE=2            # 几分才规划
AGENT_PLANNING_MAX_STEPS=6            # 单次最多几步
AGENT_PLANNING_REPLAN_ENABLED=true    # 失败是否重排
AGENT_PLANNING_MAX_REPLANS=1          # 最多重排几次
```

### 18.2 跑测试

```bash
python -m pytest tests/test_agent_planning.py -q
```

### 18.3 五条边界值得手动试

| 试什么 | 预期 |
|---|---|
| `MIN_SCORE=99` | 永不规划，链路与关闭时一致 |
| 让某步工具失败 + `MAX_REPLANS=0` | 日志"跳过重规划"，按原计划继续 |
| 让规划 LLM 返回非法 JSON | 日志"规划降级（按单步执行）"，任务照常完成 |
| 让模型返回 `s1,s3,s2` 乱序 | 排序后仍是 `s1,s2,s3`（按 depends_on） |
| 一句话里塞 8 件事 | 最多截到 `MAX_STEPS=6` 步（planner.py:339） |

---

## 十九、排查手册

| 现象 | 可能原因 | 定位 |
|---|---|---|
| 从不进 plan 节点 | 开关没开 / `planner is None` / 打分不够 | 看"规划判定"日志；没有这行日志说明 `should_plan` 在两道前置闸就返回了 |
| 进了 plan 但只有 1 步 | LLM 返回非法 → 降级单步 | 搜"规划降级（按单步执行）" |
| 计划生成了但模型不按步骤走 | 计划文本没注入 | 看"计划上下文已注入 prompt"；没有则说明 `progress_text()` 返回空（`steps` 为空） |
| 步骤顺序不对 | 拓扑排序失败（依赖成环/引用不存在 id） | planner.py:475 忽略无效依赖，482-484 兜底追加 |
| 做完了还在调工具 | 没触发收敛提示 | 看 progress_text 是否有"所有步骤已完成"；有则说明模型没听，靠 `max_iterations` 兜底 |
| 反复重排 | 每步都失败 | 看 `replan_count`；`MAX_REPLANS` 用尽后日志"跳过重规划" |
| 计划推进了但 state 里没变 | 节点回写漏字段 | graph.py:208-213（act 节点回写 sub_tasks 等 4 个字段） |
| 重排后某步不见了 | pending 被标 skipped 且未拼回 | 设计如此，见 12.4 |

---

## 二十、自测 15 问

1. plan 节点一次运行跑几次？为什么不回 plan？
2. "3*7 等于多少" 会进 plan 吗？在哪一步短路？
3. 模型返回 s1、s3、s2，最终执行顺序是什么？靠什么纠正？
4. 计划存在 prompt 还是 state？存在 prompt 会丢什么能力？
5. reason 一轮里，计划进度是怎么进到模型输入里的？为什么不走 prompt 槽位？
6. 步骤失败后，已完成的那步会被重排吗？为什么？
7. 重排后为什么 s2 还在、s3 不见了？
8. `is_complete` 为什么把 failed 也算完成？
9. 所有步骤做完后 prompt 多出哪句话？少了会怎样？
10. 规划 LLM 挂了会抛异常吗？用户最终看到什么？
11. 被限流拦下的工具调用，会让计划走到哪一步状态？（提示：failed → 可能触发重排）
12. 重排次数用尽后再失败，会发生什么？
13. `current_task_index` 与 `plan.current()` 谁才是真正的推进依据？
14. 工具返回很长会影响什么？（提示：after_call 的 token 记账 → reason 的预算闸）
15. 断点续跑时，计划进度靠什么恢复？

<details>
<summary>答案</summary>

1. 一次；`add_edge("plan","reason")`，`act→reason` 不回 plan。
2. 不会；长度 <8 且打分不够（0 分 < 2）。
3. s1→s2→s3；`_sort_steps` 按 `depends_on` 拓扑排序。
4. state；存 prompt 会丢序列化/断点续跑/可观测三项能力。
5. `_with_plan` 追加到最后一条 user 消息末尾；模板槽位固定 8 个，多传会被丢弃。
6. 不会；已完成的结果有成本（调过工具、花过 token），且可能带副作用。
7. failed 是关键排查信息（`[!]` 保留）；pending 从没做过，留着让模型困惑。
8. 否则计划永远完不成，模型一直停在"还有步骤没做"。
9. "所有步骤已完成，请基于下面的结果汇总出最终答案，不要再调用新的工具"；少了会被 `max_iterations` 硬截断。
10. 不会；降级为单步计划，用户拿到正常答案。
11. `before_call` 返回 block → `act_step` 记为失败 → 当前步标 failed → 若开启且次数未用尽则触发重排。
12. 沿用原计划继续做（日志"跳过重规划"）。
13. `plan.current()`（第一个 pending 步骤）；`current_task_index` 只是 `_save_plan` 算出的展示值。
14. `after_call` 把结果 token 记进预算，逼近 6000 上限时 reason 会强制收尾。
15. 整个 state（含 `sub_tasks`、`plan_replan_count`）序列化在 PG，靠 `thread_id=session_id` 取回。

</details>

---

## 附录：文件与行号导航

| 想知道 | 看哪 |
|---|---|
| 图结构、节点与边 | `modules/agent/graph.py:63-137` |
| 入口初始 state / 断点续跑 | `graph.py:373-418` |
| prepare 后路由 | `graph.py:326-338` |
| 计划字段回写（plan 节点） | `graph.py:160-172` |
| act 节点回写（含计划推进） | `graph.py:197-214` |
| answer / save / 自动初审 | `graph.py:218-229` / `233-276` / `296-322` |
| 复杂度打分 | `modules/agent/planner.py:262-292` |
| 生成计划 / 输出契约 / 提示词 | `planner.py:299-356`、`69-70`、`313-327` |
| JSON 解析兜底 | `planner.py:443-459` |
| 拓扑排序 | `planner.py:463-485` |
| 重规划 | `planner.py:363-436` |
| 进度文本渲染 / 收敛提示 | `planner.py:189-209` |
| 降级单步计划 | `planner.py:490-494` |
| 判定要不要规划 | `react_agent.py:317-321` |
| prepare 六件事 | `react_agent.py:118-215` |
| 预算预检 | `react_agent.py:461-472` |
| L0 工具路由 | `react_agent.py:474-504` |
| 八槽位 prompt / 计划注入 | `react_agent.py:529-547` |
| LLM 决策 + 两道刹车 | `react_agent.py:549-595` |
| act 六步 + 两条工具日志 | `react_agent.py:621-682` |
| 推进计划 | `react_agent.py:396-416` |
| 工具调度六步流水线 | `modules/tools/tool_registry.py:188-336` |
| Harness 准入三关 | `modules/harness/harness.py:230-276` |
| 沙箱选择 / 调用后处理 | `harness.py:328-340` / `345-365` |
| 配置项 | `config.py:104-108` |
| 状态字段定义 | `modules/agent/state.py:49-57`、`101-105` |
| 测试 | `tests/test_agent_planning.py` |
