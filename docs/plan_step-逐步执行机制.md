# plan_step 是怎么"一步一步执行"的

> 面向：读代码读到这里卡住的人
> 配套：`docs/Planning全流程-阅读文档.md`（全局视角）、`docs/Planning执行闭环补强.md`
> 本文所有输出均为**实测**（真实驱动 `ReActAgent.plan_step` / `act_step`，LLM 与工具用桩替换）
> 行号以当前代码为准（2026-10-03 校准；第九节「逐步沙盘」为 2026-10-04 增补，行号同步校准）

---

## 〇、先建立直觉：后厨比喻（30 秒版）

把整个 Plan-and-Execute 想象成一家小餐馆：

| 代码 | 角色 | 干的事 |
|---|---|---|
| `plan_step` | **店长抄单** | 接到订单（query），抄一张任务清单贴墙上。**只抄单，不炒菜** |
| `_plan_context` | **每道菜开火前，厨师瞄一眼清单** | "现在该做哪道？"——把清单拼进 prompt 提醒模型 |
| `act_step` | **厨师炒菜** | 真的去调工具 |
| `_advance_plan` | **打勾员** | 一道菜出锅，把清单上那道划掉，手指移到下一道 |
| `planner.replan` | **改菜单** | 某道菜原料没了（工具失败），重新安排后面的菜单 |

一句话结论（先破除最大的误解）：

> **`plan_step` 自己一步都不执行。** 它只做一件事：调一次 LLM，把一句话拆成 N 个步骤，
> 写进 `state.sub_tasks`，然后返回。真正的"一步一步"是后面 **reason ↔ act 循环**完成的。

口诀：**生成一次 → 提示 N 次 → 推进 N 次**。

---

## 一、JS 类比：会 JS 的话 5 分钟建立映射

```js
// ① plan_step —— 只发生一次，相当于"往 state 里放一份 todoList"
state.subTasks = await llmSplitIntoSteps(query);     // LLM 拆任务，返回 [{id:'s1',...},...]

// ② _plan_context —— 每一轮 reason 相当于一次 render，把 todoList 渲染成文本
const planText = renderTodoList(state);              // "## 执行计划 ... - [ ] s1: ...（当前步骤）"
prompt = lastUserMessage + planText;                 // _with_plan：拼到最后一条 user 消息末尾

// ③ act_step —— 真正执行：把模型选的工具跑一遍
const result = await registry.executeTool(state.action);

// ④ _advance_plan —— 相当于 reducer：把当前项标记 completed/failed
state.subTasks = markCurrent(state.subTasks, result.ok ? 'done' : 'failed');

// ⑤ 指针 —— 注意！没有维护 currentIndex 变量，每次都重新 find：
const idx = state.subTasks.findIndex(t => t.status === 'pending');
```

关键直觉：**模型每轮都是"无状态"的**——就像 React 组件不记得上次的 render，
你不把清单重新拼进 prompt，它就不知道自己做到第几步了。这就是为什么"提示 N 次"。

---

## 二、全景流程图（一图闭环）

```
 用户提问（复杂问句）
      │
      ▼
 ┌─ prepare 节点 ─────────────────────────────────────────────┐
 │  召回记忆/技能/RAG（逐步沙盘见第九节）                    │
 └──────────────────────────┬─────────────────────────────────┘
                            ▼
 ┌─ 路由 _after_prepare (graph.py:331) ───────────────────────┐
 │  should_plan(state)?  ──复杂──▶ 走 "plan"                   │
 │        │简单                                                 │
 └────────┼─────────────────────────────────┼─────────────────┘
          ▼                                 ▼
 ┌─ reason 节点 ◀──────────┐      ┌─ plan 节点 (graph.py:161) ──┐
 │ ①每轮渲染计划进 prompt   │      │ plan_step(react_agent:339)   │
 │   (react_agent:563-565) │      │   └→ create_plan(planner:304)│
 │ ②问 LLM：调工具还是答题？ │      │      只拆步骤，只此一次 LLM   │
 │   (react_agent:634)     │      └──────────┬──────────────────┘
 └────────┬────────────────┘                 │ sub_tasks 写入 state
          │ need_tool_call=True              │
          ▼                                  │
 ┌─ act 节点 (react_agent:730) ──────────────┘
 │ ③真干活 registry.execute_tool (759)
 │ ④记账 tool_calls_history
 │ ⑤推进计划 _advance_plan (788 调用，定义在 410)
 └────────┬───────────────┘
          └────────── 回到 reason（指针已指向下一步）──────────┐
                            …循环直到模型直接给答案…           │
                             ▼                              │
                      answer 节点收尾 ◀──────────────────────┘
```

三个问题一个答案：**谁拆任务？plan_step（一次）。谁执行？reason↔act 循环（N 次）。谁记账？_advance_plan（每次工具跑完）。**

---

## 三、plan_step 逐行：只点菜，不炒菜

```339:351:modules/agent/react_agent.py
    def plan_step(self, state: AgentState) -> AgentState:
        """生成执行计划。规划不可用时 planner 会返回单步计划，行为等同不规划。"""
        if self.planner is None:
            return state
        plan = self.planner.create_plan(
            state.user_query,
            tools_desc=self._tools_overview(),
            skill_context=state.skill_context,
        )
        self._save_plan(state, plan)
        _logger.info(f"计划已写入 state: {len(state.sub_tasks)} 步 | 思路={state.plan_rationale[:40]}")
        return state
```

| 行 | 干什么 | 通俗解释 |
|---|---|---|
| 341-342 | 没装规划器 → 原样返回 | 开关没开（`AGENT_PLANNING_ENABLED=False`）时，plan 节点空转，行为回到改造前 |
| 343-347 | **唯一一次 LLM 调用** | 把 query + 工具清单（`_tools_overview()`，400 行）+ 技能 SOP 发给 LLM："帮我拆成有序步骤" |
| 348 | 写进 state | 见下方 `_save_plan` |
| 349-350 | 打日志、返回 | **到此结束**——没有循环、没有工具调用、没有执行 |

### `_save_plan` 写进去的四样东西（366-373）

```366:373:modules/agent/react_agent.py
    def _save_plan(self, state: AgentState, plan: "ExecutionPlan") -> None:
        state.sub_tasks = [s.to_dict() for s in plan.steps]      # 步骤列表（dict，因为 LangGraph 只认 dict）
        state.plan_rationale = plan.rationale                    # 总体思路
        state.plan_replan_count = plan.replan_count              # 已重排几次
        state.current_task_index = next(
            (i for i, s in enumerate(plan.steps) if s.is_open), len(plan.steps)
        )                                                        # 指针 = 第一个未闭合步骤的下标
```

**指针不是单独维护的变量**——每次都是"第一个还开着的步骤"重新算出来的
（对照 JS 类比：`findIndex(t => t.status === 'pending')`，不是 `currentIndex++`）。

这个设计有两个好处：

1. 推进时不需要手动 +1（少一处"忘了加"或"加了两次"的 bug）；
2. 重排后新步骤插进来，指针自动落到新步骤上（见第五节的实测序列）。

---

## 四、执行循环三件套（缺一不可）

```
        ┌──────────────── reason 节点 ────────────────┐
        │ 563  plan_text = self._plan_context(state)   │  ← ① 每轮重新渲染进度
        │ 565  rec = self._with_plan(rec, plan_text)   │  ←   拼到 user 消息末尾
        │ 634  state.action = tool_calls[0]            │  ← ② LLM 决定"这一步做什么"
        └───────────────────┬─────────────────────────┘
                            ↓ need_tool_call
        ┌──────────────── act 节点 (730) ─────────────┐
        │ 759  registry.execute_tool(...)              │  ← ③ 真干活（含权限/沙箱/审计）
        │ 788  self._advance_plan(state, tool, result) │  ← ④ 推进计划
        └───────────────────┬─────────────────────────┘
                            ↓
                        回到 reason，指针已指向下一步
```

| 部件 | 频率 | 为什么要这个频率 |
|---|---|---|
| ① 每轮注入计划 | **每轮一次** | 模型无状态，不提醒就忘记自己在第几步（React 不传 props 也会丢状态，同理） |
| ②/③ 执行工具 | 每轮 0 或 1 次 | 由模型的决策决定；直接给答案就不调 |
| ④ 推进计划 | **每次工具调用后** | 全项目只有这里会改 `sub_tasks` 的状态 |

**关键：`_advance_plan` 只在 `act_step` 末尾被调用（788 行）。**
如果模型这一轮直接给答案（不调工具），计划**不会被推进**——计划是引导，不是硬调度。

---

## 五、一步一步执行：完整实测序列

驱动方式：`plan_step` 生成 → 手动模拟模型的决策（写 `state.action`）→ `act_step` 执行 → 打印 `_plan_context`。

> 注：实测输出里 stdout 与日志（stderr）交错，下面按**逻辑顺序**重排。
> 每段输出后加了【解读】，告诉你这一步 state 里发生了什么。

### 步骤 0：判定 + 生成

```
INFO  规划判定: score=4/2 → 规划 | ['含序列词', '含汇总/对比词', '长问句']
INFO  计划已生成: 3 步 | rationale=先取两个季度的事实数据，再对比，最后成文
INFO  计划已写入 state: 3 步 | 思路=先取两个季度的事实数据，再对比，最后成文

sub_tasks: [('s1','pending'), ('s2','pending'), ('s3','pending')]
current_task_index: 0
```

【解读】`score=4/2` 是 `needs_planning` 的复杂度打分：4 分 ≥ 门槛 2 分 → 决定规划。
命中规则打印在后面（含序列词"然后"、汇总词"对比"、长问句）。

此时**注入 prompt 的计划块长这样**（这就是模型第 1 轮看到的）：

```
## 执行计划（共 3 步，已完成 0）
总体思路：先取两个季度的事实数据，再对比，最后成文
当前步骤：s1 —— 请只围绕这一步决定要不要调工具、调哪个工具。
- [ ] s1: 查询 2024 Q4 各区域销售额（当前步骤）
- [ ] s2: 查询 2024 Q3 各区域销售额
- [ ] s3: 对比两季度数据并整理成一页报告
```

### 步骤 1：`data_query(quarter=Q4)` 成功

```
INFO  工具调用: data_query | args={'quarter': 'Q4'}
INFO  计划推进: s1 → 完成 | 进度 1/3
```

```
## 执行计划（共 3 步，已完成 1）
当前步骤：s2 —— 请只围绕这一步决定要不要调工具、调哪个工具。
- [x] s1: 查询 2024 Q4 各区域销售额 → data_query 结果: {...}
- [ ] s2: 查询 2024 Q3 各区域销售额（当前步骤）     ← 指针自动前移
- [ ] s3: 对比两季度数据并整理成一页报告
```

【解读】注意 s1 后面多了 `→ data_query 结果: {...}`——这是 `_advance_plan` 记的
200 字结果摘要（193 行 `brief`），下一轮模型看清单时就知道 s1 已经拿到了什么。

### 步骤 2：`data_query(quarter=Q3)` 失败 → 触发重排

```
WARNING 工具执行失败: data_query | 无权限访问 Q3 销售明细
INFO    已重规划: 保留 1 步完成项，重排 2 步 （第 1 次）
INFO    计划推进: s2 → 失败 | 进度 1/4
```

```
## 执行计划（共 4 步，已完成 1）        ← 步骤数从 3 变 4
当前步骤：r1 —— 请只围绕这一步决定要不要调工具、调哪个工具。
- [x] s1: 查询 2024 Q4 各区域销售额 → ...
- [!] s2: 查询 2024 Q3 各区域销售额 → 无权限访问 Q3 销售明细     ← 失败项保留，作为"此路不通"的证据
- [ ] r1: 从月度汇总表获取 Q3 各区域销售额（当前步骤）          ← 新步骤
- [ ] r2: 对比两季度数据并整理成一页报告
```

【解读】两个细节：
1. **失败项不删除**，保留 `[!]` 状态和错误摘要——下一轮 LLM 看到"此路不通"，
   重排的新步骤才会真的绕开它；
2. 步骤从 3 变 4：新计划把"取 Q3"和"成文"重排成 r1、r2，**已完成的 s1 原样保留**
   （不会重复劳动），指针自动落在第一个未闭合的 r1 上——这就是 `_save_plan`
   "每次重算指针"设计兑现的好处。

### 步骤 3：`data_query(Q3汇总表)` 成功 → 步骤 4：`report_write` 成功

```
INFO 计划推进: r1 → 完成 | 进度 2/4
INFO 计划推进: r2 → 完成 | 进度 3/4
```

```
## 执行计划（共 4 步，已完成 3）
状态：所有步骤已完成，请基于下面的结果汇总出最终答案，不要再调用新的工具。
- [x] s1: ... → done
- [!] s2: ... → failed
- [x] r1: ... → done
- [x] r2: ... → done

current_task_index: 4      ← 等于 len(steps)，没有"下一个"了
plan_replan_count: 1
```

【解读】`current_task_index=4` 说明指针走完了——`_plan_context` 渲染时发现
"没有当前步骤了"，于是改输出那句"请基于结果汇总最终答案"，引导模型收尾。

**注意最后那句"状态：所有步骤已完成"**：`is_complete()` 的判定是"没有 open 步骤"，
`failed` 也算 closed（见《全流程文档》实验 F）。所以"已完成 3"却说"全部完成"——
措辞矛盾但行为正确：模型该收尾了。

---

## 六、`_advance_plan` 逐行（410-429）

```410:429:modules/agent/react_agent.py
    def _advance_plan(self, state: AgentState, tool_name: str, result) -> None:
        """推进计划：成功→done，失败→failed（并按配置重排剩余步骤）。"""
        plan = self._plan_of(state)          # 从 state 的 dict 还原成 ExecutionPlan 对象
        if plan is None:                     # 没有计划（未开启/不复杂）：直接返回，行为等同改造前
            return
        step = plan.current()                # 取当前步骤（不是用 tool_name 去匹配！）
        if step is None:                     # 全部闭合：没有当前步了
            return
        ok = bool(getattr(result, "success", False))
        brief = str(getattr(result, "result", ""))[:200] if ok else str(getattr(result, "error", ""))[:200]
        plan.mark(step.id, STATUS_DONE if ok else STATUS_FAILED, brief)   # 打标 + 记 200 字摘要
        state.task_results.append({...})     # 另存一份流水（API/前端展示用）
        if not ok and self.planner is not None:
            plan = self.planner.replan(plan, observation=state.observation or "",
                                       tools_desc=self._tools_overview())
        self._save_plan(state, plan)         # 写回 state：sub_tasks + rationale + 指针
        _logger.info(f"计划推进: {step.id} → {'完成' if ok else '失败'} | "
                     f"进度 {len(plan.done())}/{len(plan.steps)}")
```

通俗串一遍：打勾员先看墙上有没有清单（`plan is None` 走人）→ 找到手指当前指的
那道菜 → 出锅了打 `[x]`（附 200 字"菜样"），炒糊了打 `[!]`（附 200 字"糊因"）→
炒糊了还要问店长要不要改菜单（`replan`）→ 最后把新清单重新贴回墙上（`_save_plan`）。

### 分支矩阵

| 情况 | 行为 |
|---|---|
| `plan is None` | 直接返回（没开规划时的常规路径） |
| `plan.current() is None` | 直接返回（全部做完后再来一次也不报错） |
| 工具成功 | 当前步 → `done`，记 `result` 前 200 字 |
| 工具失败 | 当前步 → `failed`，记 `error` 前 200 字，**然后重排剩余步骤** |
| 失败但 `planner is None` | 只标 failed，不重排 |
| 失败且重排次数用尽 | `replan()` 原样返回（实测《全流程文档》实验 B） |
| 被限流/预算/权限拦下 | `result.success=False` → 同上，按**失败**处理 → **会消耗一次重排机会** |

---

## 七、两个必须知道的坑

### 坑 1：记账按"当前步"，不按"你调了哪个工具"

`_advance_plan` 用的是 `plan.current()`（418 行），**不是**拿 `tool_name` 去匹配步骤。

通俗版：打勾员不核对"你炒的是不是清单上这道菜"——他只划掉**手指当前指的那道**。
厨师跳步先炒了 s3 的菜？对不起，账记在 s1 头上，s1 被标 done，而真正的 s1 没人做。

JS 类比：不是 `tasks.find(t => t.tool === toolName).done = true`，
而是 `tasks[currentIdx].done = true`。

这是"计划是软引导"的必然代价：代码不校验模型的动作与当前步是否匹配。
好处是**不会因模型跳步而卡死**（标记 + 前移，循环继续）；坏处是**极端情况下账目会错位**。

### 坑 2：被拦下也算"失败"，会白白吃掉重排机会

限流（`rate_limited`）、预算（`budget_exceeded`）、权限（`policy_denied`）返回的都是
`success=False`，走同一条失败分支 → 触发重排。

通俗版：**门卫拦你**（等一下能进）和**菜真的炒糊了**（必须换做法），在打勾员眼里
都是"这一步失败"——都会触发"改菜单"。但门卫拦你等会儿再试就行，不该换菜单。
所以 `MAX_REPLANS` 默认只给 1 次，防止被限流几次就把重排机会烧光。

---

## 八、"下一步做什么"到底谁说了算

| 角色 | 决定什么 | 硬不硬 |
|---|---|---|
| 代码 | 指针 = 第一个未闭合步骤；把进度文本拼进 prompt | 硬 |
| 代码 | 工具跑完后标记这一步 done/failed | 硬 |
| **模型** | 这一轮调不调工具、调哪个工具、给什么参数 | **软**（只是被"当前步骤"引导） |

通俗版：**代码负责记账和指路牌，模型负责干活和做判断**。计划块是给模型的
"当前任务卡"，不是调度指令——就像给新员工发微信"先做 A 再做 B"，
他有可能直接做了 B，你不会因此解雇他，下一轮提醒他就好。

好处：不挑模型（弱模型也能跑，只是可能跑偏）；坏处：不能保证严格按序执行。
真要硬调度，得改成"代码直接指定 `state.action`、模型只填参数"的 Plan-and-Execute
严格版（本项目没这么做——这是"引导式"与"调度式"的架构取舍）。

---

## 九、逐步沙盘：一个例子走完每一步（2026-10-04 增补）

前面讲机制，这节用**一个问题**把整条链路的 12 个站位走一遍。

**贯穿全节的例子**：用户 U007（财务部员工）在网页上问：

> "财务部有多少人？平均工资多少？"

每个站位固定四行格式：

```
【规划】这一步要达成什么
【执行】真实代码路径（文件:行号）
【派发】谁，把什么数据，交给谁
【例子】本例里的真实文本
```

### 9.0 沙盘总览（先记住这张时间表）

| # | 站位 | 规划什么 | 执行入口 | 派发动作（数据形态）|
|---|---|---|---|---|
| 0 | HTTP 入口 | 鉴权、解析身份 | `server.py` /chat | 请求体 → `graph.run()` |
| 1 | prepare | 备料 6 件事 | [graph.py:142](../modules/agent/graph.py#L142) | → state 被填满 |
| 2 | 红绿灯① | 拦截/复杂/简单 | [graph.py:329](../modules/agent/graph.py#L329) | 返回 plan/reason/answer |
| 3 | plan | 拆成有序步骤 | [react_agent.py:339](../modules/agent/react_agent.py#L339) | → `sub_tasks` 列表 |
| 4 | reason 第1轮 | 问模型"这轮干嘛" | [react_agent.py:434](../modules/agent/react_agent.py#L434) | → `state.action` |
| 5 | 红绿灯② | 调工具还是答题 | [graph.py:345](../modules/agent/graph.py#L345) | 返回 act/answer |
| 6 | act 第1轮 | 真执行 data_query | [react_agent.py:836](../modules/agent/react_agent.py#L836) | → observation + 打勾 |
| 7 | reason 第2轮 | 看到人数，接着算平均 | 同 434 | → calculator action |
| 8 | act 第2轮 | 真执行 calculator | 同 729 | → "17500" + 打勾 |
| 9 | reason 第3轮 | 模型看到全 ✅ | 同 434 | → 纯文本，**无** tool_calls |
| 10 | answer | 完整答案 + 摘要 | [graph.py:229](../modules/agent/graph.py#L229) | → final_answer |
| 11 | save | 落库 | [graph.py:244](../modules/agent/graph.py#L244) | → PG + MySQL |

---

### 站位 0：HTTP 入口 —— 请求变成初始 state

【规划】校验 API Key、解析出登录身份（principal），把一句话连同身份交给图。
【执行】`POST /chat` → `AgentGraph.run()`（[graph.py:376](../modules/agent/graph.py#L376)）。
【派发】HTTP 请求体（JSON）→ 初始 state。**此时 state 只有 5 个字段**（[graph.py:396-402](../modules/agent/graph.py#L396-L402)），别的全是空的；另配 `recursion_limit=30`（图跳转步数的天花板，[graph.py:404](../modules/agent/graph.py#L404)）。
【例子】

```python
initial_state = {
    "session_id": "s-001",
    "user_id": "U007",
    "user_query": "财务部有多少人？平均工资多少？",
    "principal": <Principal 财务身份>,   # CLI 调试时为 None
    "prepared": False,
}
```

通俗版：客人进门，前台只登记了"谁、点了什么"，厨房还一口没动。

### 站位 1：prepare 节点 —— 开工前把料备齐

【规划】一次完成 6 件备料：**开预算账 → 注入检测 → 清洗问句 → 匹配技能 → 召回记忆 → 召回知识库**。
【执行】[graph.py:142](../modules/agent/graph.py#L142)（`_prepare_node`）→ `prepare_state()`（[react_agent.py:130](../modules/agent/react_agent.py#L130)）。
【派发】原始 query → 一个被填满的 state，各字段都是后面拼 prompt 的原料。

| 顺序 | 干的事 | 代码行 |
|---|---|---|
| 1 | 开一本新预算账（次数/token/墙钟从 0 算） | [react_agent.py:150](../modules/agent/react_agent.py#L150) |
| 2 | Prompt 注入检测；**命中直接返回 finished**，不进后续流程 | [react_agent.py:153](../modules/agent/react_agent.py#L153) |
| 3 | 清洗：去控制字符、截断、去空白 | [react_agent.py:L164](../modules/agent/react_agent.py#L164) |
| 4 | 匹配技能，拿到 SOP 操作上下文 | [react_agent.py:175](../modules/agent/react_agent.py#L175) |
| 5 | 召回长期/短期记忆（先算 embedding 再向量检索） | [react_agent.py:189](../modules/agent/react_agent.py#L189) |
| 6 | RAG 召回（按身份过滤知识库）+ 标准答案库 | [react_agent.py:205](../modules/agent/react_agent.py#L205)、[217](../modules/agent/react_agent.py#L217) |

【例子】备完后 state 里有了这些文本（后面会原样拼进 prompt）：

```
state.memory_context = "U007 上次问过财务部花名册…"
state.rag_context    = "【知识库参考】员工与薪资表结构…"
state.skill_context  = "（通用企业助手流程）"
```

两个细节：

1. 节点返回时打上 `prepared=True`（[graph.py:155](../modules/agent/graph.py#L155)）——循环再经过这里时**不会重复备料**（重复 embedding 和检索都是钱）。
2. 每一步召回都包了降级：记忆/RAG 服务挂了只记 warning、对应字段置空，对话不中断。

### 站位 2：红绿灯① —— 去 plan、reason 还是 answer

【规划】决定下一站：拦截话术直接收尾、复杂问题先规划、简单问题直接进推理。
【执行】`_after_prepare()`（[graph.py:329](../modules/agent/graph.py#L329)）。
【派发】它只返回一个**字符串**，LangGraph 拿字符串当 key 查边表（[graph.py:103-111](../modules/agent/graph.py#L103-L111)）决定下一个节点。

```python
if state.error or (state.finished and state.final_answer):
    return "answer"     # 注入拦截：话术必须绕过 LLM，否则会被模型输出盖掉
if agent.should_plan(state):
    return "plan"       # 复杂：先拆步骤
return "reason"         # 简单：直接进 ReAct
```

【例子】本例命中汇总词"平均"+ 一句问两件事，复杂度打分过门槛 → 返回 `"plan"`。

### 站位 3：plan 节点 —— 拆步骤（只此一次）

【规划】调一次 LLM，把一句话拆成**有序、带工具提示**的步骤。
【执行】[graph.py:161](../modules/agent/graph.py#L161) → `plan_step()`（[react_agent.py:339](../modules/agent/react_agent.py#L339)）→ `planner.create_plan()`（planner.py:304）。
【派发】PlanResult → `_save_plan()`（[react_agent.py:366](../modules/agent/react_agent.py#L366)）写四样：`sub_tasks` / `plan_rationale` / `plan_replan_count` / 指针。
【例子】本例拆出 2 步：

```
s1: 查询财务部人数与工资总额      （tool_hint=data_query，无依赖）
s2: 计算平均工资                 （tool_hint=calculator，依赖 s1）
指针 current_task_index = 0      （第一个未闭合步骤）
```

再强调一次第三节的结论：**plan 节点到此结束，它一步都不执行。**

### 站位 4：reason 第1轮 —— 拼材料、问模型

【规划】把全部上下文拼成 messages，让模型决定"调不调工具、调哪个、给什么参数"。
【执行】`reason_step()`（[react_agent.py:434](../modules/agent/react_agent.py#L434)），内部 8 个动作按序：

| 序 | 动作 | 行号 |
|---|---|---|
| a | `iteration` 0→1 | [467](../modules/agent/react_agent.py#L467) |
| b | 预算闸门：超时/token 打满 → 强制收尾 | [475-486](../modules/agent/react_agent.py#L475-L486) |
| c | 路由出本轮工具名单 + RBAC 权限过滤 | [503-519](../modules/agent/react_agent.py#L503-L519) |
| d | 拼工具历史文本 | [532](../modules/agent/react_agent.py#L532) |
| e | 8 槽位模板填出 [system, user] 两条消息 | [548](../modules/agent/react_agent.py#L548) |
| f | **计划进度拼到 user 消息末尾** | [563-565](../modules/agent/react_agent.py#L563-L565) |
| g | 调一次 LLM（Function Calling） | [568](../modules/agent/react_agent.py#L568) |
| h | 拿到 tool_calls → 重复/环/无进展检查 → 写 action | [593-634](../modules/agent/react_agent.py#L593-L634) |

【派发】模型返回的 tool_calls[0] → `state.action`（一个 dict，act 节点只认它）。
【例子】这一轮模型看到的拼接区：

```
【执行计划】1.查人数和工资总额 ⬜（当前步骤）  2.算平均工资 ⬜
【上轮观察】（初始状态，暂无工具观察）
【工具历史】（无）
```

模型返回：

```python
tool_calls = [{"name": "data_query", "arguments": {"dept_code": "D_FIN"}}]
state.need_tool_call = True
state.action = {"name": "data_query", "arguments": {"dept_code": "D_FIN"}}
```

### 站位 5：红绿灯② —— 全图唯一的"act/answer"开关

【规划】模型要干活且没出错 → act；否则 → answer 收尾。
【执行】`_should_continue()`（[graph.py:345](../modules/agent/graph.py#L345)）。

```python
if state.finished:                              return "answer"
if state.need_tool_call and not state.error:    return "act"
return "answer"   # 想调工具但出错了：也收尾，不再硬调
```

【例子】`need_tool_call=True`、error 为空 → 返回 `"act"`。

### 站位 6：act 第1轮 —— data_query 真执行

【规划】把模型选的工具真正跑一遍，结果回写，然后给计划打勾。
【执行】`act_step()`（[react_agent.py:836](../modules/agent/react_agent.py#L836)）。
【派发】action → `registry.execute_tool()`（[865](../modules/agent/react_agent.py#L865) → [tool_registry.py:205](../modules/tools/tool_registry.py#L205)）。**注意调用时第 4 个参数**：服务端把私有参数（`_scopes/_dept_code/_user_code/_mask_level`）作为 context 注入，模型在 schema 里根本看不到这些字段，也伪造不了（[tool_registry.py:317-319](../modules/tools/tool_registry.py#L317-L319) 后写入覆盖）。

registry 内部按序过闸：查注册表 → 能力裁决 before_call → 单工具限次 → 同参去重/缓存 → **真执行** → after_call 脱敏裁剪。

【例子】

```python
# 模型给的参数
{"dept_code": "D_FIN"}
# 服务端额外注入（模型看不到）
{"_dept_code": "D_FIN", "_scopes": {"users": "dept", ...}, "_mask_level": 2}

# 工具返回
ToolResult(success=True, result="财务部共25人，工资总额437,500元")
```

执行后三连（[react_agent.py:875-896](../modules/agent/react_agent.py#L875-L896)）：

```
1. 审计落 MySQL
2. tool_calls_history / tool_call_logs 各加 1 条
3. state.observation = "财务部共25人，工资总额437,500元"
```

`_advance_plan()`（[787 调用，410 定义](../modules/agent/react_agent.py#L410)）：s1 打 ✅（附 200 字结果摘要），指针自动前移到 s2。

### 站位 7：reason 第2轮 —— 同样的代码，变化的拼接区

【规划】代码路径与站位 4 **完全相同**；这里重点看模型"凭什么知道下一步"。
【执行】同 [react_agent.py:434](../modules/agent/react_agent.py#L434)。
【派发】这一轮的拼接区三处变化（绿色逻辑：全是上一步喂进来的）：

```
【执行计划】1.查人数和工资总额 ✅ → 结果: 25人，总额437,500
            2.算平均工资 ⬜（当前步骤）
【上轮观察】财务部共25人，工资总额437,500元
【工具历史】data_query({"dept_code":"D_FIN"}) → 25人…
```

【例子】模型据此返回：

```python
tool_calls = [{"name": "calculator", "arguments": {"expression": "437500/25"}}]
```

这就是"模型无状态"的实证：它**不是记得** s1 做完了，而是每轮从拼好的 ✅ 和 observation 里重新读出来的。

### 站位 8：act 第2轮 —— calculator 真执行

【规划】同站位 6：执行 → 记账 → 打勾。
【执行】同 [react_agent.py:836](../modules/agent/react_agent.py#L836)。
【例子】

```python
registry.execute_tool("calculator", {"expression": "437500/25"})
# → ToolResult(success=True, result="17500")

state.observation = "17500"
```

`_advance_plan()`：s2 打 ✅，**两个步骤全部闭合，指针走到头**（`current_task_index=2 = len(steps)`）。

### 站位 9：reason 第3轮 —— 模型决定收尾

【规划】代码不替模型判断"做完没"——即使计划已全 ✅，仍要让模型看一眼材料自己决定。
【执行】同 [react_agent.py:434](../modules/agent/react_agent.py#L434)。
【派发】拼接区现在是：

```
【执行计划】1.查人数和工资总额 ✅  2.算平均工资 ✅
【上轮观察】17500
【工具历史】2 条记录
指针：全部步骤已闭合
```

【例子】**关键区别在模型的返回**：这次没有 tool_calls，只有纯文本。

```python
content = "财务部共 25 人，平均工资为 17,500 元/人。"
tool_calls = None
```

走 else 分支（[react_agent.py:636-637](../modules/agent/react_agent.py#L636-L637)）：

```python
state.need_tool_call = False
state.final_answer = "财务部共 25 人，平均工资为 17,500 元/人。"
```

红绿灯② 随即返回 `"answer"`。**正常出口只有这一个：模型自己不再要工具。**

### 站位 10：answer 节点 —— 出完整答案 + 一句话摘要

【规划】生成（或保留）可发给用户的完整答案，再生成一句话摘要供会话列表展示。
【执行】[graph.py:229](../modules/agent/graph.py#L229) → `generate_answer()`（[react_agent.py:901](../modules/agent/react_agent.py#L901)）、`generate_summary()`（[react_agent.py:923](../modules/agent/react_agent.py#L923)）。
【派发】两个分支：

```python
if state.final_answer and not state.need_tool_call:
    pass          # 站位9 已产出答案：原样保留，不多花一次 LLM
else:
    # 汇总 RAG/观察/推理，再问一次模型补写完整答案
```

【例子】

```
final_answer = "财务部共 25 人，平均工资为 17,500 元/人。"
llm_summary  = "财务部25人，平均工资17,500元"
finished     = True
```

### 站位 11：save 节点 —— 全部落库

【规划】把这一轮的对话、摘要向量、工具流水持久化，并触发自动初审。
【执行】[graph.py:244](../modules/agent/graph.py#L244)。
【派发】四个落点：

| 落点 | 内容 | 行号 |
|---|---|---|
| PG `chat_history` | 原文 + 答案 + 摘要向量 + 工具日志 JSONB | [262](../modules/agent/graph.py#L262) |
| PG 会话状态快照 | state 序列化（**先剔除 principal**，活体对象无法 JSON 化） | [277-283](../modules/agent/graph.py#L277-L283) |
| MySQL `tool_call_log` | 每条工具调用一行，审计原始依据 | [293、296](../modules/agent/graph.py#L296) |
| 人工复核队列 | 命中敏感词/工具失败/低置信/无 RAG 时入队 | [309、331](../modules/agent/graph.py#L331) |

落库完成 → 边 `"save" → END`（[graph.py:123](../modules/agent/graph.py#L123)），终态 dict 返回给前端，网页显示最终答案。

---

### 9.12 "派发任务"的另一层：主管 → 团队成员

上面 12 个站位是**单 Agent** 的循环，它的"派发"其实就是一个 dict：

```python
state.action = {"name": "calculator", "arguments": {"expression": "437500/25"}}
```

项目里还有**团队层**的派发——一个主管把活派给多个 Agent 并行干，入口是独立的 `POST /agent/team/run`（[server.py:713](../modules/api/server.py#L713)）。主流程在 [multiagent/harness.py:371](../modules/multiagent/harness.py#L371)，五步：

| 步 | 干的事 | 代码行 |
|---|---|---|
| 1 | 主管规划：返回 `PlanResult`——要么 `subtasks`，要么 `direct_answer`（二选一，模型校验强制） | [harness.py:430](../modules/multiagent/harness.py#L430)；规划器定义 [agents.py:457](../modules/multiagent/agents.py#L457) |
| 2 | `topo_batches()` Kahn 拓扑分层：同批无依赖可并行，环被剔除 | [harness.py:476](../modules/multiagent/harness.py#L476)；定义 [339](../modules/multiagent/harness.py#L339) |
| 3 | `_run_batch()` 线程池并行执行，`future.result(timeout)` 把单任务超时转成失败 | [harness.py:500](../modules/multiagent/harness.py#L500)；定义 [771、837-845](../modules/multiagent/harness.py#L771) |
| 4 | 每个产出同时进 `result.outputs`（对外账单）和 `ctx.upstream[task_id]`（下游输入） | [harness.py:506-512](../modules/multiagent/harness.py#L506-L512) |
| 5 | 一致性检测/仲裁 → 主管汇总 synthesize → reviewer 终审 | [harness.py:540-641](../modules/multiagent/harness.py#L540-L641) |

默认花名册（[agents.py:574](../modules/multiagent/agents.py#L574)）：

```
supervisor    主管：拆活、汇总
researcher_1  检索：可用 data_query
analyst_1    分析：可用 calculator
reviewer_1   审核：终审打分
```

【例子】本例在团队模式下，主管规划出的 `PlanResult` 长这样：

```json
{
  "thought": "先取事实数据，再做除法，最后成文",
  "subtasks": [
    {"id": "t1", "assignee": "researcher_1",
     "instruction": "查询财务部人数与工资总额", "depends_on": [], "priority": 2},
    {"id": "t2", "assignee": "analyst_1",
     "instruction": "用工资总额除以人数，给出平均工资", "depends_on": ["t1"], "priority": 1}
  ],
  "needs_review": true
}
```

数据怎么流动：t1 跑完，产出进 `ctx.upstream["t1"]` → t2 的 prompt 里出现 `【上游产出 · t1】25人，总额437,500`（拼接逻辑 [agents.py:398-406](../modules/multiagent/agents.py#L398-L406)）→ analyst_1 不用重复查数据，直接算。

**边界（别混淆两种入口）**：

- 普通 `/chat` **不会自动触发**团队派发；团队协作只走 `/agent/team/run`（主管在团队模式内部仍可对寒暄直答，即 `is_direct` 分支，[harness.py:439](../modules/multiagent/harness.py#L439)）。
- 执行者中途"把自己缺的一块"再转给别人，是**委派**（delegation），和主管事前**派活**（dispatch）不是一回事，详见 [`多Agent委派.md`](./多Agent委派.md)；团队全流程见 [`多Agent协作全景.md`](./多Agent协作全景.md)。

### 9.13 一句话收拢

整条链路就是三句话：

> **prepare 备一次料（站位 0-2）→ plan 拆一次单（站位 3）→ reason↔act 循环若干轮，直到模型自己不再要工具（站位 4-9）→ answer/save 收尾（站位 10-11）。**

代码全程只做"记账 + 摆路牌 + 保险丝"；**每一步做什么，都是模型看完拼好的材料自己决定的**。

---

## 十、复现脚本（可直接在项目根跑）

```python
import json, types
from modules.agent.react_agent import ReActAgent
from modules.agent.state import AgentState
from modules.agent.planner import TaskPlanner
from modules.tools.base_tool import ToolResult

QUERY = "帮我统计 2024 年 Q4 各区域销售额，然后和 Q3 对比，最后整理成一页分析报告"
PLAN_V1 = json.dumps({"rationale": "先取两个季度的事实数据，再对比，最后成文", "steps": [
    {"id": "s1", "instruction": "查询 2024 Q4 各区域销售额", "tool_hint": "data_query", "depends_on": []},
    {"id": "s2", "instruction": "查询 2024 Q3 各区域销售额", "tool_hint": "data_query", "depends_on": []},
    {"id": "s3", "instruction": "对比两季度数据并整理成一页报告", "tool_hint": "report_write", "depends_on": ["s1", "s2"]},
]}, ensure_ascii=False)
PLAN_V2 = json.dumps({"rationale": "Q3 明细无权限，改从月度汇总表取 Q3", "steps": [
    {"id": "r1", "instruction": "从月度汇总表获取 Q3 各区域销售额", "tool_hint": "data_query", "depends_on": []},
    {"id": "r2", "instruction": "对比两季度数据并整理成一页报告", "tool_hint": "report_write", "depends_on": ["r1"]},
]}, ensure_ascii=False)

class QueueLLM:
    def __init__(self, replies): self.replies = list(replies)
    def invoke(self, prompt, system_prompt=""): return self.replies.pop(0) if self.replies else "{}"

class FakeRegistry:
    def list_tools(self):
        return [types.SimpleNamespace(name="data_query", description="查询业务数据"),
                types.SimpleNamespace(name="report_write", description="生成报告")]
    def execute_tool(self, name, params=None, **kw):
        params = params or {}
        if name == "data_query" and params.get("quarter") == "Q3":
            return ToolResult(success=False, result="", error="无权限访问 Q3 销售明细")
        return ToolResult(success=True, result=f"{name} 结果: {params}", error="", duration_ms=12)

agent = ReActAgent(llm_client=QueueLLM([]), tool_registry=FakeRegistry(),
                   planner=TaskPlanner(llm=QueueLLM([PLAN_V1, PLAN_V2]), enabled=True,
                                       replan_enabled=True, max_replans=1))
state = AgentState(session_id="s-demo", user_id="U007", user_query=QUERY, max_iterations=10)

print("should_plan:", agent.should_plan(state))       # True
agent.plan_step(state)                                # 只生成
print(agent._plan_context(state))                     # 第 1 轮模型看到的计划块

for name, args in [("data_query", {"quarter": "Q4"}), ("data_query", {"quarter": "Q3"}),
                   ("data_query", {"quarter": "Q3汇总表"}), ("report_write", {"title": "Q4 vs Q3"})]:
    state.action = {"name": name, "arguments": args}
    agent.act_step(state)                             # 执行 + 推进
    print(agent._plan_context(state))                 # 每步之后的进度
```

---

## 十一、五问自测（答得上来就算真懂了）

1. `plan_step` 里有没有循环？→ 没有，只调一次 `create_plan`。
2. 计划靠什么"一步一步"前进？→ `act_step` 末尾的 `_advance_plan`（788 行调用，410 行定义）。
3. 指针是怎么前移的？→ 没有手动 +1，是 `_save_plan` 每次重算"第一个未闭合步骤"（369-372）。
4. 模型这一轮没调工具（直接给答案），计划会推进吗？→ 不会，`_advance_plan` 只在 act 里调。
5. 工具被限流拦下，计划会怎样？→ 按失败处理，标 failed 并消耗一次重排机会。
