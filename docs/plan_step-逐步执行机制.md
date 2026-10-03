# plan_step 是怎么"一步一步执行"的

> 面向：读代码读到这里卡住的人
> 配套：`docs/Planning全流程-阅读文档.md`（全局视角）、`docs/Planning执行闭环补强.md`
> 本文所有输出均为**实测**（真实驱动 `ReActAgent.plan_step` / `act_step`，LLM 与工具用桩替换）
> 行号以当前代码为准（2026-10-03 校准）

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
 │  召回记忆/技能/RAG（你已看过，本文略）                        │
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

## 九、复现脚本（可直接在项目根跑）

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

## 十、五问自测（答得上来就算真懂了）

1. `plan_step` 里有没有循环？→ 没有，只调一次 `create_plan`。
2. 计划靠什么"一步一步"前进？→ `act_step` 末尾的 `_advance_plan`（788 行调用，410 行定义）。
3. 指针是怎么前移的？→ 没有手动 +1，是 `_save_plan` 每次重算"第一个未闭合步骤"（369-372）。
4. 模型这一轮没调工具（直接给答案），计划会推进吗？→ 不会，`_advance_plan` 只在 act 里调。
5. 工具被限流拦下，计划会怎样？→ 按失败处理，标 failed 并消耗一次重排机会。
