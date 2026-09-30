# plan_step 是怎么"一步一步执行"的

> 面向：读代码读到这里卡住的人
> 配套：`docs/Planning全流程-阅读文档.md`（全局视角）、`docs/Planning执行闭环补强.md`
> 本文所有输出均为**实测**（真实驱动 `ReActAgent.plan_step` / `act_step`，LLM 与工具用桩替换）

---

## 〇、一句话结论（先破除误解）

**`plan_step` 自己一步都不执行。** 它只做一件事：调一次 LLM，把一句话拆成 N 个步骤，写进 `state.sub_tasks`，然后返回。

"一步一步执行"是**后面的 reason ↔ act 循环**完成的，靠三个东西配合：

| 部件 | 在哪 | 干什么 |
|---|---|---|
| `_plan_context` | react_agent.py:364-368 | **每一轮**把"总计划 + 已完成 + 当前步"拼进 prompt，告诉模型该做哪一步 |
| `act_step` | react_agent.py:623-684 | 真的去调工具 |
| `_advance_plan` | react_agent.py:398-418 | 工具跑完后**把这一步标记 done/failed 并让指针前移** |

所以是"**生成一次 → 提示 N 次 → 推进 N 次**"，不是"生成时就把活干完"。

---

## 一、plan_step 逐行（329-338）

```327:338:modules/agent/react_agent.py
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

| 行 | 干什么 | 说明 |
|---|---|---|
| 329-330 | 没装规划器 → 原样返回 | `planner=None` 是关闭态，plan 节点变成空转 |
| 331-335 | **唯一一次 LLM 调用** | 输入：query + 工具清单（`_tools_overview()`，388-394）+ SOP |
| 336 | 写进 state | 见下方 `_save_plan` |
| 337-338 | 打日志、返回 | **结束了**，没有循环、没有执行 |

### `_save_plan` 写进去的四样东西（354-360）

```354:360:modules/agent/react_agent.py
    def _save_plan(self, state: AgentState, plan: "ExecutionPlan") -> None:
        state.sub_tasks = [s.to_dict() for s in plan.steps]      # 步骤列表（dict，因为 LangGraph 只认 dict）
        state.plan_rationale = plan.rationale                    # 总体思路
        state.plan_replan_count = plan.replan_count              # 已重排几次
        state.current_task_index = next(
            (i for i, s in enumerate(plan.steps) if s.is_open), len(plan.steps)
        )                                                        # 指针 = 第一个未闭合步骤的下标
```

**指针不是单独维护的变量**，每次都是"第一个还开着的步骤"重新算出来的。这个设计有两个后果：

1. 不需要在推进时手动 +1（少一处易错代码）；
2. 重排后新步骤插进来，指针自动落到新步骤上（实测 4.2 可见）。

---

## 二、一步一步执行：完整实测序列

驱动方式：`plan_step` 生成 → 手动模拟模型的决策（写 `state.action`）→ `act_step` 执行 → 打印 `_plan_context`。

> 注：实测输出里 stdout 与日志（stderr）交错，下面按**逻辑顺序**重排。

### 步骤 0：判定 + 生成

```
INFO  规划判定: score=4/2 → 规划 | ['含序列词', '含汇总/对比词', '长问句']
INFO  计划已生成: 3 步 | rationale=先取两个季度的事实数据，再对比，最后成文
INFO  计划已写入 state: 3 步 | 思路=先取两个季度的事实数据，再对比，最后成文

sub_tasks: [('s1','pending'), ('s2','pending'), ('s3','pending')]
current_task_index: 0
```

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

**注意最后那句"状态：所有步骤已完成"**：`is_complete()` 的判定是"没有 open 步骤"，`failed` 也算 closed（见《全流程文档》实验 F）。所以"已完成 3"却说"全部完成"——措辞矛盾但行为正确：模型该收尾了。

---

## 三、执行循环的三件套（缺一不可）

```
        ┌──────────────── reason 节点 ────────────────┐
        │ 546  plan_text = self._plan_context(state)   │  ← ① 每轮重新渲染进度
        │ 548  rec = self._with_plan(rec, plan_text)   │  ← 拼到 user 消息末尾
        │ 551  llm.invoke_with_tools(...)              │
        │ 588  state.action = tool_calls[0]            │  ← 模型决定"这一步做什么"
        └───────────────────┬─────────────────────────┘
                            ↓ need_tool_call
        ┌──────────────── act 节点 ───────────────────┐
        │ 652  registry.execute_tool(...)              │  ← ② 真干活（含权限/沙箱/审计）
        │ 671  tool_calls_history.append(...)          │  ← ③ 记账（供下轮与重复检测）
        │ 681  self._advance_plan(state, tool, result) │  ← 推进计划
        └───────────────────┬─────────────────────────┘
                            ↓
                        回到 reason，指针已经指向下一步
```

| 部件 | 频率 | 作用 |
|---|---|---|
| ① 每轮注入计划 | **每轮一次** | 模型是"无状态"的，不提醒它就会忘记自己在第几步 |
| ② 执行工具 | 每轮 0 或 1 次 | 由模型的决策决定 |
| ③ 推进计划 | **每次工具调用后** | 只有这里会改 `sub_tasks` 的状态 |

**关键：`_advance_plan` 只在 `act_step` 末尾被调用（681 行）。** 如果模型这一轮直接给答案（不调工具），计划**不会被推进**，也没有任何步骤被标记完成——计划是引导，不是硬调度。

---

## 四、`_advance_plan` 逐行（398-418）

```398:418:modules/agent/react_agent.py
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

## 五、两个必须知道的坑

### 坑 1：记账按"当前步"，不按"你调了哪个工具"

`_advance_plan` 用的是 `plan.current()`（403 行），**不是**拿 `tool_name` 去匹配步骤。

后果：如果模型跳步（比如当前是 s1，它却调了 s3 该用的工具），执行结果**照样记在 s1 头上**，s1 被标 done，而真正的 s1 没做。

这是"计划是软引导"的必然代价：代码不校验模型的动作与当前步是否匹配。好处是**不会因模型跳步而卡死**（标记 + 前移，循环继续）；坏处是**极端情况下账目会错位**。

### 坑 2：被拦下也算"失败"，会白白吃掉重排机会

限流（`rate_limited`）、预算（`budget_exceeded`）、权限（`policy_denied`）返回的都是 `success=False`，走的是同一条失败分支 → 触发重排。但这类失败**等一下就好了**，不该换路子。所以 `MAX_REPLANS` 默认只给 1 次。

---

## 六、"下一步做什么"到底谁说了算

| 角色 | 决定什么 | 硬不硬 |
|---|---|---|
| 代码 | 指针 = 第一个未闭合步骤；把进度文本拼进 prompt | 硬 |
| 代码 | 工具跑完后标记这一步 done/failed | 硬 |
| **模型** | 这一轮调不调工具、调哪个工具、给什么参数 | **软**（只是被"当前步骤"引导） |

所以完整的一句话是：**代码负责记账和指针，模型负责干活和判断；计划块是给模型的"当前任务卡"，不是调度指令。**

好处：不挑模型（弱模型也能跑，只是可能跑偏）；坏处：不能保证严格按序执行。真要硬调度，得改成"代码直接指定 `state.action`、模型只填参数"的 Plan-and-Execute 严格版（本项目没这么做）。

---

## 七、复现脚本（可直接在项目根跑）

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

## 八、五问自测

1. `plan_step` 里有没有循环？→ 没有，只调一次 `create_plan`。
2. 计划靠什么"一步一步"前进？→ `act_step` 末尾的 `_advance_plan`（681 行）。
3. 指针是怎么前移的？→ 没有手动 +1，是 `_save_plan` 每次重算"第一个未闭合步骤"（358-360）。
4. 模型这一轮没调工具（直接给答案），计划会推进吗？→ 不会，`_advance_plan` 只在 act 里调。
5. 工具被限流拦下，计划会怎样？→ 按失败处理，标 failed 并消耗一次重排机会。
