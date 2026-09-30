# Planning（计划与执行）：复杂流程怎么做到"先规划再动手"

> 代码：`modules/agent/planner.py`（规划器）+ `graph.py` 的 `plan` 节点 + `react_agent.py` 的注入与推进。
> 开关：`AGENT_PLANNING_ENABLED`（**默认关闭**，关掉时整条链路与改造前逐字一致）。
>
> **想跟着一句话走完整条链路**（入口 → prepare → 判定 → 拆步骤 → 执行 → 推进 → 重规划 → 收敛 → 落库，
> 每步带实测输出）：[`Planning全流程-阅读文档.md`](./Planning全流程-阅读文档.md)。本文是设计速查版。

---

## 一、为什么 ReAct 之外还要 Planning

ReAct 是"走一步看一步"：每轮只看当前状态决定下一步。多步任务上它有三个病灶：

| 病灶 | 表现 | 原 ReAct 的兜底 |
|---|---|---|
| 偏航 | 中间某步结果把话题带跑，最后答非所问 | 无 |
| 打转 | 换个参数反复调同一个工具 | 同参去重 + 单工具限次（事后刹车） |
| 漏步 | 用户要三件事，做完两件就收尾 | 无 |

Plan-and-Execute 的解法：**先花一次 LLM 把"分几步、谁先谁后"定下来**，执行时每一步都带着"我在第几步、前面得到了什么"去问模型。

---

## 二、复杂流程是怎么被规划的（完整链路）

```
prepare（清洗 / 记忆 / RAG / 技能）
   │
   ├─ should_plan? ← 复杂度打分（序列词/汇总词/步骤标记/并列项/长度/SOP）
   │     否 ──────────────────────────▶ reason（原 ReAct，零额外开销）
   │     是 ↓
   │  plan 节点：LLM 拆步骤 → ExecutionPlan（id/instruction/tool_hint/depends_on）
   │            按 depends_on 拓扑排序 → 写进 state.sub_tasks
   ↓
reason ←────────────────┐
   │  prompt 注入"计划进度"（总思路 + 已完成 + 当前步）
   ↓                     │
act（执行工具）           │
   │  推进计划：本步 done/failed，结果入账，指针前移
   │  失败且允许 → replan：只重排剩余步骤（成果保留）
   └─────────────────────┘
   ↓
全部步骤完成 → prompt 提示"请汇总作答" → answer → save
```

图的实际结构（`graph.py`）：

```python
workflow.add_node("plan", self._plan_node)     # 第 6 个节点
add_conditional_edges("prepare", _after_prepare, {"plan": "plan", "reason": "reason", "answer": "answer"})
add_edge("plan", "reason")                     # 规划完必进推理，之后不再回 plan
```

---

## 三、什么时候才规划（判定"复杂流程"）

`TaskPlanner.needs_planning()` 打分，累计 ≥ `AGENT_PLANNING_MIN_SCORE`（默认 2）才规划：

| 信号 | 分 | 例子 |
|---|---|---|
| 序列词（然后/接着/最后/先…再…） | +2 | "查天气，**然后**写首诗" |
| 步骤标记（第一步/1./①） | +2 | "**第一步**查订单，**第二步**退款" |
| 汇总词（对比/汇总/统计/报告/方案） | +1 | "**对比**两年营收" |
| 并列项 ≥ 2 处（；、/、以及） | +1 | "查北京、上海、广州" |
| 长度 ≥ 40 字 | +1 | 长复合需求 |
| 技能 SOP 本身是多步的 | +1 | 命中流程型技能 |

**不满足就是不满足**：`3*7 等于多少`、`你好` 直接走原 ReAct —— 规划是一次额外 LLM 调用，简单问题不该付这个钱。

---

## 四、计划长什么样（是数据，不是一段话）

```python
@dataclass
class PlanStep:
    id: str            # s1 / s2 / r1（重规划后）
    instruction: str   # 这一步做什么
    tool_hint: str     # 建议工具（只是提示，模型仍按工具清单自己决定）
    depends_on: list   # 依赖的步骤 id —— 决定执行顺序
    status: str        # pending / done / failed / skipped
    result: str        # 本步产出摘要（前 200 字）
```

**为什么必须结构化**：步骤要能序列化（随 state 断点续跑）、能标记状态（走到一半改计划）、能观测（日志/审计）。往 prompt 里塞一句"请你先做 A 再做 B"做不到这三件事。

**给模型的输出契约**（精简版 JSON Schema，省 token）：

```json
{"rationale": "string", "steps": [{"id": "string", "instruction": "string",
 "tool_hint": "string", "depends_on": ["string"]}]}
```

生成后按 `depends_on` 做拓扑排序（引用不存在的 id 忽略、成环则剩余按原序兜底），保证**被依赖的步骤排在前面**。

---

## 五、执行阶段：每轮都带着计划问模型

`reason_step` 把计划进度并进 user 消息（模板槽位是固定的 8 个，所以计划段落直接拼在 user 内容末尾，无计划时 messages 一字不动）：

```
## 执行计划（共 2 步，已完成 1）
总体思路：先取事实再创作
当前步骤：s2 —— 请只围绕这一步决定要不要调工具、调哪个工具。
- [x] s1: 查北京天气 → 晴 26℃
- [ ] s2: 写一首关于它的诗（当前步骤）
```

全部做完后文案换成：**"所有步骤已完成，请基于下面的结果汇总出最终答案，不要再调用新的工具。"**
——不提示的话模型会继续找工具调用，表现为"做完了还在调工具"，最后被 `max_iterations` 硬截断。

`act_step` 执行完推进计划：

| 结果 | 动作 |
|---|---|
| 成功 | 本步 `done`，结果摘要入 `task_results`，指针前移 |
| 失败 | 本步 `failed`，失败原因入账；开了重规划就只重排剩余步骤 |
| 参数缺失 / 无注册表 | 不推进（指针不动），由模型追问参数 |

---

## 六、走偏或失败：只重排剩下的

`TaskPlanner.replan()` 的三条原则：

1. **已完成的成果保留**：`done` 步骤原样留下，不推倒重来（重来等于把已花 token 拿到的信息丢掉，还可能把副作用再做一遍）
2. **没达成的才算待重排**：`pending` 标 `skipped`（作废但留痕），`failed` 保留 `failed`（失败本身就是关键信息）
3. **次数有上限**：`AGENT_PLANNING_MAX_REPLANS`（默认 1），用尽就不再重排，让模型按原计划继续

```python
plan.steps = done_steps + failed_steps + new_steps   # 重排后的新计划
plan.replan_count += 1
```

---

## 七、降级表（规划不是必经环节）

| 情况 | 行为 |
|---|---|
| `AGENT_PLANNING_ENABLED=false`（默认） | planner 为 None，`should_plan` 恒 False —— **与改造前完全一致** |
| 装配失败 | 记 warning，按无规划运行 |
| LLM 不可用 / 调用失败 | 返回**单步计划**（等同不规划），ReAct 照跑 |
| 返回非法 JSON / 步骤为空 | 同上，走 `repair_json` 修复，修不好就单步 |
| 重规划次数用尽 | 沿用原计划的剩余步骤 |
| 步骤数超 `max_steps` | 截断到上限 |

一句话：**Planning 是增强项，不是依赖**——任何一环坏了，最坏结果是"不规划"，不是"不可用"。

---

## 八、配置项

| 配置 | 默认 | 作用 |
|---|---|---|
| `AGENT_PLANNING_ENABLED` | false | 总开关 |
| `AGENT_PLANNING_MIN_SCORE` | 2 | 复杂度几分以上才规划 |
| `AGENT_PLANNING_MAX_STEPS` | 6 | 单次计划最多几步 |
| `AGENT_PLANNING_REPLAN_ENABLED` | true | 步骤失败时是否重规划 |
| `AGENT_PLANNING_MAX_REPLANS` | 1 | 一次运行最多重规划几次 |

---

## 九、观测

- 日志：`规划判定: score=3/2 → 规划 | ['含序列词', '含汇总/对比词']`、`计划已生成: 2 步`、`计划推进: s1 → 完成 | 进度 1/2`、`已重规划: 保留 1 步完成项，重排 1 步`
- state 字段：`sub_tasks`（步骤）、`current_task_index`（指针）、`task_results`（结果流水）、`plan_rationale`（思路）、`plan_replan_count`（重规划次数）——都在 `save` 节点随状态快照落 PG
- 工具调用仍走 Harness，审计与预算不受规划影响

---

## 十、和多 Agent 团队里的 Planning 有什么区别

| | 单 Agent Planning（本文） | 多 Agent 团队 Planning（`modules/multiagent`） |
|---|---|---|
| 拆出来的东西 | 同一个 Agent 自己按序执行的步骤 | 派给不同角色 Agent 的子任务（带 assignee） |
| 谁执行 | 一个 ReAct 循环，逐步做 | 多个 Agent 并行/分批做 |
| 产物 | `ExecutionPlan(steps)` | `PlanResult(subtasks)` |
| 重规划触发 | 某一步工具失败 | 某个 Agent 失败 |
| 收敛方式 | 步骤做完 → 提示汇总 | consensus（一致性）→ arbitration（仲裁）→ synthesize |
| 适用 | 单人多步任务 | 需要分工/交叉验证的任务 |

两者可以叠加：主链路规划出步骤，其中某步交给团队去做（当前未接线，接口已具备）。

另见 [`多Agent委派.md`](./多Agent委派.md)：团队内部的**委派**（执行者中途把活转给别人，
真实执行 + 产出回传 + 治理），是与上面两种 planning 不同的第三种"任务流转"方式。

---

## 十一、文件与测试

| 文件 | 职责 |
|---|---|
| `modules/agent/planner.py` | `PlanStep` / `ExecutionPlan` / `TaskPlanner`（判定 + 生成 + 重排） |
| `modules/agent/graph.py` | `plan` 节点、prepare→plan→reason 边、act 后回写计划字段 |
| `modules/agent/react_agent.py` | `should_plan` / `plan_step` / 计划注入 prompt / `_advance_plan` 推进 |
| `modules/agent/state.py` | `sub_tasks` / `current_task_index` / `task_results` / `plan_rationale` / `plan_replan_count` |
| `config.py` | 5 个 `AGENT_PLANNING_*` 配置 |
| `tests/test_agent_planning.py` | 19 个用例：判定/解析/排序/推进/重排/降级/图内端到端 |
