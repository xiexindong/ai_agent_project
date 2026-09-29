# 多 Agent 协作全景

> 一句话：一次提问进到团队之后，先由**主管规划拆活**，再按 DAG **分批派活**，成员执行期间可以**横向对话**、**把活转给别人（委派）**，做完由**一致性检测**查打架、**仲裁**定案、**重规划**补漏，最后**终审 + 汇总**出答案，全程由**中心化编排**治理、**trace + span** 留痕。
>
> 本文是「脉络索引」：把散在各文档里的能力按一次运行的先后顺序串起来。细节各归各文档（见文末索引）。

---

## 一、全景图

```
                          用户提问 query
                               │
        ┌──────────────────────▼──────────────────────┐
        │ ① 规划 plan          主管：拆成 SubTask DAG   │──▶ PlanResult.subtasks
        └──────────────────────┬──────────────────────┘
                               │ 按 depends_on 分批
        ┌──────────────────────▼──────────────────────┐
        │ ② 派活 dispatch      线程池并行跑每批任务      │──▶ AgentOutput[]
        │    ├─ 委派 delegation（可选，默认关）         │──▶ DelegationRecord[]
        │    ├─ 消息总线（横向对话：finding/clarify）   │──▶ MessageBus
        │    └─ 步数预算 / 总超时                      │
        └──────────────────────┬──────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │ ③ 一致性 consensus   产出之间是否打架         │──▶ ConsensusResult
        │ ④ 仲裁 arbitration   打架时让审核者定案       │──▶ AgentOutput(仲裁结论)
        │ ⑤ 重规划 replan      有人失败时重拆（默认关） │──▶ replan_count
        │ ⑥ 人工介入 hitl      敏感动作等人拍板（默认关）│──▶ HITL 待办
        └──────────────────────┬──────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │ ⑦ 终审 review        reviewer 挑毛病          │──▶ result.review
        │ ⑧ 汇总 synthesize    主管合成最终答案         │──▶ result.final_answer
        └──────────────────────┬──────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │ 贯穿全程：中心化编排（限流/配额/防环/校验）   │
        │         观测（trace 事件 + span 树 + 日志）   │
        └─────────────────────────────────────────────┘
```

---

## 二、十个环节逐段说明

### ① 规划（plan）

| 项 | 内容 |
|---|---|
| 做什么 | 主管读花名册 + 问题，输出 `PlanResult`：思考过程 / 子任务列表 / 是否直接答 / 是否需要终审 |
| 输入 → 输出 | 输入 `query="2024 Q4 营收环比增长多少？"`，花名册 `["researcher_1","analyst_1","reviewer_1"]`<br>输出 `subtasks=[{id:"t1", assignee:"researcher_1", instruction:"查 Q3/Q4 营收", depends_on:[]}, {id:"t2", assignee:"analyst_1", instruction:"算环比", depends_on:["t1"]}]` |
| 不规划的分支 | 主管判断"不用拆"时填 `direct_answer`，直接返回，不跑任何成员（`RunStatus.SUCCESS`，`outputs=[]`） |
| 代码 | `agents.py` 的 `SupervisorAgent.plan()`；`harness.py` 第 405 行包在 `plan` span 里 |

### ② 派活（dispatch）

| 项 | 内容 |
|---|---|
| 做什么 | 按 `depends_on` 做拓扑分批，批内用线程池并行（`max_workers=4`），批间串行 |
| 两条预算 | `max_total_steps=30`（整次运行总步数，超了后续批次直接标记失败）、`timeout_seconds=120`（整次运行总超时） |
| 输入 → 输出 | 第 1 批 `[t1]` → `[AgentOutput(agent="researcher_1", success=True, content="Q3 1100 万，Q4 1200 万", duration_ms=812)]` |
| 代码 | `harness.py` 的 `_run_batch()`；每批一个 `dispatch.batch_N` span |

### ③ 委派（delegation）

| 项 | 内容 |
|---|---|
| 做什么 | 执行者中途把"更适合别人的那部分"转出去：真实执行 → 产出回传 → 并入自己的结论 |
| 默认 | **关闭**（要多一次"要不要转派"的 LLM 决策） |
| 五道闸 | 开关 → 资格（`can_delegate`）→ 存在性 → 深度/次数/单人次数 → 防环 |
| 输入 → 输出 | `analyst_1` 缺 Q3 数据 → `DelegationRecord(from_agent="analyst_1", to_agent="researcher_1", instruction="查 Q3/Q4 营收", depth=0, success=True, content="Q3 1100 万，Q4 1200 万", duration_ms=812)` |
| 被拒行为 | 返回 `None`，写一条拒绝 trace，**主流程照常往下走** |
| 代码 | `delegation.py`；`harness.py` 的 `_new_delegation()` / `delegate()`；产出落在 `RunResult.delegations` |
| 详文 | [`多Agent委派.md`](./多Agent委派.md) |

### ④ 消息总线（横向对话）

| 项 | 内容 |
|---|---|
| 做什么 | 成员之间发 `finding`（共享发现）/ `clarify`（请求澄清）/ `objection`（异议）/ `delegate`（转派意向） |
| 三条约束 | 消息只在同一批次内有效、总线有容量上限、异议不触发新的执行轮次（防退化成群聊） |
| 共享产出 | `share_findings=True` 时每批结束把产出广播，队友 prompt 的「同伴消息」里可见，避免重复查同一件事 |
| 代码 | `coordination.py` 的 `MessageBus`；实际挂的是 `orchestration/bus.py` 的 `OrchestratedBus` 门面（读直达、写走治理） |

### ⑤ 一致性检测（consensus）

| 项 | 内容 |
|---|---|
| 做什么 | 比较多个产出的文本相似度，低于阈值判定为"打架" |
| 默认 | 开（只多记一条 trace，不改答案路径） |
| 输入 → 输出 | 两份产出相似度 `0.62` ≥ `consensus_threshold=0.35` → `ConsensusResult(conflict=False, similarity=0.62)`<br>若相似度 `0.12` → `conflict=True`，触发仲裁 |
| 代码 | `coordination.py` 的 `Consensus.detect()` |

### ⑥ 仲裁（arbitration）

| 项 | 内容 |
|---|---|
| 做什么 | 冲突时让审核者读全部产出后定案，结论作为最终答案，跳过主管汇总 |
| 默认 | 开（`arbitration_enabled=True`） |
| 与人工介入的关系 | 开了 HITL 时，仲裁结论会升级成人待办，由真人拍板；没人值守则按超时处理 |
| 代码 | `coordination.py` 的 `Arbitrator.arbitrate()`；结果进 `RunResult.arbitration` |

### ⑦ 重规划（replan）

| 项 | 内容 |
|---|---|
| 做什么 | 有成员失败时，让主管基于已完成的产出重新拆一次，只补失败/未做的部分 |
| 默认 | **关闭**（要多调一次 LLM，是真金白银的成本），开启后最多 `max_replans=1` 次 |
| 输入 → 输出 | `failed_agents=["analyst_1"]` → 主管补出 `[{id:"t2b", assignee:"analyst_1", instruction:"只算 Q4 同比"}]`，`replan_count=1` |
| 代码 | `coordination.py` 的 `ReplanController` |

### ⑧ 人工介入（HITL）

| 项 | 内容 |
|---|---|
| 做什么 | 敏感动作前停下来等人：工具审批、仲裁升级、终审修改 |
| 默认 | **关闭**（没人值守的场合会一直等到超时） |
| 三条路径 | `gate.approve()` 审批工具 / 仲裁升级成人待办 / 终审后人工修改答案（优先级最高） |
| 代码 | `modules/hitl/`；harness 通过 `ctx.gate` 与 `Orchestrator.escalations` 接线 |
| 详文 | [`HITL与多Agent协调.md`](./HITL与多Agent协调.md) |

### ⑨ 终审（review）+ ⑩ 汇总（synthesize）

| 项 | 内容 |
|---|---|
| 终审 | `review_enabled=True` 时由 reviewer 对最终答案挑毛病，结果进 `RunResult.review`（不改答案，只留意见） |
| 汇总 | 主管把所有产出合成一段答案；若人工改过（`human_final` 非空）则直接采用人工版本，跳过汇总 |
| 输入 → 输出 | 合成输入 `2` 份产出 → `final_answer="2024 Q4 营收 1200 万，环比 Q3（1100 万）增长 9.1%"` |

---

## 三、三条「任务流转」路径怎么区分

| 机制 | 发起方 / 时机 | 层数 | 是否真执行 | 落在哪 |
|---|---|---|---|---|
| **派活** dispatch | 主管 / 规划阶段 | 一层，事前定好 | 是 | `PlanResult.subtasks` → `RunResult.outputs` |
| **委派** delegation | 执行者 / 执行过程中 | 可递归，有深度上限 | 是，产出回传并入委派方 | `RunResult.delegations` |
| 转派消息 `DELEGATE` | 谁都能发（worker 发会被 guard 拒） | — | 否，只是一句话 | `MessageBus` |

记忆点：**想让活真的被干掉，用委派；只想说一句"这块你来做"，用消息。**

---

## 四、贯穿全程的两条底座

### 中心化编排（`modules/orchestration/`）

```
发消息 → guard 校验（敏感词/长度/权限）→ 限流（令牌桶 20 QPS / 突发 40）
       → 配额（单运行 100 条 / 单轮 30 条 / 单发送者单轮 3 条）
       → 防环（跳数 ≤ 2、重复 ≤ 2、乒乓 ≤ 2）→ 投递 → 记 span
```

被拒的消息不是丢掉，而是记进 `escalations`（升级）或返回 `RejectCode`，排查时能看到"为什么这条没送到"。

### 观测（三条线）

| 线 | 粒度 | 看什么 |
|---|---|---|
| `RunResult.trace` | 事件流（12 类：`plan/dispatch/agent_start/agent_end/synthesize/review/done/error/consensus/arbitration/hitl/replan/delegate`） | 发生了什么 |
| `RunResult.spans` | span 树（`run / plan / dispatch.batch_N / agent.X / delegate.A→B / consensus / arbitration / synthesize / review`） | 慢在哪一环 |
| 日志 | 每步一行 INFO | 启动参数、汇总数字 |

失败的那次最需要 trace：所以 span 快照与委派台账都写在 `finally` 里，任何退出路径都留痕。

---

## 五、RunResult 各字段由谁填

| 字段 | 填写者 | 环节 |
|---|---|---|
| `plan` | `SupervisorAgent.plan()` | ① |
| `outputs` | 各成员 `Agent.run()` | ② |
| `delegations` | `DelegationLedger.records` | ③ |
| `consensus` | `Consensus.detect()` | ⑤ |
| `arbitration` | `Arbitrator.arbitrate()` | ⑥ |
| `replan_count` | `ReplanController` | ⑦ |
| `review` | `ReviewerAgent.review()` | ⑨ |
| `final_answer` | 主管汇总 / 人工 / 仲裁 | ⑩ |
| `trace` / `spans` / `trace_id` | `Tracer` | 全程 |
| `status` / `error` / `duration_ms` | `AgentHarness.run()` 的 `finally` | 收口 |

一次完整运行长这样（示意值）：

```python
RunResult(
    run_id="run_3f2a1b7c", status="success",
    plan=PlanResult(subtasks=[t1, t2]),          # 2 个子任务、2 批
    outputs=[AgentOutput(agent="researcher_1"), AgentOutput(agent="analyst_1")],
    delegations=[DelegationRecord(label="analyst_1→researcher_1", success=True)],
    consensus=ConsensusResult(conflict=False, similarity=0.62),
    arbitration=None, replan_count=0,
    final_answer="2024 Q4 营收 1200 万，环比 Q3（1100 万）增长 9.1%",
    trace=[...12 条事件], spans=[...7 个 span], duration_ms=2410,
)
```

---

## 六、怎么接进来

| 入口 | 方式 |
|---|---|
| HTTP | `POST /agent/team/run`，`response_model=RunResult`（`delegations` 自动带出）；还可用 `/agent/eval`、`/agent/backends` |
| 代码 | `h = AgentHarness(config=TeamConfig(...)); r = h.run(query)` |
| 命令行 | `python main.py serve` 起服务；多 Agent 由服务装配，CLI `chat` 仍是单 Agent |
| 配置 | `.env` 里 `MULTIAGENT_*`（含 4 个 `MULTIAGENT_DELEGATION_*`）；优先级：显式传参 > `.env` > 默认值 |

后端切换：`MULTIAGENT_BACKEND=native`（自研 Harness，默认）/ `crewai`（角色扮演编排，需另配 litellm 模型）。

---

## 七、相关文档

| 文档 | 内容 |
|---|---|
| [`多AgentHarness执行流程骨架.md`](./多AgentHarness执行流程骨架.md) | **骨架**：`run()` 的三段生命周期、十步流水线、分支点与退出路径、新增能力挂哪 |
| [`多Agent委派.md`](./多Agent委派.md) | 委派：8 步链路、五道闸、降级表、API |
| [`HITL与多Agent协调.md`](./HITL与多Agent协调.md) | 人工介入三条路径、消息类型、与委派的分工 |
| [`Planning与执行.md`](./Planning与执行.md) | 主链路 Planning、ReAct、反思重试，与团队内 planning 的区别 |
| [`Harness执行流程.md`](./Harness执行流程.md) | 多 Agent 运行时的执行流程细节 |
| [`工具Harness.md`](./工具Harness.md) | 工具侧：能力声明、调用前校验、审批、沙箱、预算、结果过滤 |
| [`评测与性能基线.md`](./评测与性能基线.md) | 多 Agent 评测（`POST /agent/eval`）与基线数字 |

演示脚本：

```bash
python examples/multiagent_demo.py               # 团队协作主线
python examples/multiagent_delegation_demo.py    # 委派五段演示
python examples/hitl_coordination_demo.py        # 人工介入与成员对话
python examples/orchestration_demo.py            # 中心化编排治理
```
