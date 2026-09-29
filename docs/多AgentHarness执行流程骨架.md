# 多 Agent Harness 执行流程骨架

> 对象：`modules/multiagent/harness.py` 的 `AgentHarness.run()`。
> 定位：**团队运行时**——负责「规划 → 派活 → 收口」，不负责单个工具怎么调。
> 注意别混：工具侧的流程（能力声明 / 调用前校验 / 审批 / 沙箱 / 预算 / 结果过滤）在 `modules/harness/`，见 [`工具Harness.md`](./工具Harness.md)。
>
> 本文按「骨架」组织：先给一张图，再逐段对到行号，最后给出**新增能力该挂在哪**。
> 各能力本身的细节见 [`多Agent协作全景.md`](./多Agent协作全景.md)。

---

## 一句话

**一次 `run()` 就是一条流水线：装配一次、跑十步、收口一次；中间任何环节出问题都是降级或提前 return，绝不向外抛异常。**

---

## 骨架总图

```
构造期（一次）                          运行期（每次 run）                        收口（finally）
─────────────                          ──────────────────                        ──────────────
TeamConfig                            ① 起 trace + 根 span                       cancel_waiting()
  ├─ Tracer / Guard / Limiter / Quota ② 建 RunContext + 委派装配                 委派台账 → result.delegations
  ├─ Orchestrator + OrchestratedBus   ③ 主管规划 plan                            关根 span / 封 trace
  ├─ Consensus / Arbitrator / Replan  ④ 直接答？→ 收工                           span 快照 → result.spans
  └─ register(spec) × N               ⑤ 计划审批（HITL，可选）
                                      ⑥ DAG 分批 → 线程池并行 → 委派/消息
                                      ⑦ 重规划（可选）
                                      ⑧ 一致性 → 仲裁 → 人工裁决（可选）
                                      ⑨ 汇总
                                      ⑩ 终审 → 终稿审批 → 定状态
```

---

## 一、三段生命周期

| 段 | 什么时候 | 建什么 | 为什么这样切 |
|---|---|---|---|
| **构造期** | `AgentHarness(config=...)` | tracer / guard / limiter / quota / orchestrator / obus / consensus / arbitrator / replan / 委派策略 | 全部**无状态**，可跨次运行复用；每轮的清理交给 `orchestrator.reset()` |
| **运行期** | 每次 `run()` | trace、委派台账与执行器、`RunContext` | 委派次数是"这一轮问题"的预算，不该被上一轮吃掉 |
| **收口** | `finally` | 作废等待中的中断请求、写台账、关 span、导快照 | `run` 有 6 个 `return`，任何一条路径都要留痕 |

---

## 二、构造期装配（harness.py:96-176）

| 组件 | 行号 | 作用 | 备注 |
|---|---|---|---|
| `self.gate` | 97 | 人工闸门（HITL） | 默认 `None` = 不引入任何等待 |
| `self.consensus` | 106 | 产出相似度 → 是否打架 | 阈值 `consensus_threshold=0.35` |
| `self.bus` | 107 | 消息总线（存储与检索） | Agent 拿到的是 `obus` 门面，不是它 |
| `self.replan` | 108 | 失败后重拆 | 主管在注册完后回填（172 行） |
| `delegation_policy` | 114 | 委派边界四条 | 台账/执行器按次运行建（115-116） |
| `self.tracer` | 127 | 观测 | `orchestration_trace_enabled` 默认开 |
| `self.guard` | 128 | 权限 / 敏感词 / 长度 | 名单靠 `register()` 同步（202 行） |
| `self.limiter` | 135 | 全局桶 + 每发送者桶 | 单发送者只给全局的 1/4：4 是默认并行度，全员一起发刚好用满，不会"一人吃光" |
| `self.quota` | 144 | 总量 / 单轮 / 单人 / 跳数 | 100 / 30 / 3 / 2 |
| `self.orchestrator` | 150 | 串起 guard→限流→配额→防环→投递→trace | 升级目标指向主管（176 行）：踢皮球永远有收口的人 |
| `self.obus` | 163 | 给 Agent 的门面 | **读**直达总线，**写**必经编排器，不存在绕过路径 |
| `register(spec)` | 183 | 造 Agent 并挂 gate | 第一个主管自动成为默认主管 |

> 装配顺序有个硬依赖：`arbitrator`、`replan.supervisor`、`orchestrator.supervisor` 都要在**注册完之后**才能定（168-176 行）——
> 那时才按角色找得到 reviewer。

---

## 三、运行期十步（harness.py:419-667）

| # | 步 | 行号 | 默认 | 产物 / 副作用 |
|---|---|---|---|---|
| ① | 起 trace + 根 span | 400-405 | 开 | `result.trace_id` |
| ② | 建 `RunContext` + 委派装配 + 清总线 | 411-417 | — | `ctx.bus=obus`、`ctx.gate`；委派必须在派活前装好 |
| ③ | 主管规划 plan | 427-436 | 开 | `result.plan`，trace `plan` |
| ④ | `is_direct` → 直接答收工 | 439-444 | — | 第一个提前 return，不跑任何成员 |
| ⑤ | 计划审批（HITL `BEFORE_PLAN`） | 449-473 | 关 | abort / 未批准 → 提前 return |
| ⑥ | DAG 分批 + 并行执行 | 476-525 | 开 | `result.outputs`、`ctx.upstream`、FINDING 广播 |
| ⑦ | 重规划 | 529-530 | **关** | 补派新任务 |
| ⑧ | 一致性 → 仲裁 → 人工裁决 | 540-601 | 一致性开 / 仲裁开 / HITL 关 | `result.consensus`、`result.arbitration`、`human_final` |
| ⑨ | 汇总 | 603-629 | 开 | `result.final_answer` |
| ⑩ | 终审 → 终稿审批 → 定状态 | 631-667 | 终审开 / HITL 关 | `result.review`、`result.status` |

### 第 ⑥ 步内部（最重要的一段）

```
topo_batches(subtasks)  →  [[t1], [t2,t3], [t4]]        # Kahn 分层，同层可并行
  环怎么办 → 剩下的剔出来，返回 ([[t1]], ["t5"])，不卡死

for batch in batches:
    撞 deadline？        → 全批标记"总超时，未执行"，status=TIMEOUT，break
    步数用完？           → 全批标记"预算耗尽，未执行"，break
    ThreadPoolExecutor(min(max_workers, len(batch)))      # 不多开空转线程
        _one(task):
            任务级审批（BEFORE_TASK，可选）→ 人可改 instruction（改副本，原计划留在 trace 里）
            emit agent_start → span(agent.X) → agent.run(task, ctx)   # 委派在这一层内部发生
            emit agent_end
        future.result(timeout=剩余时间，至少 1 秒)
            超时 → AgentOutput.failed("任务执行超时")
            异常 → AgentOutput.failed(异常名)
    产出记两处：result.outputs（对外账单）+ ctx.upstream（下游输入）
    share_findings → 成功产出截 400 字发 FINDING（也走编排器，不然几轮就把总线灌满）
```

**三本预算/超时账**：

| 账 | 值 | 粒度 | 撞上会怎样 |
|---|---|---|---|
| 整次运行墙钟 | `timeout_seconds=120` | 全局 deadline | 后续批次标记失败，`status=TIMEOUT` |
| 整次运行步数 | `max_total_steps=30` | 累计 `out.steps` | 后续批次不再执行（**不改 status**，状态由最终产出定） |
| 单任务等待 | `max(deadline - now, 1.0)` | 单个 future | 该任务记 failed，同批其他人不受影响 |

---

## 四、分支点一览（判断条件写死在哪）

| 分支 | 条件 | 走向 |
|---|---|---|
| 直接答 | `plan.is_direct` | 直接用 `direct_answer`，跳过全部执行 |
| 计划审批 | `gate.policy.kind_enabled(BEFORE_PLAN)` | abort → FAILED；不 go → FAILED |
| 超时 | `time.time() > deadline` | 全批失败 + TIMEOUT |
| 步数耗尽 | `steps_used >= max_total_steps` | 全批失败，继续走汇总 |
| 重规划 | `replan_enabled and replan.need_replan(outputs)` | 补派一次（最多 `max_replans=1`） |
| 一致性 | `consensus_enabled and len(outputs) >= 2` | 相似度 < 0.35 → 冲突 |
| 仲裁 | 冲突 + `arbitration_enabled` | reviewer 定案，结论并入 `synth_inputs` |
| 人工裁决 | 冲突 + `kind_enabled(ON_DISPUTE)` | `MODIFY` 的人改文本 → `human_final` |
| 汇总 | `human_final` > 有成功产出 > TIMEOUT > 全失败 | 四选一，见 603-629 |
| 终审 | `review_enabled and plan.needs_review` | reviewer 挑毛病（**不改答案**） |
| 终稿审批 | `kind_enabled(BEFORE_FINAL)` | 默认兜底放行；`modify` 用改后文本，`reject` 不发出 |

**汇总的四路优先级**（603-629）：

1. 人改过 → 用人改的，`emit("synthesize","采用人工裁决结论（跳过汇总）")`
2. 有任意成功产出 → 主管汇总全部 + 仲裁结论
3. 撞了 TIMEOUT → 保留 `TIMEOUT` 状态，**不降级成 FAILED**（超时要扩容，Agent 失败要查人，混成一种监控上就一直重复）
4. 全失败 → `FAILED`，提前 return

---

## 五、状态怎么定的

| 状态 | 触发 |
|---|---|
| `SUCCESS` | 有答案，且无成员失败 |
| `PARTIAL` | 有答案，但有成员失败（`error="部分 Agent 失败: [...]`） |
| `FAILED` | 无任何可用产出 / 人工终止 / 未注册主管 / 未捕获异常 |
| `TIMEOUT` | 撞上整次 deadline |

定状态写在第 ⑩ 步（660-664）：`result.status != TIMEOUT` 时才覆盖成 `SUCCESS/PARTIAL`——
**TIMEOUT 一旦定了就不再被覆盖**，这是刻意保留的运维信号。

---

## 六、六条退出路径与 finally

`run()` 有 6 个 `return`：

| # | 路径 | 行号 | status |
|---|---|---|---|
| 1 | 未注册 supervisor | 424 | FAILED |
| 2 | `is_direct` 直接答 | 444 | SUCCESS |
| 3 | 计划审批 abort / 不批准 | 467 / 473 | FAILED |
| 4 | 冲突裁决 abort | 599 | FAILED |
| 5 | 全员失败 | 629 | FAILED |
| 6 | 正常完成 | 667 | SUCCESS / PARTIAL |

第 7 条是异常兜底（669-675）：捕获后记 `FAILED`，不抛给调用方。

**`finally` 三件事**（676-717）：

1. `gate.cancel_waiting()`：运行提前结束时唤醒还在 WAITING 的中断请求——不唤醒的话，等待线程要挂到超时才释放，并行度被白白占住，审批页上还挂着一张永远不会生效的卡片。
2. 委派台账写进 `result.delegations`（还有记录或拒绝时才打汇总日志）。
3. 关根 span → 封 trace → 按 `start_ms` 排序导出 `result.spans`（`span_id` 是随机串，字典序毫无意义；按开始时间排，前端不用再排就能画时间线）。

---

## 七、观测骨架

### span 树（kind 取值）

```
run                                    kind=run
├─ plan                                kind=plan
├─ dispatch.batch_1                    kind=dispatch
│   ├─ agent.researcher_1              kind=agent
│   └─ delegate.analyst_1→researcher_1 kind=delegate
├─ dispatch.batch_2                    kind=dispatch
│   └─ agent.analyst_1                 kind=agent
├─ consensus                           kind=consensus
├─ arbitration                         kind=arbitration
├─ synthesize                          kind=synthesize
└─ review                              kind=review
```

根 span 的状态由最终 status 映射：`TIMEOUT→TIMEOUT`、`FAILED→ERROR`、其余 `OK`。

### trace 事件（12 类）

`plan / dispatch / agent_start / agent_end / synthesize / review / done / error /
consensus / arbitration / hitl / replan / delegate`

委派**放行与拒绝都记**——"没想转"和"想转被拦"在排查时是两件事。

### 日志

启动各一行（Agent 注册、委派启用、编排装配）；每批结束 `第 N 批完成: K 个产出`；
收口 `委派汇总: {...}` 与 `status=... 耗时=...ms`。

---

## 八、新增能力该挂在哪（骨架的价值）

| 想加的东西 | 挂点 | 说明 |
|---|---|---|
| 新的协调环节（如投票） | 第 ⑧ 步之后、第 ⑨ 步之前 | 仿 `consensus`/`arbitration`：开一个 `kind=xxx` 的 span，结论 append 进 `synth_inputs` |
| 新的执行期能力（如委派） | 第 ⑥ 步 `_one()` 内，靠 `ctx` 传递 | 仿委派：策略构造期建、台账运行期建、`finally` 收口 |
| 新的人工介入点 | 用 `gate.ask(kind=...)`，走 `InterruptKind` | 记得在 `finally` 里已被 `cancel_waiting()` 兜住 |
| 新的预算 | 参 `steps_used`，在批次循环开头判 | 撞上要有明确的失败文案，不要让任务静默消失 |
| 新的观测 | `self.tracer.span(...)` + `self._emit(...)` | span 看耗时，event 看发生了什么 |

三条约定：

1. **可选能力默认关闭**，关闭时行为与加入前逐字一致（委派、重规划、HITL 都是这个口径）。
2. **可选能力失败要降级**，不能让主流程为一个增强项陪葬（委派被拒返回 `None` 而非抛异常）。
3. **任何新增的运行期状态都要在 `finally` 里收口**，否则失败的那次恰恰没有数据。

---

## 九、文件与配置

| 文件 | 职责 |
|---|---|
| `modules/multiagent/harness.py` | 运行时本体：装配 / `run()` / `_run_batch()` / `_try_replan()` / `_new_delegation()` / `build_harness()` |
| `modules/multiagent/agents.py` | 五类 Agent：`Supervisor / Researcher / Analyst / Reviewer / Writer`，`BaseAgent.run` 内含委派分支 |
| `modules/multiagent/models.py` | `TeamConfig`、`RunResult`、`AgentOutput`、`PlanResult`、委派契约 |
| `modules/multiagent/coordination.py` | `MessageBus / Consensus / Arbitrator / ReplanController` |
| `modules/multiagent/delegation.py` | 委派策略 / 台账 / 执行器 |
| `modules/orchestration/*` | guard / 限流 / 配额 / 防环 / 投递 / trace / span |
| `modules/hitl/*` | 人工闸门与四类中断点 |

关键配置（`.env` → `TeamConfig`）：`MULTIAGENT_ENABLED`、`MAX_WORKERS=4`、`MAX_TOTAL_STEPS=30`、
`TIMEOUT_SECONDS=120`、`REVIEW_ENABLED=true`、`CONSENSUS_*`、`REPLAN_*`、4 个 `DELEGATION_*`、
`ORCHESTRATION_*`（治理与观测）、`HITL_ENABLED=false`。

---

## 十、验证骨架没被改坏

```bash
python -m pytest tests -q                       # 261 passed
python examples/multiagent_demo.py              # 主线：plan→dispatch→consensus→synthesize→review
python examples/multiagent_delegation_demo.py   # 委派：开/关对比、五道闸、显式 API
python examples/orchestration_demo.py           # 编排：限流/配额/防环/升级
python examples/hitl_coordination_demo.py       # 人工介入四个点
```

看骨架是否完整：任一次运行后 `RunResult.spans` 里应当有 `run / plan / dispatch.batch_1 / agent.*`，
`RunResult.trace` 里应当有 `plan → dispatch → agent_start → agent_end → synthesize → done`。
