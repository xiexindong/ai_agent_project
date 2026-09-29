# 多 Agent 委派（Delegation）

> 代码：`modules/multiagent/delegation.py`（策略 / 台账 / 执行器）+ `agents.py` 的委派分支 + `harness.py` 的装配与 `delegate()` API。
> 开关：`MULTIAGENT_DELEGATION_ENABLED`（**默认关闭**；关闭时零额外调用、行为与改造前一致）。
> 演示：`python examples/multiagent_delegation_demo.py`
>
> **想从头读一遍**（为什么需要委派、跟着代码走一遍、实测输出、自测 10 问）：
> [`多Agent委派-阅读文档.md`](./多Agent委派-阅读文档.md)。本文是速查版。

**脉络位置**：一次团队运行的完整顺序见 [`多Agent协作全景.md`](./多Agent协作全景.md)，
委派是其中的第 ③ 环节（派活之后、一致性检测之前）。文档总索引见 [`README.md`](./README.md)。

---

## 一、什么是委派，和派活有什么区别

| 机制 | 谁发起 / 什么时候 | 层数 | 落在哪 |
|---|---|---|---|
| **派活** dispatch | 主管、规划阶段 | 一层，事前定好 | `PlanResult.subtasks` |
| **委派** delegation | 执行者、执行过程中 | 可递归，有深度上限 | `RunResult.delegations` |
| 转派消息 `DELEGATE` | 只是一条消息 | 不执行、不回传、不记账 | `MessageBus` |

**典型场景**：主管派 `analyst_1`「算 Q4 环比」→ analyst 手上没有 Q3 数据
→ 委派 `researcher_1`「查 Q3/Q4 营收」→ 数据回来 → analyst 算完交差。

没有委派能力时，analyst 只有两条路：**编一个数**（污染答案）或**整单失败**（白跑一轮团队）。

> 注：消息层那条 `MessageKind.DELEGATE` 仍然保留，且指挥权归主管（worker 发会被 guard 拒）。
> 它解决"说一句这块你来做"，委派解决"这块真的被干掉并拿回结果"。两者不冲突，层级不同。

---

## 二、完整链路（一次委派从头到尾 8 步）

```
1. Agent.run 拿到任务 → 委派已启用 且 本 Agent 有 can_delegate 资格
2. 结构化决策 → DelegationDecision（要不要转 / 转给谁 / 转什么 / 为什么）
3. 治理五道闸 → DelegationPolicy.can()：开关 → 资格 → 存在性 → 深度/次数 → 防环
4. 放行 → ledger.open() 开一条 DelegationRecord（先记账，再执行）
5. 执行器造子任务 → 真调受托 Agent 的 run()
6. 受托方自己还能再转（depth + 1），直到深度上限
7. ledger.close() 回填产出/失败原因/耗时
8. 受托产出作为「你委派出去的工作产出」拼进委派方 prompt → 委派方据此出自己的结论
```

派出去的子任务 id 形如 `t1>researcher_1`：trace 里一眼能看出它不是主管派的原任务。

---

## 三、五道闸（任一不过 = 不委派，主流程照常往下走）

| 闸 | 规则 | 拦的是什么 |
|---|---|---|
| ① 开关 | `delegation_enabled` | 关闭时连"要不要转派"这次决策调用都不发生 |
| ② 资格 | `spec.can_delegate`；`delegate_to` 白名单非空时还要在名单里 | 无权转派、"专职二传手" |
| ③ 存在性 | 受托方在花名册、且 `enabled=True` | 转给不存在/已停用的人 |
| ④ 边界 | `depth < max_depth`、总次数 `< max_per_run`、单人 `< max_per_agent` | A→B→C 无限甩锅、小问题被拆成几十次转派 |
| ⑤ 防环 | 受托方 ≠ 自己，且不在委派链上 | A→B→A 互相甩锅直到耗尽预算 |

**被拒也要留痕**：`ledger.denials` 全记下来，trace 里写一条 `delegate` 事件。
排查"这块为什么没人接"时，"根本没想转"和"想转但被拦"是两件完全不同的事。

**被拒为什么返回 None 而不是抛异常**：委派是锦上添花的能力，
转不出去原任务照样要做；抛异常会让整个子任务失败，等于为可选能力赔上主流程。

---

## 四、产物长什么样

```python
DelegationRecord(
    id="dlg_a1b2c3d4", from_agent="analyst_1", to_agent="researcher_1",
    instruction="查 2024 年 Q3 与 Q4 的营收原始数据",
    reason="我手上只有 Q4 数据，缺 Q3 无法算环比",
    parent_task_id="t1", depth=0, chain=["analyst_1"],
    success=True, content="Q3 营收 1100 万，Q4 营收 1200 万", duration_ms=812,
)
```

三个落点：

| 位置 | 内容 |
|---|---|
| `RunResult.delegations` | 本次运行所有转派（API 直接可见） |
| `RunResult.trace` | `kind="delegate"` 事件：放行与拒绝都记 |
| `AgentOutput.delegations` / `delegated_by` | 单个产出的委派痕迹：转出去了什么、自己是受谁委派来的 |

委派成功后还会往总线发一条 FINDING（受 `share_findings` 控制），
让队友看到"这块已经有人做了"，避免同一件事被第二个人重复查一遍。

---

## 五、怎么用

### 1）自动委派（执行者自己判断）

```python
from modules.multiagent import AgentSpec, AgentRole, TeamConfig, AgentHarness

team = TeamConfig(
    agents=[
        AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR),
        AgentSpec(name="analyst_1", role=AgentRole.ANALYST, can_delegate=True),   # 开资格
        AgentSpec(name="researcher_1", role=AgentRole.RESEARCHER),
    ],
    delegation_enabled=True,          # 开开关
    delegation_max_depth=1,           # 最多转一层
)
r = AgentHarness(config=team).run("Q4 营收环比多少？")
print(r.delegations)                  # [analyst_1→researcher_1 ...]
```

### 2）显式委派 API（代码 / 主管 / 重规划后主动转）

```python
rec = h.delegate("analyst_1", "researcher_1", "查 Q3/Q4 营收", query="环比多少")
# 被治理拒绝时返回 None；调用方照常自己处理
```

---

## 六、配置

| 配置 | 默认 | 作用 |
|---|---|---|
| `MULTIAGENT_DELEGATION_ENABLED` | false | 总开关（关闭 = 零开销、行为不变） |
| `MULTIAGENT_DELEGATION_MAX_DEPTH` | 1 | 最多转派几层 |
| `MULTIAGENT_DELEGATION_MAX_PER_RUN` | 6 | 单次运行最多委派几次 |
| `MULTIAGENT_DELEGATION_MAX_PER_AGENT` | 2 | 单个 Agent 最多委派几次 |

`TeamConfig` 上也有同名字段（`delegation_*`），显式传参 > .env > 默认值（与其余参数同一套优先级）。

---

## 七、降级表

| 情况 | 行为 |
|---|---|
| 开关关闭 | 不进委派分支，不多花一次调用 |
| `can_delegate=False` | 不做委派决策 |
| 委派决策解析失败 / 模型不可用 | 按"不委派"处理，Agent 自己干 |
| 决策填一半（漏 to_agent/instruction） | 降级为不委派（`_downgrade_incomplete`），不派空任务 |
| 治理不放行 | 返回 None，写一条拒绝 trace，主流程继续 |
| 受托方执行失败 / 抛异常 | 记 failed，委派方拿到"（委派失败）原因"后自己改路子 |
| 委派产出 | 截断 2000 字进台账，避免把全文塞进结果 |

---

## 八、观测

```
委派已启用: depth<=1 总次数<=6 单人<=2
委派完成: analyst_1→researcher_1 | 成功 | depth=0 | 812ms
委派被拒: analyst_1→analyst_1 | 不能委派给自己
委派汇总: {'total': 1, 'ok': 1, 'failed': 0, 'denied': 1, 'max_depth': 0}
```

---

## 九、文件与测试

| 文件 | 职责 |
|---|---|
| `modules/multiagent/delegation.py` | `DelegationPolicy`（边界）/ `DelegationLedger`（记账+防环）/ `DelegationExecutor`（执行）/ `decide_delegation`（决策） |
| `modules/multiagent/models.py` | `DelegationRecord` / `DelegationDecision`；`AgentSpec.can_delegate`、`AgentOutput.delegations`、`RunResult.delegations`、`TraceKind` 新增 `delegate` |
| `modules/multiagent/agents.py` | `RunContext` 委派三件套；`BaseAgent.run` 的委派分支；委派产出并入 prompt |
| `modules/multiagent/harness.py` | 每次 run 建台账/执行器、`delegate()` 公开 API、trace 事件、结果汇总 |
| `config.py` / `.env.example` | 4 个 `MULTIAGENT_DELEGATION_*` |
| `examples/multiagent_delegation_demo.py` | 五段演示：开关对比 / 全过程 / 五道闸 / 显式 API / 三者区别 |
| `tests/test_delegation.py` | 24 个用例：五道闸、决策降级、执行回传、并入产出、预算按轮重置、关闭时不变 |
