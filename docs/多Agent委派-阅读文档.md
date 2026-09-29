# 多 Agent 委派：从零读懂（阅读文档）

> 这是一篇**按阅读顺序组织**的长文：从头读到尾就能理解委派为什么存在、怎么实现、怎么验证。
> 需要快速查字段和开关时，用速查文档 [`多Agent委派.md`](./多Agent委派.md)；
> 需要看它在整条团队链路里的位置时，看 [`多Agent协作全景.md`](./多Agent协作全景.md)。

---

## 怎么读这篇

| 路线 | 时间 | 读哪些 |
|---|---|---|
| 速读 | 10 分钟 | 第 1、2、3 章 + 第 10 章的实测输出 |
| 跟读代码 | 30 分钟 | 全文，配合附录 A 的行号导航打开源码 |
| 动手 | 15 分钟 | 第 10 章：跑 demo + 改 4 个配置试边界 |

前置：知道团队由主管规划、成员执行即可（见 [`多Agent协作全景.md`](./多Agent协作全景.md)）。

---

## 第 1 章 从一次失败开始

主管派给 `analyst_1` 一个任务：**「算 2024 Q4 营收环比」**。
但 `analyst_1` 手上只有 Q4 数据，Q3 数据在 `researcher_1` 那里。

没有委派能力时，它只有两条路：

| 选项 | 后果 |
|---|---|
| 凭 Q4 硬答 | 编一个数或者答非所问，答案被污染，而且**看不出来是编的** |
| 整单失败 | 前面几步全白跑，重规划还要再花一次 LLM 调用 |

演示脚本里这两条路的真实输出：

```
委派开关=False:
  委派次数     : 0
   analyst_1 结论: Q3 营收 1100 万，Q4 营收 1200 万
   最终答案     : Q3 营收 1100 万，Q4 营收 1200 万

委派开关=True:
  委派次数     : 1
   analyst_1 结论: Q3 营收 1100 万，Q4 营收 1200 万，环比增长 9.1%
   最终答案     : Q3 营收 1100 万，Q4 营收 1200 万
```

**注意看 `analyst_1 结论` 那一行**：开关打开后多出了「环比增长 9.1%」——
这才是委派的直接收益：**执行者拿到了它原本拿不到的数据，所以算得出原本算不出的结论**。

> 为什么两行的「最终答案」一样？
> 演示用的是固定假答案的假 LLM，主管汇总这一环不受委派影响。
> 真实链路里，主管汇总的输入就是这批产出，analyst 的产出变好了，最终答案才会跟着变。

---

## 第 2 章 三个词先分清

| 机制 | 谁发起 / 什么时候 | 层数 | 是否真执行 | 落在哪 |
|---|---|---|---|---|
| **派活** dispatch | 主管 / 规划阶段 | 一层，事前定好 | 是 | `PlanResult.subtasks` |
| **委派** delegation | 执行者 / 执行过程中 | 可递归，有深度上限 | 是，产出回传 | `RunResult.delegations` |
| 转派消息 `DELEGATE` | 谁都能发（worker 发会被 guard 拒） | — | 否，只是一句话 | `MessageBus` |

**记忆点**：想让活真的被干掉，用委派；只想说一句「这块你来做」，用消息。

两者的差别不是"谁更先进"，而是**层级不同**：消息是沟通，委派是一次带治理、带产出、能进台账的真实调用。

---

## 第 3 章 一次委派的 8 步

```
1. Agent.run 拿到任务 → 委派已启用 且 本 Agent 有 can_delegate 资格
2. 结构化决策  → DelegationDecision（要不要转 / 转给谁 / 转什么 / 为什么）
3. 治理五道闸  → DelegationPolicy.can()：开关 → 资格 → 存在性 → 边界 → 防环
4. 放行        → ledger.open() 先开一条 DelegationRecord（先记账，再执行）
5. 执行        → 造子任务，真调受托 Agent 的 run()
6. 递归        → 受托方自己还能再转（depth + 1），直到深度上限
7. 收账        → ledger.close() 回填产出 / 失败原因 / 耗时
8. 回传        → 受托产出拼进委派方的 prompt，委派方据此出自己的结论
```

行号导航（打开源码对着看）：

| 步 | 文件 | 位置 |
|---|---|---|
| 1-2 | `modules/multiagent/agents.py` | `BaseAgent.run` 委派分支，303-322 行 |
| 2 | `modules/multiagent/delegation.py` | `decide_delegation()`，308 行 |
| 3 | `modules/multiagent/delegation.py` | `DelegationLedger.can()`，115-160 行 |
| 4-7 | `modules/multiagent/delegation.py` | `DelegationExecutor.delegate()`，253-303 行 |
| 8 | `modules/multiagent/agents.py` | `_delegation_text()` 361 行、prompt 拼接 394-395 行 |
| 装配 | `modules/multiagent/harness.py` | `_new_delegation()` 233 行、公开 `delegate()` 314 行 |
| 收口 | `modules/multiagent/harness.py` | `finally` 里写 `result.delegations`，690-693 行 |

---

## 第 4 章 读「决策」：模型怎么决定转不转

一句话：`decide_delegation()` 是多花一次 LLM 调用，问模型「这块要不要转给别人」，
并且只要模型填了一半就自动降级成「不转」。

### 代码（agents.py:303-322）

```python
delegated: list = []
dex = getattr(ctx, "delegation", None)
if dex is not None and getattr(dex, "enabled", False) \
        and self.spec.can_delegate and task is not None:
    decision = decide_delegation(
        self.llm, instruction, dex.candidates(self.name), self.system_prompt
    )
    if decision is not None and decision.should_delegate:
        rec = dex.delegate(
            from_agent=self.name, to_agent=decision.to_agent,
            instruction=decision.instruction, parent_task_id=task.id,
            depth=getattr(ctx, "delegation_depth", 0),
            reason=decision.reason,
            chain=list(getattr(ctx, "delegation_chain", []) or []),
        )
        if rec is not None:
            delegated.append(rec)
```

### 大白话拆解

| 部分 | 大白话 |
|---|---|
| `dex is not None` | 本次运行装配了委派执行器吗 |
| `dex.enabled` | 总开关开了吗（第一位判断：关着就一次调用都不多花） |
| `spec.can_delegate` | 这个 Agent 有转派资格吗 |
| `task is not None` | 有具体任务吗（没有 task = 主管直接问答，谈不上转派） |
| `dex.candidates(self.name)` | 候选名单：排除自己、主管、停用的人 |
| `decision` 为 None | 模型没答出来 / 解析失败 → 当作"不转" |
| `rec is not None` | 治理放行了才有记录；被拒则 `delegated` 保持为空 |

### 输入 → 输出

```python
# 输入
instruction = "算 2024 Q4 营收环比"
candidates  = [AgentSpec(name="researcher_1", role=RESEARCHER)]

# 输出（模型填的结构化决策）
DelegationDecision(
    should_delegate=True,
    to_agent="researcher_1",
    instruction="查 2024 年 Q3 与 Q4 的营收原始数据",
    reason="我手上只有 Q4 数据，缺 Q3 无法算环比",
)
```

### 为什么要这么写

| 做法 | 原因（不这么写会怎样） |
|---|---|
| 开关判断放第一位 | 放在后面就会先多花一次 LLM 调用再发现不能转派——关闭状态就不是"零开销"了 |
| 用结构化输出而不是让模型自由发挥 | 自由文本要靠正则捞名字，捞不到就只能放弃；结构化字段漏填还能**降级** |
| 漏填降级为不转（`_downgrade_incomplete`） | 不降级就会派出一个 `to_agent=""` 的空任务，执行阶段才炸 |
| 候选名单排除主管 | 转回主管 = 甩回给派活的人，形成 主管→A→主管 的空转；真要重拆走 `replan` |

---

## 第 5 章 读「治理」：五道闸

一句话：`can()` 是唯一入口，任何一次委派（自动或显式）都要过它；不过就返回 `(False, 中文原因)`。

### 代码（delegation.py:132-160）

```python
if not p.enabled:              return deny("委派未启用")
if delegator == delegatee:     return deny("不能委派给自己")
if delegatee in chain:         return deny(f"委派成环（链上已有 {delegatee}）")
if depth >= p.max_depth:       return deny(f"已达委派深度上限 {p.max_depth}")
if len(self.records) >= p.max_per_run:  return deny(f"已达单次运行委派上限 {p.max_per_run}")
if self.count(delegator) >= p.max_per_agent: return deny(f"{delegator} 已达单人委派上限 {p.max_per_agent}")
# 后面还有：can_delegate 资格、delegate_to 白名单、受托方在册且未启用检查
```

### 每道闸拦的是什么真实故障

| 闸 | 拦的故障 |
|---|---|
| 开关 | 关闭状态仍偷偷多调一次模型 |
| 转给自己 | A→A 自我循环 |
| 成环 | A→B→A 互相甩锅，日志却全是"正常执行"，直到耗光步数预算 |
| 深度上限 | A→B→C→… 无限外包，最后没人知道活是谁干的 |
| 总次数 / 单人次数 | 一个小问题被拆成几十次转派；或出现"专职二传手" |
| 存在性 / 停用 | 转给一个不存在或已停用的人 |

### 输入 → 输出对照

```python
ledger.can("analyst_1", "researcher_1", depth=0, chain=["analyst_1"])
# → (True, "放行")

ledger.can("analyst_1", "analyst_1", depth=0, chain=["analyst_1"])
# → (False, "不能委派给自己")

ledger.can("researcher_1", "analyst_1", depth=1, chain=["analyst_1"])
# → (False, "委派成环（链上已有 analyst_1）")      # max_depth=1 时也会命中深度闸
```

### 为什么返回中文原因而不是 bool

排查「这块为什么没人接」时，**根本没想转** 和 **想转但被拦** 是两件完全不同的事。
只留一个 `False`，事后只能靠猜。所以每道闸都写进 `ledger.denials` 并记一条日志。

### 实测输出（demo 第 3 段，逐条真实打印）

```
  未启用                → 拒绝 | 委派未启用
  转给自己               → 拒绝 | 不能委派给自己
  受托方停用              → 拒绝 | 受托方 intern 已停用
  受托方不存在             → 拒绝 | 受托方 ghost 不在花名册
  成环（回传给上游）          → 拒绝 | 委派成环（链上已有 analyst_1）
  超过深度上限             → 拒绝 | 已达委派深度上限 1
  超出单人上限             → 拒绝 | analyst_1 已达单人委派上限 1
```

---

## 第 6 章 读「执行」：delegate() 干了六件事

一句话：先记账 → 造子任务 → 真调受托方 → 异常兜底 → 回填 → 通知，全程不抛异常给主流程。

### 代码（delegation.py:263-297，节选）

```python
ok, why = self.ledger.can(from_agent, to_agent, depth=depth, chain=chain)
if not ok:
    if self.on_event is not None:
        self.on_event(None, why)          # 拒绝也要进 trace
    return None                            # 返回 None，不是抛异常

rec = DelegationRecord(from_agent=..., to_agent=..., depth=depth, chain=chain + [from_agent])
self.ledger.open(rec)                      # 先记账
task = SubTask(id=f"{parent_task_id or 'task'}>{to_agent}", ...)
started = time.time()
try:
    out = self._run_fn(to_agent, task, {"extra": {
        "delegated_by": from_agent,
        "delegation_depth": depth + 1,     # 受托方再往下转就是第 depth+1 层
        "delegation_chain": rec.chain,
    }, "ctx": ctx})
except Exception as e:
    out = AgentOutput.failed(to_agent, task.id, f"{type(e).__name__}: {e}")
self.ledger.close(rec, out, started)
```

### 六件事

| # | 做什么 | 为什么 |
|---|---|---|
| 1 | 过闸，不过就返回 `None` | 委派是锦上添花，转不出去原任务照样要做；抛异常等于为可选能力赔上主流程 |
| 2 | `ledger.open()` 先记账 | 执行中途崩了也留得下"这次转派发生过"，否则台账与 trace 全丢 |
| 3 | 造子任务，id 带委派痕迹 `t2>researcher_1` | trace 里一眼看出它不是主管派的原任务 |
| 4 | 真调受托方 `run()`，`depth + 1` | 递归委派靠这个 +1 计数，配合深度闸收敛 |
| 5 | `try/except` 兜底成失败产出 | 受托方炸了不能连累委派方，委派方拿到"（委派失败）原因"后自己改路子 |
| 6 | `ledger.close()` 回填产出/原因/耗时 | 台账才是给人和 API 看的最终账 |

### 输入 → 输出

```python
# 输入
h.delegate("analyst_1", "researcher_1", "查 2024 年 Q3 与 Q4 的营收原始数据")

# 输出（实测）
委派 analyst_1→researcher_1: 查 2024 年 Q3 与 Q4 的营收原始数据
    理由   : 我手上只有 Q4 数据，缺 Q3 无法算环比
    深度   : 0 | 链: analyst_1→researcher_1
    结果   : 成功 | Q3 营收 1100 万，Q4 营收 1200 万
```

受托方不存在时：

```python
h.delegate("analyst_1", "ghost", "...")
# → None，日志：委派被拒: analyst_1→ghost | 受托方 ghost 不在花名册
```

### 简单总结

1. 过闸不过 → `None`，**不是异常**
2. 先 `open()` 记账，再执行（崩了也有记录）
3. `depth + 1` 递归，深度闸收敛
4. 产出截断 **2000 字**进台账（`close()` 第 192 行），不把全文塞进结果

---

## 第 7 章 读「回传」：产出怎么变成委派方的输入

一句话：委派产出不是直接当答案，而是**作为一段上下文拼进委派方的 prompt**，由委派方自己出结论。

### 代码（agents.py:324-325、394-395）

```python
delegation_text = self._delegation_text(delegated)
prompt = self._build_prompt(instruction, upstream_text, vision_text, peer_text, delegation_text)
# ...
if delegation_text:
    parts.append(f"## 你委派出去的工作产出\n{delegation_text}")
```

### 输入 → 输出

```python
# 拼接前
instruction     = "算 2024 Q4 营收环比"
delegation_text = "【委派给 researcher_1】查 2024 年 Q3 与 Q4 的营收原始数据\n"
                  "产出：Q3 营收 1100 万，Q4 营收 1200 万"

# 拼进 prompt 后，模型看到的是
## 你的任务
算 2024 Q4 营收环比

## 你委派出去的工作产出
【委派给 researcher_1】... 产出：Q3 营收 1100 万，Q4 营收 1200 万

# 委派方的产出（实测）
"Q3 营收 1100 万，Q4 营收 1200 万，环比增长 9.1%"
```

### 为什么是"并入 prompt"而不是"替换答案"

委派方才是这个任务的负责人：它知道整体的口径（要不要剔除某笔收入、保留几位小数）。
受托方给的只是它那块的数据。**让负责人出结论**，
比让一个不了解全局的执行者直接顶替答案更安全，也更容易排查"这个数是谁算的"。

---

## 第 8 章 读「降级」：为什么处处都是降级

委派是**可选能力**，所以整条链路的设计原则只有一条：
**任何环节出问题，都退回"不委派"，绝不让主流程失败。**

| 情况 | 行为 |
|---|---|
| 开关关闭 | 不进委派分支，不多花一次调用 |
| `can_delegate=False` | 不做委派决策 |
| 决策解析失败 / 模型不可用 | 按"不委派"处理 |
| 决策填一半（漏 `to_agent`/`instruction`） | 降级为不委派，不派空任务 |
| 治理不放行 | 返回 `None`，写拒绝 trace，主流程继续 |
| 受托方执行失败 / 抛异常 | 记 failed，委派方拿到原因后自己改路子 |
| 委派产出过长 | 截断 2000 字进台账 |

> 反面假设：如果被拒时抛异常，一个"想转没转成"就会让整个子任务失败——
> 用一个可选能力的失败，赔上一条主链路，这笔账怎么算都亏。

---

## 第 9 章 读「观测」：怎么确认它真的发生了

三类留痕，各回答一个问题：

| 留痕 | 位置 | 回答什么 |
|---|---|---|
| 台账 | `RunResult.delegations`（API 直接可见） | 转了几次、谁转给谁、成没成 |
| 事件 | `RunResult.trace` 里 `kind="delegate"` | 放行和**拒绝**都记（拒绝也要留痕） |
| 链路 | `RunResult.spans` 里 `kind="delegate"` 的 span `delegate.A→B` | 时间线上这一环花了多久 |

单条产出的痕迹在 `AgentOutput.delegations`（我转出去的活）与 `AgentOutput.delegated_by`（我是受谁委派来的）。

### 实测日志（去掉时间戳）

```
委派能力已启用: depth<=1 总次数<=6 单人<=2
委派完成: analyst_1→researcher_1 | 成功 | depth=0 | 0ms
[analyst_1] 已把部分工作委派给 researcher_1: 查 2024 年 Q3 与 Q4 的营收原始数据
委派汇总: {'total': 1, 'ok': 1, 'failed': 0, 'denied': 0, 'max_depth': 0}
```

最后一行是 `ledger.summary()`，写在 `finally` 里——**失败的那次最需要看"活被转给了谁"**，
所以任何退出路径都要留下这行。

---

## 第 10 章 动手验证

### 1）跑一遍演示

```bash
python examples/multiagent_delegation_demo.py
```

五段：开关对比 → 全过程 → 五道闸 → 显式 API → 三者区别。
五道闸那段会把 7 种拒绝原因逐条打出来（第 5 章的实测输出就来自这里）。

### 2）代码里开起来

```python
team = TeamConfig(
    agents=[
        AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR),
        AgentSpec(name="analyst_1", role=AgentRole.ANALYST, can_delegate=True),
        AgentSpec(name="researcher_1", role=AgentRole.RESEARCHER),
    ],
    delegation_enabled=True,
    delegation_max_depth=1,
)
r = AgentHarness(config=team).run("Q4 营收环比多少？")
print(r.delegations)
```

### 3）改配置试边界

| 想看什么 | 怎么改 |
|---|---|
| 深度闸生效 | `delegation_max_depth=0`（只能主管派，谁都不能转） |
| 单人闸生效 | `delegation_max_per_agent=1`，让同一个 Agent 转第二次 |
| 成环 | 显式 `h.delegate("researcher_1", "analyst_1", ...)`，`chain` 里带上 `analyst_1` |
| 关闭时零开销 | `MULTIAGENT_DELEGATION_ENABLED=false`，对比两次运行的 LLM 调用次数 |

### 4）跑测试

```bash
python -m pytest tests/test_delegation.py -q     # 25 passed
```

覆盖：五道闸、决策降级、执行回传、产出并入、预算按轮重置、关闭时行为不变、span 树里有委派。

---

## 第 11 章 读完自测

能回答这 10 个问题，说明读懂了：

1. 派活、委派、转派消息，哪两个会真的执行？
2. 开关关闭时，委派会多花几次 LLM 调用？
3. 模型说"要转"但没填 `to_agent`，会发生什么？
4. 五道闸分别拦什么？说出其中三道。
5. 委派被拒时返回什么？为什么不抛异常？
6. 为什么先 `open()` 记账再执行？
7. 子任务 id 为什么写成 `t2>researcher_1`？
8. 受托方的产出是当了最终答案，还是被并入了委派方的 prompt？
9. 委派产出进台账时被截到多少字？
10. 委派的三类留痕分别在哪、各回答什么问题？

---

## 附录 A 文件与行号导航

| 文件 | 看什么 | 行号 |
|---|---|---|
| `modules/multiagent/models.py` | `AgentSpec.can_delegate` / `delegate_to` | 73 |
| | `AgentOutput.delegations` / `delegated_by` | 161-162 |
| | `DelegationRecord`（一条委派记录） | 188 |
| | `DelegationDecision`（决策 + 漏填降级） | 214 |
| | `RunResult.delegations` | 294 |
| | `TeamConfig.delegation_*`（4 个边界） | 358-361 |
| `modules/multiagent/delegation.py` | `DelegationPolicy`（边界策略） | 67 |
| | `DelegationLedger.can()`（五道闸） | 115-160 |
| | `DelegationLedger.open/close`（记账与回填） | 180-194 |
| | `DelegationExecutor.delegate()`（执行） | 253-303 |
| | `decide_delegation()`（决策） | 308 |
| `modules/multiagent/agents.py` | `RunContext` 委派三件套 | 111-120 |
| | `BaseAgent.run` 委派分支 | 303-322 |
| | `_delegation_text()` / prompt 拼接 | 361、394-395 |
| `modules/multiagent/harness.py` | `_new_delegation()`（装配 + span） | 233-310 |
| | `delegate()` 公开 API | 314 |
| | `finally` 写 `result.delegations` | 690-693 |
| `config.py` | 4 个 `MULTIAGENT_DELEGATION_*` | — |
| `examples/multiagent_delegation_demo.py` | 五段演示 | — |
| `tests/test_delegation.py` | 25 个用例 | — |
