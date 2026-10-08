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
| 代码 | `agents.py` 的 `SupervisorAgent.plan()`；`harness.py` 第 427 行包在 `plan` span 里 |

**代码摘录**（`...` 表示省略的行，下同）：

```python
# 出处：modules/multiagent/harness.py:L427-L436
with self.tracer.span(trace, "plan", kind="plan", parent_id=root.span_id) as sp:
    sp.inputs.update({"query": query[:200],
                      "roster": [a.name for a in self.roster]})
    plan = self.supervisor.plan(ctx, self.roster)
    sp.set_io(outputs={"subtasks": len(plan.subtasks),
                       "is_direct": plan.is_direct,
                       "thought": (plan.thought or "")[:120]})
result.plan = plan
self._emit(result, "plan", plan.thought or f"拆出 {len(plan.subtasks)} 个子任务",
           self.supervisor.name, subtasks=[t.model_dump() for t in plan.subtasks])
```

**解读**：运行一开始，harness 就给"规划"这道工序挂了一块计时牌（span）：输入记下问题和花名册，规划完记下"拆了几个子任务、是不是直接答"。表格里"包在 plan span 里"指的就是这段——事后按 span 一查，规划花了多久、当时主管看到了什么，一目了然。

```python
# 出处：modules/multiagent/agents.py:L482-L507
try:
    data = self.llm.invoke_structured(prompt, system_prompt=self.system_prompt)
    return PlanResult.model_validate(data)
except Exception as first_err:
    ...
    _logger.warning(f"[supervisor] 规划解析失败，带错误重试: {first_err}")
    try:
        retry_prompt = (
            f"{prompt}\n\n上一次输出不合法，错误是：{first_err}\n"
            "请修正后重新输出，务必包含所有必需字段。"
        )
        data = self.llm.invoke_structured(retry_prompt, system_prompt=self.system_prompt)
        return PlanResult.model_validate(data)
    except Exception as second_err:
        _logger.error(f"[supervisor] 规划二次失败，降级为单 Agent 直答: {second_err}")
        fallback = roster[0].name if roster else self.name
        return PlanResult(
            thought="规划失败，降级为单任务执行",
            subtasks=[SubTask(id="t1", assignee=fallback, instruction=ctx.query)],
            needs_review=False,
        )
```

**解读**：主管规划走"结构化输出"：让模型按固定表格填空，填完当场验收。第一次没填对，就把验收错误原样念给它听、让它自己改；再不行就降级成"派一个人直接答"，保证链路永远有产出——这是"主管读花名册 + 问题，输出 PlanResult"背后的一根保险丝。

### ② 派活（dispatch）

| 项 | 内容 |
|---|---|
| 做什么 | 按 `depends_on` 做拓扑分批，批内用线程池并行（`max_workers=4`），批间串行 |
| 两条预算 | `max_total_steps=30`（整次运行总步数，超了后续批次直接标记失败）、`timeout_seconds=120`（整次运行总超时） |
| 输入 → 输出 | 第 1 批 `[t1]` → `[AgentOutput(agent="researcher_1", success=True, content="Q3 1100 万，Q4 1200 万", duration_ms=812)]` |
| 代码 | `harness.py` 的 `_run_batch()`；每批一个 `dispatch.batch_N` span |

**代码摘录**：

```python
# 出处：modules/multiagent/harness.py:L481-L496
for batch_no, batch in enumerate(batches, 1):
    if time.time() > deadline:
        for t in batch:
            result.outputs.append(
                AgentOutput.failed(t.assignee, t.id, "总超时，未执行")
            )
        result.status = RunStatus.TIMEOUT
        self._emit(result, "error", f"第 {batch_no} 批因总超时未执行")
        break
    if steps_used >= self.config.max_total_steps:
        for t in batch:
            result.outputs.append(
                AgentOutput.failed(t.assignee, t.id, "总步数预算耗尽，未执行")
            )
        self._emit(result, "error", f"步数预算耗尽，第 {batch_no} 批起不再执行")
        break
```

**解读**：这是"两条预算"的执法现场：每批开工前先看表（总超时 `deadline`）再点账（总步数 `steps_used`），哪个不够了，这一批整队标成失败、直接收工——对应"超了后续批次直接标记失败"。批次本身是 `topo_batches()`（`harness.py:L339-L364`）按 `depends_on` 用 Kahn 算法排出来的，处在环里的任务会被剔除。

```python
# 出处：modules/multiagent/harness.py:L836-L851
# 线程数取「配置并发上限」与「本批任务数」的较小值：不多开空转线程
with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(batch))) as pool:
    futures = {pool.submit(_one, t): t for t in batch}
    for future, task in futures.items():
        wait = max(deadline - time.time(), 1.0)   # 剩余时间不足 1 秒时也至少等 1 秒，避免 0 秒立刻超时
        try:
            outputs.append(future.result(timeout=wait))
        except FutureTimeout:
            _logger.error(f"任务超时: {task.id}（等待 {wait:.1f}s 未返回）")
            outputs.append(AgentOutput.failed(task.assignee, task.id, "任务执行超时"))
        except Exception as e:
            outputs.append(
                AgentOutput.failed(task.assignee, task.id, f"{type(e).__name__}: {e}")
            )
_logger.info(f"第 {batch_no} 批完成: {len(outputs)} 个产出")
return outputs
```

**解读**：同一批任务扔进线程池一起干——像一批零件同时上 4 个工位，谁先干完谁交货。每个人的等待时间不能超过整次运行的剩余 deadline（所以叫"总超时"而不是单任务超时）；等超了记"任务执行超时"，中途抛异常记异常名——单人出事不影响同批其他人。

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

**代码摘录**：

```python
# 出处：modules/multiagent/delegation.py:L132-L160（DelegationLedger.can 的五道闸）
if not p.enabled:
    return deny("委派未启用")
if delegator == delegatee:
    return deny("不能委派给自己")
if delegatee in chain:
    return deny(f"委派成环（链上已有 {delegatee}）")
if depth >= p.max_depth:
    return deny(f"已达委派深度上限 {p.max_depth}")
if len(self.records) >= p.max_per_run:
    return deny(f"已达单次运行委派上限 {p.max_per_run}")
if self.count(delegator) >= p.max_per_agent:
    return deny(f"{delegator} 已达单人委派上限 {p.max_per_agent}")
spec = self.roster.get(delegator)
...
target = self.roster.get(delegatee)
if target is None:
    return deny(f"受托方 {delegatee} 不在花名册")
if not target.enabled:
    return deny(f"受托方 {delegatee} 已停用")
return True, "放行"
```

**解读**：这就是"五道闸"的本体：像机场安检一道一道过，任何一道不过就地拦下（`deny` 会把中文原因记进台账 `denials`，事后能查"为什么这块没转成"）。全过了才返回 `True, "放行"`——对应表格里的"开关 → 资格 → 存在性 → 深度/次数 → 防环"。

```python
# 出处：modules/multiagent/agents.py:L324-L342
dex = getattr(ctx, "delegation", None)
if dex is not None and getattr(dex, "enabled", False) \
        and self.spec.can_delegate and task is not None:
    decision = decide_delegation(
        self.llm, instruction, dex.candidates(self.name), self.system_prompt
    )
    if decision is not None and decision.should_delegate:
        rec = dex.delegate(
            from_agent=self.name,
            to_agent=decision.to_agent,
            instruction=decision.instruction,
            parent_task_id=task.id,
            depth=getattr(ctx, "delegation_depth", 0),
            reason=decision.reason,
            chain=list(getattr(ctx, "delegation_chain", []) or []),
        )
        if rec is not None:
            delegated.append(rec)
            _logger.info(f"[{self.name}] 已把部分工作委派给 {rec.to_agent}: {rec.instruction[:60]}")
```

**解读**：执行者接到活，先确认"装了委派器 + 我有资格 + 手上真有任务"三个条件齐了，才去问一次模型"这块要不要转给别人"（委派默认关闭时连这次问都不问，零额外开销）。模型说转、治理也放行，才真正转出去并记账；任何一环说不，`rec` 就是 `None`，自己接着干——对应"被拒行为：返回 None，主流程照常往下走"。

### ④ 消息总线（横向对话）

| 项 | 内容 |
|---|---|
| 做什么 | 成员之间发 `finding`（共享发现）/ `clarify`（请求澄清）/ `objection`（异议）/ `delegate`（转派意向） |
| 三条约束 | 消息只在同一批次内有效、总线有容量上限、异议不触发新的执行轮次（防退化成群聊） |
| 共享产出 | `share_findings=True` 时每批结束把产出广播，队友 prompt 的「同伴消息」里可见，避免重复查同一件事 |
| 代码 | `coordination.py` 的 `MessageBus`；实际挂的是 `orchestration/bus.py` 的 `OrchestratedBus` 门面（读直达、写走治理） |

**代码摘录**：

```python
# 出处：modules/multiagent/coordination.py:L132-L144
def inbox(self, name: str, limit: int = 20) -> list:
    """取某人能看到的消息（点对点 + 广播），按时间正序。"""
    items = [m for m in self._messages if m.visible_to(name) and m.sender != name]
    return items[: max(1, int(limit))]

def inbox_text(self, name: str, limit: int = 20) -> str:
    """把收件箱拼成可塞进 prompt 的文本。没人发消息时返回空串。"""
    msgs = self.inbox(name, limit)
    if not msgs:
        return ""
    return "\n".join(
        f"- [{m.kind.value}] {m.sender}: {m.content[:400]}" for m in msgs
    )
```

**解读**：总线本质是"公告栏 + 收件箱"：`inbox()` 取"点名给我的 + 广播给我的"，还剔除了自己发的（自己写的不用再读一遍）；`inbox_text()` 把收件箱拼成一小段文字直接塞进 prompt，每条截 400 字。没人说话就返回空串，prompt 里自然就没有"同伴消息"这一段——对应表格里的"共享产出"。

```python
# 出处：modules/orchestration/bus.py:L36-L43
def post(self, sender: str, content: str, receiver: str = "*",
         kind: MessageKind = MessageKind.FINDING, **kw) -> Optional[Envelope]:
    decision = self.orchestrator.send(sender, content, receiver, kind, **kw)
    return decision.envelope

def inbox_text(self, name: str, limit: int = 20) -> str:
    """读自己能看到的消息，拼成可进 prompt 的文本。"""
    return self.bus.inbox_text(name, limit)
```

**解读**：Agent 拿到的 bus 其实是这个门面：读消息直接转手给真总线（读直达），发消息必须经过 `orchestrator.send()` 这道关（写走治理）。好比公司前台——来的文件随便收，往外发的一律先登记审核，谁也绕不过去。这就是表格里"OrchestratedBus 门面（读直达、写走治理）"。

### ⑤ 一致性检测（consensus）

| 项 | 内容 |
|---|---|
| 做什么 | 比较多个产出的文本相似度，低于阈值判定为"打架" |
| 默认 | 开（只多记一条 trace，不改答案路径） |
| 输入 → 输出 | 两份产出相似度 `0.62` ≥ `consensus_threshold=0.35` → `ConsensusResult(conflict=False, similarity=0.62)`<br>若相似度 `0.12` → `conflict=True`，触发仲裁 |
| 代码 | `coordination.py` 的 `Consensus.detect()` |

**代码摘录**：

```python
# 出处：modules/multiagent/coordination.py:L192-L217
min_sim = 1.0
for i in range(len(usable)):
    for j in range(i + 1, len(usable)):
        min_sim = min(min_sim, text_overlap(usable[i].content, usable[j].content))

# ...（中间省略：把产出按相似度做贪心聚类，≥阈值的进同一簇）...

clusters.sort(key=len, reverse=True)
majority = [o.agent for o in clusters[0]]
has_conflict = min_sim < self.threshold
# 【什么时候需要仲裁】
#   两种情况：产出互相矛盾，或没有形成多数派（三个人或三类说法，
#   最大簇也没过半）。后者同样说明结论没收敛，不能挑一个就用。
needs_arbitration = has_conflict or len(clusters[0]) * 2 <= len(usable)
```

**解读**：`detect()` 先把所有产出两两算文本重叠度（`text_overlap`，字符 2-gram 的 Jaccard，见 `coordination.py:L58-L71`，中文不用分词），记下最小的那个——低于阈值（默认 0.35）就判"打架"；再把产出按相似度聚成几簇，最大的簇算"多数派"。注意就算不打架，三类说法各执一词（最大簇没过半）同样判"没收敛、要仲裁"——宁可多裁一次，也不把矛盾答案直接端给用户。对应"低于阈值判定为打架"和 0.62 / 0.12 那两个示例数。

### ⑥ 仲裁（arbitration）

| 项 | 内容 |
|---|---|
| 做什么 | 冲突时让审核者读全部产出后定案，结论作为最终答案，跳过主管汇总 |
| 默认 | 开（`arbitration_enabled=True`） |
| 与人工介入的关系 | 开了 HITL 时，仲裁结论会升级成人待办，由真人拍板；没人值守则按超时处理 |
| 代码 | `coordination.py` 的 `Arbitrator.arbitrate()`；结果进 `RunResult.arbitration` |

**代码摘录**：

```python
# 出处：modules/multiagent/coordination.py:L259-L284
def arbitrate(self, query: str, outputs: list[AgentOutput],
              consensus: ConsensusResult) -> Optional[AgentOutput]:
    if self.judge is None:
        return None
    blocks = [
        f"【{o.agent}】{o.content[:1200]}" if o.success else f"【{o.agent}】（失败）{o.error}"
        for o in outputs
    ]
    prompt = self.ARBITRATION_PROMPT.format(
        query=query, outputs="\n\n".join(blocks), note=consensus.note
    )
    try:
        content = self.judge.llm.invoke(
            prompt, system_prompt=getattr(self.judge, "system_prompt", ""),
            temperature=0.2,
        )
        return AgentOutput(
            agent=self.judge.name,
            ...
            content=(content or "").strip(),
            success=True, steps=1,
        )
    except Exception as e:
        _logger.warning(f"仲裁调用失败（由多数派兜底）: {type(e).__name__}: {e}")
        return None
```

**解读**：仲裁像"开庭判案"：把所有产出（每人一份、截 1200 字）连同一致性判定的结论抄给审核者，让它读完给一份裁决书——采纳什么、不采纳什么、为什么。裁决书包装成一个普通 `AgentOutput` 返回，harness 会把它追加进汇总输入；审核者自己调用失败就返回 `None`，上层拿多数派内容兜底——对应"冲突时让审核者读全部产出后定案"。

### ⑦ 重规划（replan）

| 项 | 内容 |
|---|---|
| 做什么 | 有成员失败时，让主管基于已完成的产出重新拆一次，只补失败/未做的部分 |
| 默认 | **关闭**（要多调一次 LLM，是真金白银的成本），开启后最多 `max_replans=1` 次 |
| 输入 → 输出 | `failed_agents=["analyst_1"]` → 主管补出 `[{id:"t2b", assignee:"analyst_1", instruction:"只算 Q4 同比"}]`，`replan_count=1` |
| 代码 | `coordination.py` 的 `ReplanController` |

**代码摘录**：

```python
# 出处：modules/multiagent/coordination.py:L314-L333
def build_replan(self, ctx, outputs: list[AgentOutput], roster) -> Optional[PlanResult]:
    if self.supervisor is None:
        return None
    done = [o.agent for o in outputs if o.success]
    failed = [f"{o.agent}: {o.error}" for o in outputs if not o.success]
    original = ctx.query
    try:
        ctx.query = (
            f"{original}\n\n"
            + self.REPLAN_NOTE.format(
                done=", ".join(done) or "（无）",
                failed="; ".join(failed) or "（无）",
            )
        )
        return self.supervisor.plan(ctx, roster)
    except Exception as e:
        _logger.warning(f"重规划失败（沿用原计划结果）: {type(e).__name__}: {e}")
        return None
    finally:
        ctx.query = original
```

**解读**：重规划的做法很省：把"已完成谁、失败谁"的现状临时拼进问题里，让主管照旧再 plan 一次，只补没做的部分；不管成没成，`finally` 都把原始问题还原回去，不给后续环节留"上轮谁失败了"的残渣。默认只给 1 次机会（`max_replans=1`），由 harness 的 `_try_replan`（`harness.py:L724-L767`）按这个上限驱动循环——对应"基于已完成的产出重新拆一次，只补失败/未做的部分"。

### ⑧ 人工介入（HITL）

| 项 | 内容 |
|---|---|
| 做什么 | 敏感动作前停下来等人：工具审批、仲裁升级、终审修改 |
| 默认 | **关闭**（没人值守的场合会一直等到超时） |
| 三条路径 | `gate.approve()` 审批工具 / 仲裁升级成人待办 / 终审后人工修改答案（优先级最高） |
| 代码 | `modules/hitl/`；harness 通过 `ctx.gate` 与 `Orchestrator.escalations` 接线 |
| 详文 | [`HITL与多Agent协调.md`](./HITL与多Agent协调.md) |

**代码摘录**：

```python
# 出处：modules/hitl/gate.py:L133-L163（HumanGate.ask 的三个出口，构造请求的行省略）
# ① 策略没开：直接放行。这样 HITL 可以整块开关而不改动调用点。
if not self.policy.kind_enabled(kind):
    return HumanReply(
        request_id="", action=HumanAction.APPROVE, operator="policy",
        note=f"中断点 {kind.value} 未启用，自动放行",
    )
...
# ② 自动答复器：无人值守 / 测试 / 演示
if self.resolver is not None:
    auto = self.resolver(req)
    if auto is not None:
        self.store.submit(auto)
        return auto

# ③ 等人
reply = self.store.wait(req.id, req.timeout_seconds)
if reply is not None:
    return reply
```

**解读**：`ask()` 只有三个出口：策略没开 → 立刻放行（关掉 HITL 等于没装，零开销）；装了自动答复器 → 用它的答案、不阻塞（测试和无人值守就是这么跑通的）；否则真等人，等到超时按预设的默认动作兜底——对应"没人值守的场合会一直等到超时"。

```python
# 出处：modules/multiagent/harness.py:L449-L461（harness 侧的调用点：计划确认）
if self.gate is not None and self.gate.policy.kind_enabled(InterruptKind.BEFORE_PLAN):
    reply = self.gate.ask(
        kind=InterruptKind.BEFORE_PLAN,
        title=f"确认任务分解（{len(plan.subtasks)} 个子任务）",
        reason="计划阶段人工确认已开启",
        context={"thought": plan.thought,
                 "subtasks": [t.model_dump() for t in plan.subtasks]},
        run_id=result.run_id, session_id=ctx.session_id,
        agent=self.supervisor.name if self.supervisor else "",
    )
    result.interrupt_count += 1
    self._emit(result, "hitl", f"计划审批: {reply.action.value}", "human",
               action=reply.action.value)
```

**解读**：harness 里每个 HITL 点都是同一套路：先问 gate 这个中断点开没开，开了才把上下文递过去等人，人答完写一条 `hitl` trace。这段是"计划确认"——主管刚拆完活，人可以看一眼"这么拆合理吗"，点了终止或不批准，运行就此打住，一个子任务都不会跑。

### ⑨ 终审（review）+ ⑩ 汇总（synthesize）

| 项 | 内容 |
|---|---|
| 终审 | `review_enabled=True` 时由 reviewer 对最终答案挑毛病，结果进 `RunResult.review`（不改答案，只留意见） |
| 汇总 | 主管把所有产出合成一段答案；若人工改过（`human_final` 非空）则直接采用人工版本，跳过汇总 |
| 输入 → 输出 | 合成输入 `2` 份产出 → `final_answer="2024 Q4 营收 1200 万，环比 Q3（1100 万）增长 9.1%"` |

**终审代码摘录**：

```python
# 出处：modules/multiagent/agents.py:L546-L569
def review(self, ctx: RunContext, draft: str) -> AgentOutput:
    """评审草稿。解析失败不阻断主流程：返回 passed=True 的中性结论。"""
    started = time.time()
    prompt = self.REVIEW_PROMPT.format(
        query=ctx.query, draft=draft[:4000], schema=json_schema_hint(ReviewVerdict)   # 草稿截断到 4000 字
    )
    try:
        data = self.llm.invoke_structured(prompt, system_prompt=self.system_prompt)
        verdict = ReviewVerdict.model_validate(data)
        content = (
            f"评审{'通过' if verdict.passed else '不通过'}（{verdict.score:.2f}）\n"
            f"问题：{'; '.join(verdict.issues) or '无'}\n建议：{verdict.suggestion or '无'}"
        )
        return AgentOutput(
            agent=self.name, role=self.spec.role, content=content, success=True,
            duration_ms=int((time.time() - started) * 1000),
        )
    except Exception as e:
        _logger.warning(f"[{self.name}] 评审解析失败，按通过处理: {e}")
        return AgentOutput(
            agent=self.name, role=self.spec.role,
            content=f"评审未生效（解析失败: {e}），默认放行",
            success=True, duration_ms=int((time.time() - started) * 1000),
        )
```

**解读**：终审是"质检员签字"：把最终答案（截 4000 字）连同评审表格的 schema 交给 reviewer，拿回结构化结论——过/不过、分数、问题清单、建议。它只出意见不改答案（改答案的权力在 HITL 手里）；连质检自己解析失败也按"默认放行"处理，不让质检环节卡住生产线——对应"不改答案，只留意见"。

**汇总代码摘录**：

```python
# 出处：modules/multiagent/harness.py:L603-L614
if human_final:
    result.final_answer = human_final
    self._emit(result, "synthesize", "采用人工裁决结论（跳过汇总）", "human")
elif any(o.success for o in result.outputs):
    with self.tracer.span(trace, "synthesize", kind="synthesize",
                          parent_id=root.span_id) as ssp:
        ssp.inputs.update({"inputs": len(synth_inputs)})
        draft = self.supervisor.synthesize(ctx, synth_inputs)
        ssp.set_io(outputs={"chars": len(draft or "")})
    result.final_answer = draft or "（团队未产出有效结论）"
    self._emit(result, "synthesize", "汇总完成", self.supervisor.name)
```

**解读**：汇总先看有没有"人写的答案"（`human_final`，来自冲突裁决时的人工改写），有就直接采用、连主管都不用再合成——人工版本优先级最高；没有才进 `synthesize` span，让主管把所有产出（有仲裁结论时也一并带上）合成一段。对应"若人工改过则直接采用人工版本，跳过汇总"。主管自己的 `synthesize()`（`agents.py:L511-L523`）还有一个兜底：汇总调用失败就退化为把成功产出原样拼接。

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

**代码摘录**（门面 `OrchestratedBus.post()` 会转手调 `orchestrator.send()`，四道关卡就在里面）：

```python
# 出处：modules/orchestration/orchestrator.py:L153-L199（部分返回行省略）
# ---- 第 1 关：权限 / 敏感词 / 参数 --------------------------------
verdict = self.guard.check(env)
if not verdict.passed:
    env.mark(DeliveryStatus.REJECTED, verdict.reason)
    ...
    return RouteDecision.blocked(RouteAction.REJECT, verdict.code, verdict.reason,
                                 env, self._ms(t0))

# ---- 第 2 关：限流 ------------------------------------------------
ok, by = self.limiter.acquire(sender)
if not ok:
    ...
    return RouteDecision.blocked(RouteAction.DROP, RejectCode.RATE_LIMITED,
                                 f"被 {by} 限流，消息未投递", env, self._ms(t0))

# ---- 第 3 关：配额与跳数 ------------------------------------------
allowed, why = self.quota.check(sender, env.round, env.hop)
if not allowed:
    ...
# ---- 第 4 关：防死循环（重复 / 乒乓）------------------------------
loop = self._detect_loop(env)
if loop is not None:
    env.mark(DeliveryStatus.ESCALATED, loop)
    ...

# ---- 投递 --------------------------------------------------------
env.mark(DeliveryStatus.DELIVERED)
self.bus.post_envelope(env)
self.quota.consume(sender, env.round)
```

**解读**：这段就是本节开头流程图的本体：每条消息先过 guard（敏感词/长度/权限），再从令牌桶领"通行证"（限流），再查配额与跳数，然后做重复/乒乓检测，四关全过才标 `DELIVERED` 落进总线并扣减配额。被拦的消息按原因分别标 `REJECTED` / `DROPPED` / `ESCALATED`，连同 `RouteDecision` 一起返回——所以"这条为什么没送到"永远查得到，对应"被拒的消息记进 escalations 或返回 RejectCode"。

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

**收口那段的真实代码**（表格里 `status / error / duration_ms` 一行说的就是它）：

```python
# 出处：modules/multiagent/harness.py:L676-L717（注释块有省略）
finally:
    ...
    if self.gate is not None:
        self.gate.cancel_waiting()
    ...
    if self._delegation_ledger is not None:
        result.delegations = list(self._delegation_ledger.records)
        if self._delegation_ledger.records or self._delegation_ledger.denials:
            _logger.info(f"委派汇总: {self._delegation_ledger.summary()}")
    ...
    final_status = {
        RunStatus.TIMEOUT: SpanStatus.TIMEOUT,
        RunStatus.FAILED: SpanStatus.ERROR,
    }.get(result.status, SpanStatus.OK)
    self.tracer.end_span(
        root, final_status,
        outputs={"status": result.status.value, "outputs": len(result.outputs),
                 "escalated": len(self.orchestrator.escalations)},
        error=result.error[:200],
    )
    self.tracer.finish(trace)
    result.spans = [
        sp.model_dump() for sp in
        sorted(trace.spans.values(), key=lambda s: (s.start_ms, s.span_id))
    ]
```

**解读**：`finally` 是"散场前关灯查门"：`run()` 里有直接答、人工终止、超时、异常好几个出口，不管从哪个走，都要作废还在等的人工审批、把委派台账抄进 `result.delegations`、关闭根 span 并把 span 快照按开始时间排序写进 `result.spans`。所以失败的那次也有完整链路可查——对应表格里"由 finally 收口"，也呼应第四节那句"失败的那次最需要 trace"。

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
