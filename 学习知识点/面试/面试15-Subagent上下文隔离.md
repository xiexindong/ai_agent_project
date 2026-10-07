# 面试15：Subagent 子智能体——上下文隔离怎么做（防 token 爆炸与幻觉）

> 配套代码：[agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py)（隔离命门）、[delegation.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py)（委派治理）、[subgraph.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py)（子图形态）、[harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py)（并行编排：run / topo_batches / _run_batch / 一致性检测仲裁）
> 一句话结论：**Subagent 的本质不是"多开一个 Agent"，而是"开一份独立的 messages"。主对话只把最小任务包发过去，子 Agent 在独立上下文里跑完，只把 final answer 回流——隔离做在数据结构层，不靠 prompt 约定。**

---

## 〇、面试官视角（他为什么问这个）

1. 这是 Agent 工程的**真实痛点题**，不是八股：长对话 token 超限、上下文污染导致幻觉，是所有生产级 Agent 都会撞上的墙。
2. 能分清"嘴上说隔离"和"代码里真隔离"的候选人：很多 Demo 里的子 Agent 其实共享主 messages，一问代码就露馅。
3. 顺着这题可以连问五层：**怎么隔离 → 信息怎么传 → 结果怎么回 → 失控怎么治（深度/超时/环）→ 失败怎么降级**。能答全的就是做过工程化的人。

### 〇.1 先把术语说人话

| 术语 | 大白话解释（生活比喻） |
|---|---|
| 上下文（messages） | 和 AI 对话时的**聊天记录本**。每说一轮就往本上追加，AI 每回答一次都要把整本重新读一遍——本越厚，越贵（token 费）、越慢、越容易被旧内容带偏 |
| Subagent（子智能体） | 主管**临时请来的一位专家顾问**：单独给他一间小会议室（独立上下文）、一张写着任务的纸条（最小信息包），他干完活只交一份结论报告（AgentOutput），会议室里的讨论草稿全部就地销毁 |
| Supervisor（主管 Agent） | **项目经理**：不亲自干活，负责拆任务、派活、收报告、汇总结论 |
| 委派（Delegation） | 顾问干活时发现"这归隔壁老王管"，**当场把活转给老王**——是执行中途的横向转派，不是主管事前派活 |
| 拓扑分层（topo_batches） | 把任务按依赖关系**排成一波一波的**：没有依赖的先并行干，下游等上游出结果再开工。像做饭：洗菜和烧水可以同时进行，但炒菜必须等洗完菜 |
| 仲裁（Arbitration） | 几个顾问的报告**互相打架**时，请一位裁判来裁决采信谁，裁决过程留档 |
| Token | AI 计费的"字数单位"。上下文里的每个字都按 token 算钱，上下文越大越贵 |

---

## 一、30 秒电梯回答

> "聊天轮次一多，主上下文只增不减，迟早撞 token 上限，旧信息混杂还会诱发幻觉。我们的解法是 Subagent：**主管 Agent 先把问题拆成子任务，每个子 Agent 执行时就地新建自己的 messages，只含角色提示词和本次任务的最小信息包**——子任务指令、上游产出摘要、必要的同伴线索，主对话历史一行都不进去。子 Agent 内部的工具调用和中间轮次只活在它自己的栈帧里，跑完只有一个结构化的 `AgentOutput.content` 回流给主管汇总。项目里落了三套形态：Supervisor-Worker 团队、事中委派 Delegation、LangGraph 子图；并且配了生产治理——委派五道闸（开关/资格/存在性/深度次数/防环）、子任务超时、失败降级、冲突多数决。"

---

## 二、原理详解：隔离到底落在哪几行代码

### 2.1 命门：messages 每次新建，绝不继承主对话

[agents.py:240-243](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L240-L243)，每个子 Agent 每次执行都从这两条消息起步：

```python
# agents.py:240-243（_run_with_tools 内部，每次执行重新构造）
messages = [
    {"role": "system", "content": self.system_prompt},
    {"role": "user", "content": prompt},     # prompt = 本次任务的最小信息包
]
```

注意它**不接收任何外部 messages 参数**。对比主 Agent（AgentGraph 的 ReAct 循环）：主 messages 是跨轮累积、存 PostgreSQL 的；子 Agent 的 messages 是 `_run_with_tools` 的局部变量，方法返回栈帧销毁，上下文随之释放。这就是物理隔离，而不是在 prompt 里写一句"请忽略之前的对话"。

```
主 Agent 的 messages（只增不减）            子 Agent 的 messages（每次新建）
─────────────────────────────            ─────────────────────────────
system: 你是助手                           system: 你是数据分析师
user: 第1轮提问                            user: 【任务】分析Q3营收…（最小信息包）
assistant: 第1轮回答                              ↓ 跑完
user: 第2轮提问                       →     方法返回，messages 随栈帧销毁
assistant: 第2轮回答                            ↓ 回流
...（存 PG，越滚越厚）                     只有 AgentOutput.content 回到主管
```

### 2.2 信息怎么进子 Agent：最小信息包 + 三道截断

子 Agent 不是什么都看不到，而是只看**完成本子任务所必需**的信息，在 [agents.py:303-325](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L303-L325) 的 `run()` 里分四块采集，再由 [_build_prompt（agents.py:410-427）](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L410-L427) 拼装：

| 信息块 | 来源 | 防爆处理 |
|---|---|---|
| 任务指令 | `task.instruction`（主管拆出来的） | 主管规划时要求"具体到可直接执行" |
| 上游产出 | `ctx.upstream`（依赖任务的产出） | [_upstream_text（agents.py:398-407）](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L398-L407) **每条截断 1200 字** |
| 图片描述 | vision 模型先把图"翻译"成文字 | 全链路只付一次视觉开销，图不进每个 Agent |
| 同伴线索 | MessageBus 收件箱 | **最近 20 条、每条截断 400 字** |
| 委派产出 | 受托方回传 | [_delegation_text（agents.py:386-395）](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L386-L395)，失败也如实告知 |

设计要点：传的是**上游产出的文本摘要，不是上游的 messages**。即使上游做了 10 轮工具调用，下游看到的也只是一段结论——膨胀不会沿任务链传导。

### 2.3 结果怎么回：回流边界只有 final answer

[agents.py:360-372](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L360-L372)，子 Agent 执行完只返回一个结构化产出：

```python
# agents.py:360-367（run() 的成功返回）
return AgentOutput(
    agent=self.name,
    content=(content or "").strip(),   # ← 唯一回流进主管上下文的东西
    success=True,
    steps=steps,                        # 步数只用于计费/配额
    tool_calls=logs,                    # 工具日志只做账单留痕，不进主管 prompt
    ...
)
```

工具循环的中间观察（`_run_with_tools` 里 append 的 observe 消息）全部封死在子 Agent 内部。主管拿到 N 个 `AgentOutput` 后，在 [synthesize（agents.py:511-523）](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L511-L523) 里把成功/失败产出标注后汇成一份答案，失败部分如实写"该部分未能完成"，不替成员编内容。

### 2.4 一次请求的输入 → 处理 → 输出对照

以"对比 A、B 两款产品的财报并给出投资建议"为例：

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| ① 规划（agents.py:457） | 用户 query | 主管 LLM 结构化拆任务 | `[t1: 分析A财报, t2: 分析B财报, t3: 对比并建议(depends t1,t2)]` |
| ② 拓扑分层（harness.py:339-364） | 3 个子任务 | Kahn 算法按依赖分层 | `[[t1,t2], [t3]]` —— 第 1 批并行 2 个，第 2 批 1 个 |
| ③ 并行执行（harness.py:771-851） | 本批任务 | 线程池各跑各的 `agent.run()`，各自新建 messages | t1/t2 各自的 `AgentOutput`（content 几百字结论） |
| ④ 下游收包（agents.py:410-427） | t3 的指令 + ctx.upstream | t1/t2 产出各截断 1200 字拼进 t3 的 prompt | t3 在独立上下文里只见到两段摘要，不是两份完整执行历史 |
| ⑤ 一致性检测（harness.py:540-550） | 全部产出 | 内容相似度 < 阈值 0.35 判冲突 | 无冲突直接过；有冲突进仲裁 |
| ⑥ 汇总（agents.py:511） | N 个 AgentOutput.content | 主管 LLM 汇成一份 | 最终答案回流 API |

### 2.5 端到端链路（一次请求的流转）

```
POST /agent/team/run                         server.py:713
        │
        ▼
harness.run(query)                           harness.py:371
        │
        ├─ ① supervisor.plan()               agents.py:457   拆子任务（结构化 JSON，失败重试一次再降级直答）
        │     └─ 简单问题 → plan.is_direct 直接收工，不拆
        ├─ ② topo_batches()                  harness.py:339  Kahn 拓扑分层，同批并行、环直接剔除
        ├─ ③ 逐批 _run_batch()               harness.py:500（调用处）
        │     │                                            771（定义）/ 837（ThreadPoolExecutor）
        │     └─ agent.run(task, ctx)        agents.py:303   ← 上下文隔离在这一层发生
        ├─ ④ 一致性检测 + 仲裁（可选）        harness.py:540-573  产出打架时先检测、再仲裁
        ├─ ⑤ supervisor.synthesize()         agents.py:511  只吃 AgentOutput.content
        └─ ⑥ reviewer.review()（可选）        agents.py:546  结构化评审，解析失败默认放行不阻断
```

每一层都有降级：规划两次失败→单任务直答；某 worker 异常→`AgentOutput.failed()` 不外抛；汇总 LLM 挂了→回退纯拼接。**整条链路对调用方承诺不抛异常**。

### 2.6 harness 深挖：并行、超时、仲裁三段关键代码

**① 拓扑分层——同层并行、环直接剔除**（harness.py:352-364）：

```python
# Kahn 算法：每轮挑出"依赖全部完成"的任务作为一批
done: set[str] = set()
pending = list(tasks)
batches: list[list[SubTask]] = []
while pending:
    ready = [t for t in pending if all(d in done for d in t.depends_on)]
    if not ready:
        # 剩下的全都等不到上游 = 存在环或引用了不存在的 id
        return batches, [t.id for t in pending]   # 环里的任务剔除并记进第二个返回值
    ready.sort(key=lambda t: -t.priority)
    batches.append(ready)
    done.update(t.id for t in ready)
    pending = [t for t in pending if t.id not in done]
return batches, []
```

环不剔除的后果注释里写得很直白（harness.py:350）：while 循环永远排不完，**整次运行挂死（不是报错，是卡住）**——卡住比报错更难排查。

**② 并行收口——单个任务超时/异常只标记失败，不拖死同批**（harness.py:837-849）：

```python
# 线程数取「配置并发上限」与「本批任务数」的较小值：不多开空转线程
with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(batch))) as pool:
    futures = {pool.submit(_one, t): t for t in batch}
    for future, task in futures.items():
        wait = max(deadline - time.time(), 1.0)   # 距全局截止的剩余时间
        try:
            outputs.append(future.result(timeout=wait))
        except FutureTimeout:
            outputs.append(AgentOutput.failed(task.assignee, task.id, "任务执行超时"))
        except Exception as e:
            outputs.append(AgentOutput.failed(task.assignee, task.id, f"{type(e).__name__}: {e}"))
```

`wait` 用的是**全局截止时刻的剩余时间**而不是固定值：第 2 批开始时前面已消耗的时间要扣除，全队共享一个总超时（harness.py:416 的 `deadline`）。

**③ 一致性检测 + 仲裁——主管不硬选，看到的是"矛盾产出 + 裁决"**（harness.py:538-573）：

```python
synth_inputs = list(result.outputs)
if self.config.consensus_enabled and len(result.outputs) >= 2:
    cons = self.consensus.detect(result.outputs)      # 相似度低于阈值判冲突
    result.consensus = cons
    if cons.needs_arbitration:
        # 先给少数派发异议消息留痕
        for o in result.outputs:
            if o.success and o.agent not in majority:
                self.orchestrator.send("consensus", "你的结论与多数派不一致…", ...)
        arb = self.arbitrator.arbitrate(ctx.query, result.outputs, cons)
        if arb is not None:
            result.arbitration = arb
            synth_inputs.append(arb)                  # 仲裁结论也喂给主管汇总
```

相似度阈值默认 0.35（coordination.py:170 的 `Consensus.__init__`）；仲裁之上还可挂人工裁决（harness.py:576 起，ON_DISPUTE 中断点，人可改写最终答案或直接批准仲裁结论）。

### 2.7 事中委派：子 Agent 还能再甩活，但有五道闸

Worker 接到活后发现"这块该别人干"（如 analyst 缺原始数据），可以当场委派，[delegation.py:253-303](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L253-L303)。委派前先过 [DelegationLedger.can（delegation.py:115-160）](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L115-L160) 的五道闸：

| 闸 | 规则 | 不拦会怎样 |
|---|---|---|
| ① 开关 | `enabled` 默认 False | —（关闭时连决策 LLM 调用都不发生，零开销） |
| ② 资格 | `can_delegate` + 白名单 | 任何 Agent 都能把活甩出去 |
| ③ 存在性 | 受托方在花名册且启用 | 转给一个不存在的人 |
| ④ 预算 | depth / 单轮总次数 / 单人次数 | 委派链无限延长、成本失控 |
| ⑤ 防环 | 不能转自己、不能已在链上 | A→B→A 互相甩锅到步数耗尽，日志看着还都正常 |

被拒也留痕（`ledger.denials`），且委派方**照常自己干**——"想转没转成"绝不能让主任务失败。

### 2.8 第三种形态：LangGraph 子图（图套图）

[subgraph.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py) 用状态映射表达同一件事：父图字段映射进子图独立 state（[L287-306](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py#L287-L315)），子图终态 `report` 只映射回父图 `results[i].content`。生产四件套都在里面：

- **深度限制 + 防环**：[_spawn，subgraph.py:323-330](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py#L323-L330)
- **子任务超时**：线程池 + `future.result(timeout=)`（[L261-279](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py#L261-L279)），超时逐个降级、不拖死全局
- **失败降级**：最终答案只拼成功项，末尾附注降级说明（[L405-420](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py#L405-L420)）
- **冲突收敛**：同组产出多数决，平票取先完成，全程留痕（[L368-397](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py#L368-L397)）

> 注意边界：子图模块的默认执行器是确定性 Mock（输出"[worker-1] 已完成: xxx"），定位是可替换骨架；生产链路用的是 harness + agents 那套，由 [server.py:372](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py#L372) 的 `build_harness` 注入真实 LLM 与工具注册器。

### 2.9 三套形态对照表

| 维度 | Supervisor-Worker（agents.py + harness.py） | 事中委派（delegation.py） | LangGraph 子图（subgraph.py） |
|---|---|---|---|
| 谁发起分工 | 主管事前规划 | Worker 执行中临时转派 | 父图节点调用子图 |
| 隔离方式 | 每个 Worker 新建 messages | 受托方同样新建 messages | 父图 state 字段映射进子图独立 state |
| 依赖编排 | topo_batches 拓扑分层并行 | 沿委派链递归，深度受限 | 图结构本身表达 |
| 失控治理 | max_total_steps + 总超时 + 失败降级 | 五道闸（含防环） | max_depth + 链上防环 + 任务超时 |
| 冲突处理 | 相似度检测 + 仲裁者/人工裁决 | — | 同组多数决、平票取先完成 |
| 生产地位 | **生产链路**（server.py:372 注入真实依赖） | 生产链路（默认关开关） | 可替换骨架（默认执行器是 Mock） |

---

## 三、生活类比（讲隔离时最好用的比方）

把整套机制想象成**咨询公司接了一个大项目**：

| 本项目概念 | 咨询公司对应物 |
|---|---|
| 主 Agent 累积的 messages | 项目总群聊，从立项那天起所有消息都在里面，越翻越厚 |
| Subagent 独立上下文 | 项目经理把子课题**分包给一位专家**：专家只拿到一页任务简报，进自己的小办公室干活，**看不到总群聊** |
| `_build_prompt` 最小信息包 | 那页**任务简报**：只写这个子课题需要什么——任务说明、上游环节的两三句结论，没有废话 |
| 三道截断（1200 字/20 条/400 字） | 简报的**篇幅规定**：上游报告只附摘要，同事便签只带最近几张，防止简报比总群聊还厚 |
| `AgentOutput.content` 回流 | 专家交回的**一页结论报告**；他在办公室里的草稿、电话记录、查资料的过程全部留在办公室，不带回项目群 |
| 五道闸（深度/防环） | 公司的**转包制度**：转包要审批、有层级上限、不许你转给我我再转回给你 |
| 仲裁 | 两位专家结论打架时，请**行业裁判**出裁定书，裁定书一并附进给客户的最终报告 |

一句话：**Subagent 之于主 Agent，就是分包专家之于项目经理**——隔离的是工作空间（上下文），协作只靠简报（任务包）和报告（最终结果）两样东西往来。

---

## 四、面试串讲话术（90 秒版本）

> "我先说为什么需要它：主对话的 messages 只增不减，长会话有两个必然结果——token 成本和超限风险线性涨，无关旧信息还会诱导幻觉。
>
> 我们的做法是 Supervisor-Worker 模式。主管先做一次结构化规划把问题拆成子任务，简单问题直接答、不拆；真要拆的话按依赖做拓扑分层，同批丢线程池并行。关键是每个 Worker 执行时**就地新建自己的 messages，只有 system 和本次任务两条**，任务包也不是裸数据：上游产出每条截断 1200 字、同伴消息只取最近 20 条各 400 字，图片先转成文字只付一次视觉钱。跑完之后，Worker 的中间工具轮次全部封在自己的方法栈里，只有 AgentOutput 的 content 回流给主管做汇总。
>
> 另外我们支持事中委派：Worker 发现自己缺一块能力，可以转给更合适的同伴，但要过五道闸——开关、资格、存在性、深度次数预算、防环，A→B→A 直接拦。所有被拒都留痕，而且转不出去就自己继续干，不能因为一个可选能力赔上主流程。
>
> 工程上每层都有降级：规划两次失败降级单 Agent 直答、子任务超时和异常只标记失败、汇总失败回退纯拼接，整条链路对 API 层承诺不抛异常。"

---

## 五、追问清单（提前准备好答案）

**Q1：你说隔离，那 Worker 之间完全不能通信吗？任务有依赖怎么办？**
答：两条受控通道。① 依赖产出：拓扑分层后，上游的 `AgentOutput` 进 `ctx.upstream`，下游 prompt 里能看到摘要（1200 字截断，agents.py:403）；② MessageBus：点对点/广播消息，收件箱只取最近 20 条、每条 400 字（agents.py:310 注释）。共享是"线索级"的，不是"上下文复制级"的。

**Q2：子 Agent 就不会 token 超限了吗？**
答：单子任务有 `max_steps` 轮工具循环预算（默认 3），全队还有 `max_total_steps`（默认 30，harness.py:490 检查）和总超时双闸（harness.py:416 算出全局 deadline，每批开跑前 harness.py:482 检查，_run_batch 里按剩余时间收口 harness.py:840）；输入侧靠三道截断，所以单子任务的上下文是有上界的。

**Q3：多个 Worker 产出矛盾了听谁的？**
答：一致性检测按内容相似度判冲突（`Consensus` 默认阈值 0.35，coordination.py:170；检测入口 harness.py:545），冲突先走仲裁者裁决（harness.py:566），裁决结论作为额外一条 `synth_inputs` 喂给主管汇总（harness.py:572）；还可配置人工裁决中断点（harness.py:576，人可改写最终答案）；子图形态里是同组多数决、平票取先完成（subgraph.py:368-397）。裁决过程和被否决的产出都留痕，主管看到的是"矛盾产出 + 裁决"而不是自己硬选。

**Q4：委派和主管派活有什么区别？**
答：派活是事前、中心化、只有一层（规划阶段主管分）；委派是事中、横向、可递归——执行者接到活之后才发现某块该别人做，当场转（delegation.py:253-303）。真需要主管重拆走的是 replan，不是委派，所以候选名单里直接排除主管，防"甩回派活人"的无效回路。

**Q5：为什么子 Agent 不让共享主 messages？成本不是更低吗，省得重新拼 prompt？**
答：短期省一次拼接，长期把主对话的膨胀复制 N 份；还有两个隐性成本：一是 Worker 会被无关历史带偏产生幻觉，二是越权看到其他任务/其他轮次的信息，有数据隔离问题。我们选"重新拼一个最小包"，token 花在刀刃上。

**Q6：普通 `/chat` 会自动触发 Subagent 吗？**
答：目前不会自动触发，团队协作是独立入口 `POST /agent/team/run`（server.py:713）；主管在团队模式内部会判断简单问题直答。要让普通聊天在上下文将满时自动分流，可以在 AgentGraph 规划节点加一条路由到 harness，这是预留的演进点。

**Q7：失败一个 Worker 整单就废了吗？**
答：不会。Worker 内部异常被收口成 `AgentOutput.failed()`（harness.py:846-849 连 FutureTimeout 都只是标失败），不向上抛；汇总时失败项如实标注；默认关闭、可选开启的 replan 还能在有人失败时让主管重拆补跑（harness.py:529-530，受 `max_replans` 限制）。

**Q8：同进程线程池并行，有 GIL，并行的意义是什么？**
答：Worker 的耗时几乎全在等 LLM 网络响应和工具 I/O，属于 I/O 密集，GIL 在等待期间释放，线程池并行收益实打实；只有真出现独立部署/跨机器/算力密集信号时才需要上多进程或消息队列分布式，项目里 [distributed.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/distributed.py) 是那套方案的预留实现。

**Q9：拓扑分层如果任务间有环怎么办？比如 t1 依赖 t2、t2 又依赖 t1。**
答：topo_batches 每轮挑"依赖全部完成"的任务入批；环里的任务永远等不到上游，某轮挑不出任何任务时，剩余任务整体进第二个返回值被剔除（harness.py:357-359），run 里记一条 error 事件告知"存在循环依赖，已剔除"（harness.py:477-478），其余正常任务照跑。不剔除的后果是 while 死循环——整次运行挂死，不报错只卡住（harness.py:350 注释）。

**Q10：总超时是怎么落实到每个子任务的？第 1 批跑久了第 2 批怎么办？**
答：run 开头算一个绝对截止时刻 `deadline = started + timeout_seconds`（harness.py:416）。每批开跑前检查 `time.time() > deadline`，超了整批标"总超时，未执行"（harness.py:482-488）；批内每个 future 的 `timeout` 用 `deadline - time.time()` 的剩余时间（harness.py:840），所以不是每个任务独享一份超时，而是全队共享一块总预算——前面花超了，后面自然被压缩。

---

## 六、一句话记忆卡

> **隔离靠新建 messages（[agents.py:240](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L240-L243)），传参靠最小信息包（[agents.py:410](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L410-L427)），回流只有 content（[agents.py:360](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L360-L372)），并行收口在 harness（[harness.py:837-849](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L837-L849)），失控有五道闸（[delegation.py:115](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L115-L160)）。五件事缺一，就还是 Demo。**
