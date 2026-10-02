# 面试15：Subagent 子智能体——上下文隔离怎么做（防 token 爆炸与幻觉）

> 配套代码：[agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py)（隔离命门）、[delegation.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py)（委派治理）、[subgraph.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py)（子图形态）、[harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py)（并行编排）
> 一句话结论：**Subagent 的本质不是"多开一个 Agent"，而是"开一份独立的 messages"。主对话只把最小任务包 postMessage 过去，子 Agent 在独立上下文里跑完，只把 final answer 回流——隔离做在数据结构层，不靠 prompt 约定。**

---

## 〇、面试官视角（他为什么问这个）

1. 这是 Agent 工程的**真实痛点题**，不是八股：长对话 token 超限、上下文污染导致幻觉，是所有生产级 Agent 都会撞上的墙。
2. 能分清"嘴上说隔离"和"代码里真隔离"的候选人：很多 Demo 里的子 Agent 其实共享主 messages，一问代码就露馅。
3. 顺着这题可以连问五层：**怎么隔离 → 信息怎么传 → 结果怎么回 → 失控怎么治（深度/超时/环）→ 失败怎么降级**。能答全的就是做过工程化的人。

---

## 一、30 秒电梯回答

> "聊天轮次一多，主上下文只增不减，迟早撞 token 上限，旧信息混杂还会诱发幻觉。我们的解法是 Subagent：**主管 Agent 先把问题拆成子任务，每个子 Agent 执行时就地新建自己的 messages，只含角色提示词和本次任务的最小信息包**——子任务指令、上游产出摘要、必要的同伴线索，主对话历史一行都不进去。子 Agent 内部的工具调用和中间轮次只活在它自己的栈帧里，跑完只有一个结构化的 `AgentOutput.content` 回流给主管汇总。项目里落了三套形态：Supervisor-Worker 团队、事中委派 Delegation、LangGraph 子图；并且配了生产治理——委派五道闸（开关/资格/存在性/深度次数/防环）、子任务超时、失败降级、冲突多数决。"

---

## 二、原理详解：隔离到底落在哪几行代码

### 2.1 命门：messages 每次新建，绝不继承主对话

[agents.py:240-243](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L240-L243)，每个子 Agent 每次执行都从这两条消息起步：

```python
messages = [
    {"role": "system", "content": self.system_prompt},
    {"role": "user", "content": prompt},     # prompt = 本次任务的最小信息包
]
```

注意它**不接收任何外部 messages 参数**。对比主 Agent（AgentGraph 的 ReAct 循环）：主 messages 是跨轮累积、存 PostgreSQL 的；子 Agent 的 messages 是 `_run_with_tools` 的局部变量，方法返回栈帧销毁，上下文随之释放。这就是物理隔离，而不是在 prompt 里写一句"请忽略之前的对话"。

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

### 2.4 端到端链路（一次请求的流转）

```
POST /agent/team/run                         server.py:681
        │
        ▼
harness.run(query)                           harness.py:371
        │
        ├─ ① supervisor.plan()               agents.py:457   拆子任务（结构化 JSON，失败重试一次再降级直答）
        │     └─ 简单问题 → plan.is_direct 直接收工，不拆
        ├─ ② topo_batches()                  harness.py:339  Kahn 拓扑分层，同批并行、环直接剔除
        ├─ ③ ThreadPool 并行 _run_batch      harness.py:481  每批各跑各的 run()，各自新建 messages
        │     └─ agent.run(task, ctx)        agents.py:303   ← 上下文隔离在这一层发生
        ├─ ④ 一致性检测 + 仲裁（可选）        harness.py:540  产出打架时多数决
        ├─ ⑤ supervisor.synthesize()         agents.py:511  只吃 AgentOutput.content
        └─ ⑥ reviewer.review()（可选）        agents.py:546  结构化评审，解析失败默认放行不阻断
```

每一层都有降级：规划两次失败→单任务直答；某 worker 异常→`AgentOutput.failed()` 不外抛；汇总 LLM 挂了→回退纯拼接。**整条链路对调用方承诺不抛异常**。

### 2.5 事中委派：子 Agent 还能再甩活，但有五道闸

Worker 接到活后发现"这块该别人干"（如 analyst 缺原始数据），可以当场委派，[delegation.py:253-303](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L253-L303)。委派前先过 [DelegationLedger.can（delegation.py:115-160）](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L115-L160) 的五道闸：

| 闸 | 规则 | 不拦会怎样 |
|---|---|---|
| ① 开关 | `enabled` 默认 False | —（关闭时连决策 LLM 调用都不发生，零开销） |
| ② 资格 | `can_delegate` + 白名单 | 任何 Agent 都能把活甩出去 |
| ③ 存在性 | 受托方在花名册且启用 | 转给一个不存在的人 |
| ④ 预算 | depth / 单轮总次数 / 单人次数 | 委派链无限延长、成本失控 |
| ⑤ 防环 | 不能转自己、不能已在链上 | A→B→A 互相甩锅到步数耗尽，日志看着还都正常 |

被拒也留痕（`ledger.denials`），且委派方**照常自己干**——"想转没转成"绝不能让主任务失败。

### 2.6 第三种形态：LangGraph 子图（图套图）

[subgraph.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py) 用状态映射表达同一件事：父图字段映射进子图独立 state（[L287-306](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py#L287-L315)），子图终态 `report` 只映射回父图 `results[i].content`。生产四件套都在里面：

- **深度限制 + 防环**：[_spawn，subgraph.py:323-330](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py#L323-L330)
- **子任务超时**：线程池 + `future.result(timeout=)`（[L263-279](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py#L263-L279)），超时逐个降级、不拖死全局
- **失败降级**：最终答案只拼成功项，末尾附注降级说明（[L405-420](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py#L405-L420)）
- **冲突收敛**：同组产出多数决，平票取先完成，全程留痕（[L368-397](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/subgraph.py#L368-L397)）

> 注意边界：子图模块的默认执行器是确定性 Mock（输出"[worker-1] 已完成: xxx"），定位是可替换骨架；生产链路用的是 harness + agents 那套，由 [server.py:338-340](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py#L338-L340) 的 `build_harness` 注入真实 LLM 与工具注册器。

---

## 三、JavaScript 类比（讲隔离时最好用的比方）

| 本项目概念 | JS 对应物 |
|---|---|
| 主 Agent 累积的 messages | 主线程作用域里不断增长的大数组 |
| Subagent 独立上下文 | `new Worker('worker.js')` 的独立作用域，访问不了主线程变量 |
| `_build_prompt` 最小信息包 | `worker.postMessage(子任务数据)`——结构化克隆，只传需要的数据 |
| `AgentOutput.content` 回流 | `self.postMessage(结果)` / `onmessage` 接收，只能拿回最终结果 |
| 中间工具调用不外溢 | Worker 内部的局部变量，主线程拿不到也不关心 |
| 五道闸（深度/防环） | 自己实现的任务转发配额 + 调用链环检测 |

一句话：**Subagent 之于主 Agent，就是 Web Worker 之于主线程**——隔离的是作用域（上下文），通信只靠消息（任务包 / 最终结果）。

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
答：两条受控通道。① 依赖产出：拓扑分层后，上游的 `AgentOutput` 进 `ctx.upstream`，下游 prompt 里能看到摘要（1200 字截断）；② MessageBus：点对点/广播消息，收件箱只取最近 20 条、每条 400 字。共享是"线索级"的，不是"上下文复制级"的。

**Q2：子 Agent 就不会 token 超限了吗？**
答：单子任务有 `max_steps` 轮工具循环预算（默认 3），全队还有 `max_total_steps`（默认 30）和总超时 120s 双闸（[harness.py:490-496](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L490-L496)）；输入侧靠三道截断，所以单子任务的上下文是有上界的。

**Q3：多个 Worker 产出矛盾了听谁的？**
答：一致性检测按内容相似度判冲突（阈值 0.35），冲突先走仲裁者裁决，可配置人工裁决；子图形态里是同组多数决、平票取先完成，裁决过程和被否决的产出都留痕，主管汇总时看到的是"矛盾产出 + 裁决"而不是自己硬选。

**Q4：委派和主管派活有什么区别？**
答：派活是事前、中心化、只有一层（规划阶段主管分）；委派是事中、横向、可递归——执行者接到活之后才发现某块该别人做，当场转。真需要主管重拆走的是 replan，不是委派，所以候选名单里直接排除主管，防"甩回派活人"的无效回路。

**Q5：为什么子 Agent 不让共享主 messages？成本不是更低吗，省得重新拼 prompt？**
答：短期省一次拼接，长期把主对话的膨胀复制 N 份；还有两个隐性成本：一是 Worker 会被无关历史带偏产生幻觉，二是越权看到其他任务/其他轮次的信息，有数据隔离问题。我们选"重新拼一个最小包"，token 花在刀刃上。

**Q6：普通 `/chat` 会自动触发 Subagent 吗？**
答：目前不会自动触发，团队协作是独立入口 `POST /agent/team/run`；主管在团队模式内部会判断简单问题直答。要让普通聊天在上下文将满时自动分流，可以在 AgentGraph 规划节点加一条路由到 harness，这是预留的演进点。

**Q7：失败一个 Worker 整单就废了吗？**
答：不会。Worker 内部异常被收口成 `AgentOutput.failed()`，不向上抛；汇总时失败项如实标注；默认关闭、可选开启的 replan 还能在有人失败时让主管重拆补跑（受 `max_replans` 限制）。

**Q8：同进程线程池并行，有 GIL，并行的意义是什么？**
答：Worker 的耗时几乎全在等 LLM 网络响应和工具 I/O，属于 I/O 密集，GIL 在等待期间释放，线程池并行收益实打实；只有真出现独立部署/跨机器/算力密集信号时才需要上多进程或消息队列分布式，项目里 [distributed.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/distributed.py) 是那套方案的预留实现。

---

## 六、一句话记忆卡

> **隔离靠新建 messages（[agents.py:240](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L240-L243)），传参靠最小信息包（[L410](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L410-L427)），回流只有 content（[L360](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L360-L372)），失控有五道闸（[delegation.py:115](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L115-L160)）。四件事缺一，就还是 Demo。**
