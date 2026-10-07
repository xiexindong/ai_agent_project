# 面试03：多 Agent 三大技术方案（子图 / 分布式 / 生产四件套）

> 配套代码：`modules/multiagent/subgraph.py`（方案1+四件套）、`modules/multiagent/distributed.py`（方案2+四件套）、既有四件套在 `delegation.py:115-160` / `harness.py:416,840-845` / `agents.py:482-507,511-523`
> 一句话结论：**绝大多数企业场景用 LangGraph 子图（同进程、状态原生、断点续跑白送）；只有独立部署/跨机器/高并发才上消息队列分布式；而委派深度限制、子任务超时、失败降级、子结果冲突处理这四件套，是区分 Demo 和生产系统的分水岭——三套方案里都必须在场。**

---

## 〇、面试官视角与 30 秒回答

**面试官到底想问什么**：① 多 Agent 系统的技术选型判断力——什么时候该上重武器、什么时候是过度设计；② 你做的系统是 Demo 还是生产级——用"四件套"一试便知。

**30 秒回答模板**：

> "多 Agent 协同我先看部署形态：同一个应用内的协作，首选 LangGraph Subgraph——管理图的 delegate 节点里 invoke 一张编译好的工作图，状态是函数调用级别的传递，接上 checkpointer 断点续跑白送，开发成本最低。只有三个信号同时出现时——Agent 要独立部署、跨机器、高并发到单进程扛不住——才上消息队列分布式，把 invoke 换成'发消息到 topic、各节点进程消费、结果回传'。不管选哪条路，生产系统都绕不开四件套：委派深度限制防 A 转 B 转 A 无限套娃，子任务超时防一个慢任务拖死全局，失败降级保证 10 个子任务成 9 个也能交付，冲突处理在多个产出打架时按多数决收敛。这四件在 Demo 里全都没有，是分水岭。"

**术语首次出现的大白话**：

| 术语 | 大白话 |
|---|---|
| LangGraph | 把 Agent 的工作流程画成"流程图"再交给框架跑的库：每个节点是一个函数，节点之间的连线是执行顺序 |
| 子图（Subgraph） | 流程图里的一个节点本身又是一张完整的流程图——像公司组织架构：一个部门内部自己还有一套分工 |
| invoke | "调用并跑完"：把初始状态丢进图里，等它跑到终点拿最终状态回来 |
| checkpointer / 断点续跑 | 游戏存档：每过一个节点自动存一档，程序崩了重开能从上次存档继续，不用从头再来 |
| 消息队列 / topic | 邮局分拣：消息按"栏目"（topic）分桶，订阅了该栏目的人轮流取件 |
| ack / nack | 回执单：活干完了签 ack 销账；干砸了签 nack，邮局重新投递 |
| 可见性超时 | 快递的"签收时限"：投出去超过时限没收到回执，邮局认为丢了，重新投一件 |
| 死信队列（DLQ） | 多次投递都失败的"疑难件"专区：不再重投，单独堆起来等人处理 |
| at-least-once | "宁可送重不可漏送"的投递承诺：消息至少到达一次，代价是可能重复到达 |
| 幂等 | 同一条消息处理两次和处理一次效果一样（比如"设置状态为已完成"），配合 at-least-once 才安全 |
| HITL（人在回路） | 关键环节请真人按"批准/驳回"按钮，机器不擅自做主 |
| 多数决 | 三个人投票，两个说 A 一个说 B，采纳 A |

---

## 一、先说结论

| 面试问 | 一句话答 |
|---|---|
| 多 Agent 协同首选什么架构？ | LangGraph Subgraph：管理图节点里 invoke 编译好的工作图，开发成本最低 |
| 什么时候才上 Kafka/RabbitMQ？ | Agent 需要独立部署、跨机器、高并发大型平台时；单体应用硬上是过度设计 |
| 没有远程机器怎么验证分布式架构？ | 多进程 + 跨进程队列模拟消息代理全套语义（topic 路由/ack/重投/死信） |
| Demo 和生产系统的差距在哪？ | 四件套：深度限制、子任务超时、失败降级、冲突处理——Demo 全都没有 |

---

## 二、方案 1：LangGraph Subgraph 子图委派（首选）

### 一句话看破

子图委派 = 管理图的 delegate 节点里调一句 `worker_graph.invoke(child_state)`。没有魔法，就是函数调用，只是被调的"函数"是一张编译好的 LangGraph 图。

### 图结构（subgraph.py）

```
管理图:  plan ──→ delegate ──→ reconcile ──→ answer ──→ END
                      │
                      ├─ 线程池并行 invoke N 张工作图（每子任务一张）
                      └─ future.result(timeout) 超时降级

工作图:  execute ──→ (delegate? ──→ execute 循环) ──→ report ──→ END
                       └ delegate 节点里再 invoke 更深一层子图（递归，深度闸拦截）
```

### 状态映射：父图字段 ↔ 子图字段

| 父图（ManagerStateDict） | 子图（WorkerStateDict） | 说明 |
|---|---|---|
| subtasks[i].task_id | task_id | 直接映射 |
| subtasks[i].assignee | assignee + chain | 子图知道自己是谁、从哪来 |
| — | depth | 第几层子图（管理图直接子图 = 第 1 层） |
| results[i].content | report | 子图终态的 report 映射回父图结果 |

映射的代码就这几行（subgraph.py:287-306，`_invoke_worker`）：

```python
child_state: WorkerStateDict = {
    "task_id": task["task_id"],
    "assignee": assignee,
    "instruction": task.get("instruction", ""),
    "depth": int(task.get("depth", 1)),
    "chain": [assignee],
    ...
}
worker = build_worker_graph(assignee, execute_fn=..., delegate_fn=self._spawn, ...)
final = worker.invoke(dict(child_state))       # 子图跑完，终态的 report 映射回父图
return {"task_id": ..., "content": final.get("report", ""), "success": True, ...}
```

### 输入 → 处理 → 输出对照（tests/test_subgraph_delegation.py 全过）

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| 起点 | `orch.run("查营收；算环比")` | plan 节点按 "；" 拆任务 | `state["subtasks"]` → 2 个 |
| 并行委派 | 2 个子任务 | 线程池各 invoke 一张工作图（subgraph.py:263-264） | 每张图终态带回 report |
| 汇总 | 2 份 report | reconcile + answer 节点拼接 | `state["results"]` → `["[worker-1] 已完成: 查营收", "[worker-2] 已完成: 算环比"]` |
| 终态 | — | 成功项以空行拼接成最终答案 | `state["final_answer"]` 两段产出；`state["degraded"]` → `[]` |

### 为什么优先选子图（不这么写会怎样）

| 做法 | 原因 |
|---|---|
| 同进程 invoke 子图 | 跨进程/跨网络调 Agent 要序列化状态、处理网络故障、对账重试——开发成本翻几倍，单体场景全是白付 |
| 子图自带状态 schema | 共用一个全局 state 的话，N 个子任务互相覆盖字段，Debug 时分不清哪条数据是谁写的 |
| 接上 checkpointer | 子图状态跟父图一起被序列化，断点续跑白送；消息队列方案要做"任务状态对账"才能续跑 |
| planner/execute_fn 注入 | 骨架与业务解耦：测试递确定性函数（毫秒级），生产递 ReActAgent 执行循环 |

---

## 三、方案 2：本地模拟分布式（消息队列架构）

### 什么时候才需要它

| 信号 | 说明 |
|---|---|
| 独立部署 | 不同 Agent 团队各自发版、各自扩缩容，不能捆在一个进程里 |
| 跨机器 | Agent 要用本机的 GPU/数据库/内网权限，物理上就得分布 |
| 高并发 | 单进程线程池到顶了，需要横向加节点 |

没有这些信号就硬上 Kafka：多一跳网络、多一套运维、多一套失败模式，换来的只有简历关键词。

### 模拟架构（distributed.py）

```
主进程                                    每个 Agent 节点 = 一个独立进程/线程
┌────────────────────────────┐          ┌──────────────────────────┐
│ DistributedCoordinator      │          │ _node_main 循环           │
│  ├─ SimulatedBroker(线程)   │ 任务队列→ │  取消息 → 执行 → 发结果    │
│  │   路由/重投/死信/统计     │ ←结果队列 │  → ack/nack 回执          │
│  └─ collect/冲突/降级       │ ←ack队列  │                          │
└────────────────────────────┘          └──────────────────────────┘
```

### 模拟的消息代理语义（换真 Kafka/RabbitMQ 一一对应）

| 本模块 | 真实中间件 |
|---|---|
| topic + 订阅者轮询投递 | 分区/消费组负载均衡 |
| ack 回执（成功才销账） | manual ack 模式 |
| 可见性超时（投出没 ack → 重投） | unacked 消息重新入队 |
| 重投 max_retries 次进死信 | 死信队列（DLQ） |
| 未 ack 一定重投 | at-least-once 投递语义 |

重投与死信的核心代码（distributed.py:240-258，`_route`）：

```python
def _route(self, topic: str, msg: dict, first: bool) -> None:
    with self._lock:
        if not first:
            if msg["retries"] >= self._max_retries:   # 重投额度耗尽
                self._stats["dead_lettered"] += 1
                self._dlq.append(dict(msg))           # 进死信，不再打扰节点
                return
            msg["retries"] += 1                        # 否则重投计数 +1
        ...
        subs[idx].put(msg)                             # 轮询投给该 topic 的下一个订阅者
        self._inflight[msg["tag"]] = {"msg": msg, "ts": time.time()}  # 挂到在途账，等 ack
```

### 输入 → 处理 → 输出对照（tests/test_distributed_sim.py 全过，含 Windows 真进程）

| 场景 | 输入 | 处理 | 输出 |
|---|---|---|---|
| 正常往返 | 1 条任务消息 | 路由→节点执行→ack 销账 | `broker_stats = {"published":1, "delivered":1, "acked":1, ...}` |
| 毒消息 | `{"task_id":"t1", "instruction":"毒消息", "mode":"fail"}` | 首投失败→重投 3 次全失败→第 4 次进死信 | `{"published":1, "delivered":4, "acked":0, "redelivered":3, "dead_lettered":1}`；`degraded = [{"task_id":"t1", "reason":"等待结果超时(3.0s) 或已进死信"}]` |
| 进程隔离 | `use_processes=True` | 节点跑在独立子进程 | results 里的 pid 与主进程 `os.getpid()` 不同 → 真的跨进程 |

### 与子图方案的取舍对照

| | 子图（方案1） | 模拟分布式（方案2） |
|---|---|---|
| 部署形态 | 同进程 | 每节点独立进程 → 可跨机器 |
| 状态传递 | 函数调用返回值 | 消息序列化过队列 |
| 断点续跑 | checkpointer 白送 | 要自建任务状态对账 |
| 节点故障 | 进程崩 = 全崩 | 死信/重投/换节点接管 |
| 开发成本 | 低 | 高（序列化/幂等/对账） |

---

## 四、方案 3：生产四件套（分水岭）

三套实现里四件套的落点一览（锚点均已核实）：

| 四件套 | 子图（subgraph.py） | 分布式（distributed.py） | 既有链路 |
|---|---|---|---|
| ① 深度限制 | `_spawn`（:323-330）: depth ≥ max_depth 拒绝 + 防环 chain 检查 | `_process_delegations`（:400）: child_depth > max_depth 拒 | `delegation.py:141` DelegationLedger.can |
| ② 子任务超时 | `_delegate_node`（:257-281）: future.result(timeout) | `collect`（:347）的整体 deadline | `harness.py:416` 绝对 deadline + `:843` FutureTimeout |
| ③ 失败降级 | 超时/异常进 degraded，答案只拼成功项并注明 | 结果缺失（超时/死信）进 degraded | `agents.py:482-507` 规划降级（解析失败重试一次→仍失败降为单 Agent 直答）/ `:511-523` 汇总降级（汇总失败回退拼接成功项） |
| ④ 冲突处理 | reconcile（:368）：同组多数决，平票取先完成，少数派标 overridden | 冗余派发 k 份 → 多数决收敛 | `harness.py:540` Consensus→Arbitrator→HITL |

四件套①的真实代码（subgraph.py:323-330，防环+深度闸）：

```python
def _spawn(self, task_id, to_agent, depth, chain):
    if not to_agent:
        return None
    if to_agent in (chain or []):          # 防环：链上出现过的名字直接拒（A→B→A 甩锅）
        return None
    if depth >= self.config.max_depth:     # 深度闸：第 max_depth 层不许再转
        return None
```

### 实测数据（四件套逐一触发）

```python
# ① 深度限制（max_depth=2）：
#   第 1 层 worker-1 委派 → 第 2 层放行（delegations +1）
#   第 2 层再委派 → denied = [{"reason": "深度/防环治理拒绝"}]

# ② 超时（timeout=0.3s，执行器 sleep 2s）：
#   degraded = [{"task_id": "t2", "reason": "子任务超时(0.3s)"}]

# ③ 降级（全部子任务抛异常）：
#   final_answer = "（全部子任务失败，降级为直接应答）请稍后重试或缩小问题范围。"
#   → 不抛异常给用户，一定给个兜底话术

# ④ 冲突（同组 A/A/B 三条产出）：
#   conflicts = [{"group": "revenue", "winner": "营收结论: A", "rule": "多数决"}]
#   少数派 B 的记录带 resolution.overridden=True（留痕，不静默丢）
```

### 为什么四件套是分水岭（不写会怎样）

| 不写 | 后果 |
|---|---|
| 深度限制 | Agent A 转 B、B 转 A，互相甩锅直到烧光 token/步数预算，日志里还"都在正常工作" |
| 子任务超时 | 一个下游卡 5 分钟，整条链路陪葬，用户端表现为"永远转圈" |
| 失败降级 | 任一子任务异常 → 整单失败；有降级 → 10 个子任务成 9 个也交付 9 个并注明 |
| 冲突处理 | 冗余/多 Agent 产出矛盾时随便挑一个，用户拿到错误答案还无从追溯 |

---

## 五、关键设计对照表：方案 A / B / C 怎么选

| | 方案 A：单 Agent 直答 | 方案 B：子图委派（本项目首选） | 方案 C：消息队列分布式 |
|---|---|---|---|
| 部署形态 | 单进程单 Agent | 同进程，管理图嵌工作图 | 每节点独立进程，可跨机器 |
| 并行能力 | 无（串行） | 线程池并行 invoke 子图 | 横向加节点，理论上无限 |
| 状态/断点续跑 | 无需 | checkpointer 白送 | 自建任务状态对账 |
| 失败域 | 一处崩全单崩 | 子任务超时/异常可降级 | 节点崩可重投、换节点接管 |
| 运维成本 | 零 | 低（还是一个进程） | 高（broker 集群、监控、对账） |
| 适用 | 简单问答 | 绝大多数企业场景 | 独立部署/跨机器/高并发平台 |

**选型决策树**：需要跨机器或独立发版？→ 否：方案 B；是：再看病死节点要不要自动接管 → 要：方案 C。拿不准就先 B，B 的"管理图/工作图"拆分思路迁到 C 时不浪费。

---

## 六、三套多 Agent 链路怎么共存

| 链路 | 文件 | 适用 |
|---|---|---|
| AgentHarness 线程池编排 | `multiagent/harness.py` | 花名册团队、DAG 分批并行、评审 |
| 消息层委派（Delegation） | `multiagent/delegation.py` | 执行者中途把活转给同伴 |
| 子图委派（Subgraph） | `multiagent/subgraph.py` | 图套图形态，状态/断点续跑原生 |
| 模拟分布式 | `multiagent/distributed.py` | 学习/验证消息队列架构，或真跨机器场景的雏形 |

四套并存按场景选：单体内协作走 harness/子图，跨机器才走消息队列。

---

## 七、面试话术（可直接背）

> "多 Agent 方案我按部署形态分档：同进程协作用 LangGraph 子图——它本质是函数调用，状态传递、断点续跑都是原生能力；独立部署、跨机器、高并发三个信号同时出现才上消息队列，把 invoke 换成 topic 投递 + ack 回执。没有远程机器时，我用多进程加跨进程队列把 topic 路由、可见性超时重投、死信这些语义在本地完整模拟一遍，架构验证不依赖外部环境。而真正区分 Demo 和生产的是四件套：深度限制防互相甩锅，超时防一颗老鼠屎，降级保证部分交付，冲突处理防随便挑一个错误答案——这四个缺一个，上线都是赌博。"

---

## 八、高频追问（含回答要点）

**Q1：子图里"递归委派"会不会无限套娃？**
- 要点：`_spawn` 两道闸（subgraph.py:327-330）：委派链防环（A→B→A 直接拒）+ depth ≥ max_depth 拒；
- 默认 max_depth=2：管理图的子图是第 1 层，它最多再委派一层，第 2 层的 delegate 请求全被拒并记入 denied 留痕；
- 既有链路同款逻辑在 DelegationLedger.can（delegation.py:136-142），还多了单人配额和总次数上限。

**Q2：超时用 future.result(timeout)，线程本身杀不掉怎么办？**
- 要点：超时的语义是"不等了"，不是"杀掉"——`fut.cancel()` 只对还没开始排队的任务生效，跑着的线程会继续跑完但结果被丢弃进 degraded；
- 防"僵尸线程堆积"靠外层预算（harness 总 deadline，harness.py:416）和子任务数的天然上限；
- 面试加分点：说明 Python 线程无法安全强杀，工程上用"结果作废 + 配额拦截"代替强杀。

**Q3：模拟分布式的 at-least-once 会不会导致同一任务被执行两次？**
- 要点：会——可见性超时后重投，而原节点可能其实正在执行（慢而不是死），于是两个节点各跑一遍；
- 真实系统靠消费端幂等消化（同一 task_id 执行两次结果相同），本项目测试里用确定性执行器天然幂等；
- 这是 at-least-once 的固有代价，面试要主动说出"投递语义选的是宁可重复不可漏"。

**Q4：冲突处理"多数决"平票怎么办？少数派直接丢吗？**
- 要点：子图 reconcile（subgraph.py:368）平票时取**先完成**的那条（先答出来的先赢）；
- 少数派不静默丢：记录带 `resolution.overridden=True` 留痕，事后能查"为什么没用 B 的答案"；
- 既有链路的更完整版本是 harness.py:540 的 Consensus（相似度判冲突）→ Arbitrator（裁决）→ HITL（人裁决）三级。

**Q5：规划失败、汇总失败分别怎么降级？**
- 要点：规划失败（agents.py:482-507）：先把校验错误原文回喂重试一次，仍失败 → 降级为单任务直答，不让整单报废；
- 汇总失败（agents.py:511-523）：LLM 汇总挂了就回退成"成功项原文拼接"，任一成员产出为空也如实标注，不替成员编内容；
- 原则一句话：每一环失败都有"次优但诚实"的退路，绝不对用户抛栈。

**Q6：为什么说消息队列方案"断点续跑要自建对账"？**
- 要点：子图方案的 state 在进程内，checkpointer 序列化即存档；
- 分布式里任务状态散落在三处：broker 的在途账、节点进程内存、主控的 collect 清单——进程一崩三方口径不一；
- 续跑前要先把"发出去了没？执行了没？结果回了没？"对齐，这就是对账，消息队列方案的最大隐性成本。

---

## 九、简单总结

1. 选型顺序：**先子图**（同进程、成本低、断点续跑白送）→ 独立部署/跨机器/高并发才上**消息队列分布式**
2. 没有远程也能验证分布式：多进程 + 跨进程队列，模拟 topic 路由、ack、可见性超时重投、死信全套语义
3. 四件套（深度限制/超时/降级/冲突处理）在两套新实现与既有链路全部落地，各自的拒绝/降级/冲突记录**都留痕**——排查"为什么没人接/为什么用这个答案"全靠它们
4. 实测结论：`tests/test_subgraph_delegation.py` 5 项 + `tests/test_distributed_sim.py` 6 项全过，现有 94 项 multiagent 测试无回归
