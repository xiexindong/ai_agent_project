# 面试03：多 Agent 三大技术方案（子图 / 分布式 / 生产四件套）

> 配套代码：`modules/multiagent/subgraph.py`（方案1+四件套）、`modules/multiagent/distributed.py`（方案2+四件套）、既有四件套在 `delegation.py:115-160` / `harness.py:416,840-845` / `agents.py:476-498`
> 一句话结论：**绝大多数企业场景用 LangGraph 子图（同进程、状态原生、断点续跑白送）；只有独立部署/跨机器/高并发才上消息队列分布式；而委派深度限制、子任务超时、失败降级、子结果冲突处理这四件套，是区分 Demo 和生产系统的分水岭——三套方案里都必须在场。**

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

### 实测（tests/test_subgraph_delegation.py 全过）

```python
orch = SubgraphOrchestrator()
state = orch.run("查营收；算环比")
# 运行结果:
# state["subtasks"] → 2 个（按 ；拆分）
# state["results"]  → ["[worker-1] 已完成: 查营收", "[worker-2] 已完成: 算环比"]
# state["final_answer"] → 两段产出以空行拼接
# state["degraded"] → []
```

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

### 实测（tests/test_distributed_sim.py 全过，含 Windows 真进程）

```python
# 基础往返 + 死信：
c.run([{"task_id": "t1", "instruction": "毒消息", "mode": "fail"}])
# 运行结果:
# broker_stats = {"published":1, "delivered":4, "acked":0,
#                 "redelivered":3, "dead_lettered":1}
#   → 首投 1 次 + 重投 3 次全部失败 → 第 4 次失败进死信
# degraded = [{"task_id": "t1", "reason": "等待结果超时(3.0s) 或已进死信"}]

# 进程隔离（use_processes=True）：
# results 里的 pid 与主进程 os.getpid() 不同 → 节点真的跑在独立进程里
```

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

三套实现里四件套的落点一览：

| 四件套 | 子图（subgraph.py） | 分布式（distributed.py） | 既有链路 |
|---|---|---|---|
| ① 深度限制 | `_spawn`: depth ≥ max_depth 拒绝 + 防环 chain 检查 | `_process_delegations`: child_depth > max_depth 拒 | `delegation.py:141` DelegationLedger.can |
| ② 子任务超时 | `_delegate_node`: future.result(timeout) | `collect` 的整体 deadline | `harness.py:840` FutureTimeout |
| ③ 失败降级 | 超时/异常进 degraded，答案只拼成功项并注明 | 结果缺失（超时/死信）进 degraded | `agents.py:476` 规划降级 / `:496` 汇总降级 |
| ④ 冲突处理 | reconcile：同组多数决，平票取先完成，少数派标 overridden | 冗余派发 k 份 → 多数决收敛 | `harness.py:540` Consensus→Arbitrator→HITL |

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

## 五、三套多 Agent 链路怎么共存

| 链路 | 文件 | 适用 |
|---|---|---|
| AgentHarness 线程池编排 | `multiagent/harness.py` | 花名册团队、DAG 分批并行、评审 |
| 消息层委派（Delegation） | `multiagent/delegation.py` | 执行者中途把活转给同伴 |
| 子图委派（Subgraph） | `multiagent/subgraph.py` | 图套图形态，状态/断点续跑原生 |
| 模拟分布式 | `multiagent/distributed.py` | 学习/验证消息队列架构，或真跨机器场景的雏形 |

四套并存按场景选：单体内协作走 harness/子图，跨机器才走消息队列。

---

## 六、简单总结

1. 选型顺序：**先子图**（同进程、成本低、断点续跑白送）→ 独立部署/跨机器/高并发才上**消息队列分布式**
2. 没有远程也能验证分布式：多进程 + 跨进程队列，模拟 topic 路由、ack、可见性超时重投、死信全套语义
3. 四件套（深度限制/超时/降级/冲突处理）在两套新实现与既有链路全部落地，各自的拒绝/降级/冲突记录**都留痕**——排查"为什么没人接/为什么用这个答案"全靠它们
4. 实测结论：`tests/test_subgraph_delegation.py` 5 项 + `tests/test_distributed_sim.py` 6 项全过，现有 94 项 multiagent 测试无回归
