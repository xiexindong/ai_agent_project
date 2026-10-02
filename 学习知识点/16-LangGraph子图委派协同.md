# 16 - LangGraph Subgraph 子图做多 Agent 委派协同

> 对应代码：`modules/multiagent/subgraph.py`（编排器 + 生产四件套）
> 测试：`tests/test_subgraph_delegation.py`（`python -m pytest tests/test_subgraph_delegation.py -q`）

---

## ① 一句话结论

子图委派就是：管理图（Manager Graph）的一个节点里，调一句 `worker_graph.invoke(child_state)`——被调的不是普通函数，而是一张编译好的 LangGraph 图；`child_state` 由父图状态映射而来，子图跑完的终态再映射回父图。**没有魔法，就是函数调用，只是"被调的函数"自带状态流转、条件分支和断点续跑能力。**

打个比方：主管（管理图）接到大活，不自己干，而是把活拆成几份，每份装进一个"工单袋"（child_state）递给工人（工作图）；工人干完，把"交付单"（终态里的 `report`）塞回袋子还给主管；主管收回所有交付单，对账（reconcile）后汇总交差。

---

## ② 为什么企业场景优先用它（而不是 Kafka/RabbitMQ）

| 维度 | LangGraph 子图（本模块） | 分布式多 Agent（Kafka/RabbitMQ） |
|---|---|---|
| 部署形态 | 同一个 Python 进程里"图套图" | 多台机器、多进程，靠消息队列传话 |
| 开发成本 | 低：定义两个 State schema + 几个节点函数 | 高：队列、序列化、重试、幂等全套都要做 |
| 状态传递 | 进程内传 dict，零序列化开销 | 消息必须序列化成 JSON/字节流过网络 |
| 断点续跑 | 接上 checkpointer 原生支持 | 要自己做状态持久化与恢复 |
| 故障半径 | 一个子任务崩，线程池里 catch 住就完事 | 一个消费者挂了，消息可能重复消费或丢失 |
| 适用规模 | 单机、绝大多数企业业务 | 跨机器、高并发、独立部署的大型平台 |

**判断口诀**：Agent 不需要独立部署、不跨机器，就用子图；只有真到了"每个 Agent 是独立服务、要水平扩容"的规模，才上消息队列。本项目两者都有实现（`subgraph.py` 与 `distributed.py`），按场景选。

---

## ③ 图结构：管理图套工作图

```
管理图（Manager Graph）
    plan ──→ delegate ──→ reconcile ──→ answer ──→ END
   拆子任务   并行调子图    冲突收敛      拼最终答案

delegate 节点内部（对每个子任务）：
    父图状态 --状态映射--> child_state
    child_state --worker_graph.invoke()--> 子图终态
    子图终态.report --映射回--> results[i].content

工作图（Worker Graph，子图）
    execute ──→ delegate? ──→ report ──→ END
        ↑          │
        └──────────┘  （execute 提出委派 → delegate 递归 invoke 更深一层子图 →
                        产出并入 context → 回到 execute 消化 → 循环上限 3 轮）
```

两张图各有一份状态 schema（这是子图委派区别于普通函数调用的关键）：

| 管理图状态 `ManagerStateDict` | 装什么 | 工作图状态 `WorkerStateDict` | 装什么 |
|---|---|---|---|
| `query` | 原始问题 | `task_id` / `assignee` / `instruction` | 父图映射进来的任务 |
| `subtasks` | plan 拆出的子任务列表 | `depth` / `chain` | 第几层子图 + 委派链（防环） |
| `results` | delegate 收回的成功产出 | `context` / `report` | 上下文 + 交付正文 |
| `degraded` | 超时/异常的降级记录 | `delegate_request` | execute 想转出去的请求 |
| `conflicts` | reconcile 检出的冲突 | `delegations` / `denied` | 本层转出去/被拦的账本 |
| `final_answer` | 最终答案 | `iteration` | execute↔delegate 循环轮数 |

---

## ④ 大白话拆解：委派到底发生在哪几行

| 部分 | 大白话解释 |
|---|---|
| `build_worker_graph(name, execute_fn, delegate_fn)` | 造一张工作图；干什么活由外部递进来的函数决定（测试递确定性函数，生产递 ReActAgent 执行循环） |
| `workflow.compile()` | 把图编译成可 invoke 的可执行对象（`build_worker_graph` 的返回值） |
| `worker.invoke(dict(child_state))` | **子图委派的全部本体**：把映射好的子状态喂给子图，同步等它跑完，返回终态 dict |
| `final.get("report", "")` | 从子图终态里取交付正文，塞进父图的 `results[i]["content"]` |
| `delegate_fn=self._spawn` | 工作图自己想再往下转时走的门；`_spawn` 里再 `invoke` 更深一层子图——递归就发生在这 |
| `depth + 1` | 每转一层深度加一；深度闸靠这个数拒绝无限套娃 |

### 输入 → 输出对照（真实数值，默认配置实测）

```python
from modules.multiagent.subgraph import SubgraphOrchestrator

orch = SubgraphOrchestrator()          # 全默认：max_depth=2, task_timeout=15.0,
                                       #          max_subtasks=5, worker_max_iterations=3
state = orch.run("查营收；算环比")      # 默认规划器按 ；拆成 2 个子任务
```

执行后（实测 `2026-10-02`，日志：`[plan] 拆出 2 个子任务` → `[delegate] 成功 2 | 降级 0`）：

```python
state["subtasks"] == [
    {"task_id": "t1", "assignee": "worker-1", "instruction": "查营收", "group": "", "depth": 1},
    {"task_id": "t2", "assignee": "worker-2", "instruction": "算环比", "group": "", "depth": 1},
]
state["results"] == [
    {"task_id": "t1", "assignee": "worker-1", "group": "",
     "content": "[worker-1] 已完成: 查营收", "success": True,
     "delegations": [], "denied": []},
    {"task_id": "t2", "assignee": "worker-2", "group": "",
     "content": "[worker-2] 已完成: 算环比", "success": True,
     "delegations": [], "denied": []},
]
state["final_answer"] == "[worker-1] [worker-1] 已完成: 查营收\n\n[worker-2] [worker-2] 已完成: 算环比"
state["degraded"]  == []          # 无超时无异常
state["conflicts"] == []          # 两个任务 group 都是空串，不参与冲突判定
state["finished"]  == True
```

`final_answer` 里出现两次 `[worker-1]`：一次是 `content` 本身自带前缀（默认执行器 `_default_execute` 生成 `"[worker-1] 已完成: 查营收"`），一次是 `answer` 节点拼接时又加了一层 `f"[{assignee}] {content}"`。生产里把执行器换成真 LLM 后，`content` 不会再自带前缀，就不会重复。

### 状态映射的具体字段对位

```
父图 subtasks[i]                子图 child_state
─────────────────               ─────────────────
task_id        "t1"      →      task_id       "t1"
assignee       "worker-1" →     assignee      "worker-1"
instruction    "查营收"   →      instruction   "查营收"
depth          1         →      depth         1
（新造）                 →      chain         ["worker-1"]
（新造）                 →      iteration     0
```

反方向只映射一个字段：子图终态的 `report` → 父图 `results[i]["content"]`。**映射是手工的、白纸黑字的**——父图看不到子图的 `context`/`iteration` 等内部字段，这正是"子图有自己的状态 schema"的含义：隔离是靠映射边界实现的，不是靠什么框架魔法。

---

## ⑤ 生产四件套：Demo 与生产的分水岭

四件套全部在 `subgraph.py` 里落地，边界值集中在 `SubgraphConfig`：

```python
SubgraphConfig(
    max_depth=2,            # 子图最多嵌套几层
    task_timeout=15.0,      # 单个子任务（一次子图 invoke）超时秒数
    max_subtasks=5,         # 单次运行最多拆几个子任务
    worker_max_iterations=3,# 单张工作图内 execute↔delegate 最多循环几轮
    max_workers=4,          # 子任务并行执行的线程数
)
```

### 四件套① 委派深度限制 + 防环（`_spawn`）

| 闸门 | 规则 | 不这么写会怎样 |
|---|---|---|
| 深度闸 | `depth >= max_depth` 直接返回 `None` | worker-a 委派 worker-b、worker-b 再委派 worker-c……无限套娃直到栈溢出或耗光预算 |
| 防环闸 | `to_agent in chain` 直接返回 `None` | A→B→A 互相甩锅，两个子图都"在正常工作"，日志看不出异常，直到超时才停 |

输入 → 输出对照（来自 `test_depth_limit_denies_third_layer`，`max_depth=2`）：

```python
# worker-1（第 1 层）execute 后提出委派 → worker-1-child（第 2 层）放行
# worker-1-child 再提出委派 → depth(2) >= max_depth(2) → 拒绝

state["results"][0]["delegations"] == [   # 第 1 层成功转出 1 次
    {"from": "worker-1", "to": "worker-1-child", "task_id": "t1>worker-1-child", ...}
]
state["results"][0]["denied"] == [        # 第 2 层的再委派被拦，留痕
    {"from": "worker-1-child", "to": "worker-1-child-child",
     "depth": 2, "reason": "深度/防环治理拒绝"}
]
```

**被拒也要留痕**：`denied` 记录"想转但被拦"这件事。排查"为什么这块没人接"时，"根本没想转"和"想转但被治理拦了"是两回事，只留一个失败状态事后只能靠猜。

### 四件套② 子任务超时（`_delegate_node`）

做法：`ThreadPoolExecutor` 并行 submit 所有子任务，逐个 `fut.result(timeout=task_timeout)`。

| 设计点 | 原因 |
|---|---|
| 线程池并行而不是串行 invoke | 串行时 3 个 10 秒任务要等 30 秒；并行总耗时 ≈ 最慢那个 |
| 逐个 `result(timeout=)` 而不是各任务单独计时 | futures 已在并行跑，逐个等待是重叠的，总等待 ≈ 最慢那个，不是各任务超时之和 |
| 超时后 `fut.cancel()` + 记 `degraded` | 不这么写：一个慢任务拖死全局，用户等 15 秒×N 还拿不到部分答案 |

输入 → 输出对照（来自 `test_subtask_timeout_degrades`，`task_timeout=0.3`，慢任务 `sleep(2.0)`）：

```python
state = orch.run("快任务；慢任务")

state["results"]  == [   # 只有快任务成功
    {"task_id": "t1", ..., "content": "[worker-1] 已完成: 快任务", "success": True, ...}
]
state["degraded"] == [{"task_id": "t2", "reason": "子任务超时(0.3s)"}]
```

### 四件套③ 失败降级（`_answer_node`）

规则：最终答案只拼 `success=True` 的产出；降级项和冲突收敛以附注形式追加在末尾；**全部失败时给兜底话术，不抛异常**。

| 场景 | final_answer 长什么样 |
|---|---|
| 全部成功 | 各产出以 `\n\n` 拼接，无附注 |
| 部分超时（上例） | `"[worker-1] 已完成: 快任务\n\n（降级说明: t2 子任务超时(0.3s)）"` |
| 全部失败 | `"（全部子任务失败，降级为直接应答）请稍后重试或缩小问题范围。"` |

不这么写会怎样：静默丢弃失败子任务 → 用户拿到的答案少了半截却毫无提示；或者直接抛异常 → 整个管理图崩掉，连已经成功的那部分产出也一起赔进去。

### 四件套④ 子结果冲突处理（`_reconcile_node`）

规则：同 `group` 的多条产出内容不一致时 → 多数决；平票取先完成（`contents[0]`）；冲突留痕（`conflicts` + 每条记录的 `resolution.overridden`）。

输入 → 输出对照（来自 `test_conflict_majority_resolution`，三个 worker 同 `group="revenue"`，产出 A/A/B）：

```python
state["conflicts"] == [
    {"group": "revenue",
     "variants": ["营收结论: A", "营收结论: B", "营收结论: A"],
     "winner": "营收结论: A",
     "rule": "多数决"}
]
# worker-2（产出 B）的记录被标：
state["results"][1]["resolution"] == {"winner": "营收结论: A", "rule": "多数决", "overridden": True}
# 多数派两条 overridden=False
# final_answer 末尾追加 "（冲突收敛: 组 revenue 按「多数决」取 营收结论: A）"
```

为什么要收敛：同一问题问了三个 Agent，拿到 A/A/B 三个答案，直接全拼进最终答案等于让用户自己仲裁。多数决是成本最低的机器仲裁；平票（A/B 两票对两票）时没有多数，取先完成的那条（`contents[0]`，并行下即最先返回者）。

---

## ⑥ 委派产出怎么回流（execute ↔ delegate 循环）

工作图内部是一个小循环，配 `worker_max_iterations=3` 的轮数闸：

```
execute ──(填 delegate_request)──→ delegate ──(invoke 更深一层子图)──→ execute ──→ report
                ↑                                                      │
                └──── iteration >= 3 强制收尾，不许再委派 ←─────────────┘
```

| 步骤 | 发生什么 |
|---|---|
| 1. execute 想转 | 返回 `{"delegate_request": {"to": ..., "instruction": ...}}` |
| 2. delegate 转出 | 调 `_spawn`（过深度/防环闸）→ invoke 受托方子图 → 拿到 `rec` |
| 3. 产出并入上下文 | `context` 追加 `"\n[委派产出:xxx] ..."`，execute 下一轮能看到并用自己的逻辑消化 |
| 4. 请求清空 | `delegate_request: None`——防止循环回来把同一个请求再委派一遍 |
| 5. 账本冒泡 | 受托方自己转出去的记录（`nested_delegations`/`nested_denied`）并入本层账本——不冒泡的话，第 3 层"为什么没人接"在父层日志里彻底消失 |

不设轮数闸会怎样：一个固执的 execute 每轮都想委派、每次都被拒，图就绕圈直到步数耗尽——表现是"一直在跑但永远不出结果"。

---

## ⑦ 与另外两套多 Agent 方案的分工

| 模块 | 形态 | 解决什么 |
|---|---|---|
| `subgraph.py`（本文） | 图套图，同进程 | 开发成本最低的多 Agent 委派；状态、断点续跑原生支持 |
| `delegation.py` | 花名册协作的消息层委派 | 执行者事中把"不该我干的活"转给同伴，五道闸治理（开关/资格/存在性/边界/防环） |
| `distributed.py` | 本地模拟分布式 | 跨进程/跨机器的消息队列形态，为真上 Kafka/RabbitMQ 做演练 |

三套并存，按场景选：业务在单机、要快速落地 → 子图；Agent 之间要横向转派 → delegation；Agent 要独立部署扩容 → distributed。

---

## ⑧ 装配到生产（骨架不动，只换两个函数）

```python
orch = SubgraphOrchestrator(
    planner=主管Agent的规划调用,        # 默认是"按 ；切分"，生产换成 LLM 拆任务
    worker_execute=ReActAgent执行循环,  # 默认是确定性模拟，生产换成真 Agent
)
state = orch.run("用户的复合问题")
```

`execute_fn` / `delegate_fn` 都是注入的，这是本模块只管"图套图"骨架的原因：测试递确定性函数（上面的实测值就是这么来的，不烧一次 LLM 调用），生产递真执行循环，四件套的边界行为两种情况下完全一致。

---

## 简单总结

1. **子图委派 = 一句 `worker_graph.invoke(child_state)`**，被调的是编译好的 LangGraph 图；父子图之间靠手工状态映射隔离（`task_id/assignee/instruction/depth` 进去，`report` 出来）。
2. **四件套的边界值**：`max_depth=2`（第 2 层不许再转）、`task_timeout=15.0s`（实测 0.3s 配置下慢任务 2s 被 degrade）、`max_subtasks=5`（超出截断并记事件）、`worker_max_iterations=3`（防 execute 绕圈）。
3. **三个坑位必须留痕**：被拒的委派进 `denied`（区分"没想转"和"想转被拦"）、超时异常进 `degraded`（最终答案附降级说明，不静默丢）、冲突收敛进 `conflicts`（多数决 A/A/B 取 A，少数派标 `overridden=True`）。
4. **选型口诀**：单机业务优先子图（本模块）；只有"独立部署、跨机器、水平扩容"才上消息队列（`distributed.py`）。
