# LangGraph 多 Agent vs 手写多 Agent 完整对比

> 本文档对比两种多 Agent 实现方式，帮助你理解各自的适用场景。

---

## 一、核心区别总览

| 维度 | 手写 Python（你的项目） | LangGraph 框架 |
|---|---|---|
| **代码量** | 多（自己管线程、上下文、依赖） | 少（框架管状态流转） |
| **上下文隔离** | **强**（新建 messages） | 弱（共享 State） |
| **状态持久化** | 弱（需自己实现） | **强**（内置 checkpoint） |
| **可视化** | 无 | **强**（可生成 Mermaid 图） |
| **调试** | 容易（每行代码可见） | 中（框架黑盒） |
| **扩展性** | **强**（想加什么直接写） | 中（受框架约束） |
| **性能** | **高**（无框架开销） | 中（有 overhead） |
| **学习成本** | 低（Python 基础即可） | 高（要学 LangGraph 概念） |
| **异步支持** | 需自己实现 | **强**（原生支持 async/await） |
| **适用场景** | 一次性问答、严格隔离、精细控制 | 复杂状态管理、断点续跑、可视化 |

---

## 二、架构对比图

### 手写版本（你的项目）

```
用户提问
   │
   ▼
┌─────────────────────────────────────────┐
│ AgentHarness.run()                      │
│  - 主管拆任务（SupervisorAgent.plan()）   │
│  - 拓扑分批（topo_sort）                 │
│  - 线程池并行（ThreadPoolExecutor）       │
│  - 汇总（synthesize）                   │
│  - 审核（review）                       │
└─────────────────────────────────────────┘
   │
   ▼
WorkerAgent.run()
  ├─ 新建 messages（隔离）
  ├─ 调 LLM / 工具
  └─ 返回 AgentOutput
```

### LangGraph 版本

```
用户提问
   │
   ▼
┌─────────────────────────────────────────┐
│ StateGraph (TeamState)                  │
│  ├─ supervisor_node（拆任务）            │
│  ├─ researcher_node（查资料）            │
│  ├─ analyst_node（算数据）               │
│  ├─ writer_node（写报告）                │
│  ├─ synthesizer_node（汇总）             │
│  └─ reviewer_node（审核）                │
│                                         │
│  边：                                    │
│  supervisor → researcher → supervisor   │
│  supervisor → analyst → supervisor      │
│  supervisor → writer → supervisor       │
│  supervisor → synthesizer → reviewer    │
└─────────────────────────────────────────┘
   │
   ▼
Worker 子图（Subgraph）
  ├─ prepare（组装上下文）
  ├─ execute（调 LLM）
  └─ finalize（整理结果）
```

---

## 三、核心代码对比

### 3.1 上下文隔离

**手写版本**（[agents.py:240-243](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L240-L243)）：

```python
# 每次执行新建 messages，彻底隔离
messages = [
    {"role": "system", "content": self.system_prompt},
    {"role": "user", "content": prompt},
]
```

**LangGraph 版本**（[langgraph_multiagent.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/langgraph_multiagent.py)）：

```python
# 所有节点共享 TeamState
class TeamState(TypedDict):
    query: str
    subtasks: list[dict]
    worker_outputs: dict[str, str]
    # ← 所有节点都能看到这个 State
```

**区别**：手写版本每个 Worker 独立，LangGraph 版本所有节点共享。

### 3.2 依赖管理

**手写版本**（[harness.py:836-851](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L836-L851)）：

```python
# 拓扑分批 + 线程池并行
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(self._execute, task, ctx): task for task in batch}
    for future in as_completed(futures):
        output = future.result()
        ctx.upstream[output.task_id] = output  # ← 写进"成果袋"
```

**LangGraph 版本**（[langgraph_multiagent.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/langgraph_multiagent.py)）：

```python
# 路由函数检查依赖
def route_after_supervisor(state: TeamState) -> str:
    for task in state["subtasks"]:
        if task["status"] == "pending":
            deps_met = all(
                dep in state["worker_outputs"] for dep in task["depends_on"]
            )
            if deps_met:
                return task["assignee"]
    return "synthesizer"
```

**区别**：手写版本用拓扑排序 + 线程池，LangGraph 版本用路由函数 + 条件边。

### 3.3 异步支持

**手写版本**：需要自己实现（`asyncio` + `aiohttp`）

**LangGraph 版本**（[langgraph_multiagent.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/langgraph_multiagent.py)）：

```python
# 原生支持异步
async def researcher_node_async(state: TeamState) -> dict:
    result = await worker_graph.ainvoke(worker_state)
    return {"worker_outputs": {**worker_outputs, task_id: output}}
```

**区别**：LangGraph 原生支持异步，手写版本需要自己包装。

### 3.4 状态持久化

**手写版本**：需要自己实现（保存到数据库/文件）

**LangGraph 版本**（[langgraph_multiagent.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/langgraph_multiagent.py)）：

```python
# 内置 checkpoint 支持
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 断点续跑
config = {"configurable": {"thread_id": "run_001"}}
state = app.get_state(config)
```

**区别**：LangGraph 内置 checkpoint，手写版本需要自己实现。

---

## 四、什么时候用哪个

### 用手写版本（你的项目）

| 场景 | 原因 |
|---|---|
| 需要严格上下文隔离 | 每个 Worker 新建 messages，不共享 State |
| 性能敏感 | 无框架 overhead，直接调 LLM |
| 需要精细控制 | 五道闸、消息总线治理、自定义路由 |
| 一次性问答 | 不需要断点续跑、状态持久化 |
| 团队熟悉 Python | 学习成本低，新人好接手 |

### 用 LangGraph 版本

| 场景 | 原因 |
|---|---|
| 需要断点续跑 | 内置 checkpoint，可恢复执行 |
| 需要可视化编排 | 可生成 Mermaid 图，直观展示流程 |
| 需要复杂状态管理 | 框架管理 State，不容易出错 |
| 需要异步支持 | 原生支持 async/await |
| 团队熟悉 LangChain | 生态一致，减少学习成本 |

---

## 五、面试话术

**如果面试官问："你为什么不用 LangGraph 的多 Agent？"**

> "三个原因：
> 1. **上下文隔离**：LangGraph 的子图共享 State，我需要每个 Worker 完全隔离，新建 messages 列表
> 2. **精细控制**：我有五道委派闸、消息总线治理、一致性检测、仲裁，这些自定义逻辑用框架反而更绕
> 3. **性能**：手写代码直接调 LLM，无框架 overhead
>
> 但 LangGraph 也有优势：内置 checkpoint 支持断点续跑、原生支持异步、可生成可视化图。如果业务需要这些能力，我会考虑用 LangGraph。"

**如果面试官问："那你怎么实现多 Agent 的？"**

> "核心是三层：
> - **编排层**：`AgentHarness` 主管拆任务、拓扑分批、线程池并行、汇总答案
> - **执行层**：`WorkerAgent` 每个 Worker 独立上下文，有工具调工具，无工具直接调 LLM
> - **通信层**：`MessageBus` 消息总线 + `ctx.upstream` 上游产出透传
>
> 代码在 `modules/multiagent/` 目录，从 `agents.py` 的 `SupervisorAgent.plan()` 开始看。"

---

## 六、代码位置速查表

| 功能 | 手写版本 | LangGraph 版本 |
|---|---|---|
| 主管拆任务 | `modules/multiagent/agents.py:457-476` | `langgraph_multiagent.py` 的 `supervisor_node` |
| Worker 执行 | `modules/multiagent/agents.py:303-381` | `langgraph_multiagent.py` 的 `researcher_node` 等 |
| 上下文隔离 | `modules/multiagent/agents.py:240-243` | `langgraph_multiagent.py` 的 `TeamState` |
| 依赖管理 | `modules/multiagent/harness.py:836-851` | `langgraph_multiagent.py` 的 `route_after_supervisor` |
| 消息总线 | `modules/multiagent/coordination.py:132-144` | `langgraph_multiagent.py` 的 `MessageBus` |
| 异步支持 | 需自己实现 | `langgraph_multiagent.py` 的 `researcher_node_async` |
| 状态持久化 | 需自己实现 | `langgraph_multiagent.py` 的 `MemorySaver` |
| 子图（Subgraph） | 无 | `langgraph_multiagent.py` 的 `create_worker_subgraph` |

---

## 七、一句话总结

> 手写版本胜在**隔离性和性能**，LangGraph 胜在**状态管理和可视化**。你的项目选择手写，是因为业务需要严格的 Worker 隔离和精细的自定义控制。但如果需要断点续跑、异步支持、可视化编排，LangGraph 是更好的选择。
