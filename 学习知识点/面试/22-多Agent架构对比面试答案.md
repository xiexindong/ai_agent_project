# 面试 22：多 Agent 架构对比——三种模式怎么选

> 面试官给你一张图，上面有三种多 Agent 架构方案，问你怎么选、为什么。
> 本文是标准面试答案，按六段式组织：面试官视角 → 30秒电梯答案 → 原理+代码 → 面试话术 → 追问。

---

## 一、面试官视角

面试官想听的不是"哪个最好"，而是：

1. 你能不能说出三种模式的**本质区别**
2. 你能不能根据**业务场景**做选择，而不是技术偏好
3. 你知不知道每种模式的**代价和风险**

---

## 二、30 秒电梯答案

> 三种模式没有绝对优劣，按场景选：
>
> - **绝大多数业务**用**自研主管-Worker（模式 A）**：结构清晰、可控性强、token 相对省
> - **需要复杂多轮协作、状态持久化**才考虑 **LangGraph 编排（模式 B）**
> - **技术验证期、冷启动想少写代码**才用**现成框架（模式 C）**
>
> 关键不是选哪个，而是知道**什么时候不能用模式 A**。

---

## 三、原理讲解（通俗版）

### 3.1 三种模式的本质区别

| 模式 | 核心思想 | 类比 |
|---|---|---|
| **A 主管-Worker** | 一个中心大脑拆任务、派活、收口 | 餐厅：前台接单 → 后厨分工 → 传菜汇总 |
| **B 流水线** | 任务按固定流程流转，每个节点干一件事 | 工厂流水线：A 工位拧螺丝 → B 工位质检 → C 工位包装 |
| **C 群聊** | 去中心化，谁想发言谁发言 | 头脑风暴：大家围一桌，想到什么说什么 |

---

## 三、三种模式的架构图

### 模式 A：主管-Worker（Supervisor-Worker）

```
                    用户提问
                       │
                       ▼
              ┌─────────────────┐
              │   Supervisor     │
              │   （主管/项目经理）│
              │                 │
              │  拆任务 → 派活   │
              │  汇总 → 审核     │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Worker A │   │ Worker B │   │ Worker C │
   │ researcher│   │ analyst  │   │ reviewer │
   │ 查资料    │   │ 算数据   │   │ 挑毛病   │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Synthesizer    │
              │   （汇总答案）    │
              └─────────────────┘
```

**特点**：中心化管理，主管决定拆不拆、怎么拆、给谁做。Worker 只执行，不决策。

---

### 模式 B：流水线（Pipeline / DAG）

```
用户提问
   │
   ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ 检索员   │ → │ 分析师   │ → │ 撰写员   │ → │ 审核员   │
│ 查资料   │    │ 算数据   │    │ 写报告   │    │ 挑毛病   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
   │              │              │              │
   └──────────────┴──────────────┴──────────────┘
                       │
                       ▼
                  最终答案
```

**特点**：严格串行，前一步的输出是后一步的输入。适合步骤有严格依赖的场景。

---

### 模式 C：群聊（Group Chat）

```
         ┌─────────────────┐
         │    消息总线      │
         │  （广播/订阅）   │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌───────┐    ┌───────┐    ┌───────┐
│Agent A│ ←→ │Agent B│ ←→ │Agent C│
│ 发言   │    │ 发言   │    │ 发言   │
│ 倾听   │    │ 倾听   │    │ 倾听   │
└───────┘    └───────┘    └───────┘
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ▼
           最终答案（涌现）
```

**特点**：去中心化，谁想发言谁发言。灵活但难收敛、token 成本高。

---

### 3.2 模式 A 的代码实现（本项目）

**主管拆任务**（[agents.py:457-476](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L457-L476)）：

```python
def plan(self, ctx: RunContext, roster: list[AgentSpec]) -> PlanResult:
    roster_text = "\n".join(
        f"- {a.name}（{a.role.value}）：{a.description or '无'}"
        for a in roster if a.role != AgentRole.SUPERVISOR
    )
    prompt = self.PLAN_PROMPT.format(
        query=ctx.query,
        roster=roster_text,
        schema=json_schema_hint(PlanResult),
    )
    # 调 LLM 生成 PlanResult
    raw = self.llm.invoke_structured(prompt, ...)
    return PlanResult(**raw)
```

**Worker 执行**（[agents.py:303-381](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L303-L381)）：

```python
def run(self, task: Optional[SubTask], ctx: RunContext) -> AgentOutput:
    # 1. 准备：拼上游产出 + 图片 + 同伴消息
    upstream_text = self._upstream_text(ctx)
    vision_text = self.describe_images(ctx)
    peer_text = ctx.bus.inbox_text(self.name) if ctx.bus else ""

    # 2. 拼提示词
    prompt = self._build_prompt(instruction, upstream_text, vision_text, peer_text, ...)

    # 3. 执行（有工具走 _run_with_tools，无工具直接 invoke）
    if self._tool_schemas():
        content, steps, logs = self._run_with_tools(prompt, self.spec.max_steps, ...)
    else:
        content = self.llm.invoke(prompt, system_prompt=self.system_prompt, ...)

    return AgentOutput(agent=self.name, task_id=task.id, success=True, content=content, ...)
```

**上下文隔离的关键**（[agents.py:240-243](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L240-L243)）：

```python
messages = [
    {"role": "system", "content": self.system_prompt},
    {"role": "user", "content": prompt},
]
# ← 每次执行新建 messages，不继承主对话历史
```

**依赖等待**（[harness.py:836-851](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L836-L851)）：

```python
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(self._execute, task, ctx): task for task in batch}
    for future in as_completed(futures):  # ← 谁先完成谁先返回
        task = futures[future]
        output = future.result()
        ctx.upstream[output.task_id] = output  # ← 结果写进"成果袋"
```

### 3.3 模式 B（LangGraph）的代码实现

```python
from langgraph.graph import StateGraph

workflow = StateGraph(TeamState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("worker_a", worker_a_node)
workflow.add_node("worker_b", worker_b_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route_to_worker, {
    "worker_a": "worker_a",
    "worker_b": "worker_b",
    "synthesizer": "synthesizer"
})
workflow.add_edge("worker_a", "synthesizer")
workflow.add_edge("worker_b", "synthesizer")
```

**和模式 A 的区别**：LangGraph 的节点共享一个 `State`，隔离性不如纯 Python 类实现。

### 3.4 模式 C（群聊）的代码实现

```python
# 伪代码：所有 Agent 共享一个消息群
class GroupChat:
    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self.messages = []

    def broadcast(self, sender: str, content: str):
        self.messages.append({"sender": sender, "content": content})
        # 所有 Agent 都能看到
        for agent in self.agents:
            if agent.name != sender:
                agent.receive(content)
```

**代价**：token 成本高、难收敛、容易跑题。

---

## 四、面试话术（可直接背）

### 4.1 决策矩阵

| 业务特征 | 推荐方案 | 理由 |
|---|---|---|
| 任务能清晰拆解、Worker 独立性强 | ① 自研主管-Worker | 结构清晰、可控性强、token 省 |
| 需要复杂多轮协作、状态持久化 | ② LangGraph 编排 | 支持 checkpoint、可回溯、可视化 |
| 技术验证期、冷启动 | ③ 现成框架（AutoGen/CrewAI） | 少写代码、快速验证 |
| 有歧义需澄清 | ④ 澄清后单 Agent | 多 Agent 反而增加复杂度 |

### 4.2 关键追问及回答

**Q1：什么时候不能用模式 A？**

> 三种情况：
> 1. **任务无法拆解**："理解我的意思"这种模糊任务
> 2. **需要深度上下文**："根据我上周问的报销政策"——Worker 没有记忆召回
> 3. **需要标准答案命中**：Skills 里的固定问答——Worker 没有 Skills 命中

**Q2：Worker 怎么隔离上下文？**

> 每次执行新建 `messages` 列表（[agents.py:240-243](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L240-L243)），不继承主对话历史。这是隔离的命门。

**Q3：主管怎么知道 Worker 都完成了？**

> `as_completed` + `ctx.upstream` 字典（[harness.py:836-851](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L836-L851)）：
> - `as_completed` 保证"谁先完成谁先写进字典"
> - `ctx.upstream` 是一个字典，key 是 `task_id`，value 是 `AgentOutput`
> - 下游任务通过 `ctx.upstream` 拿到所有上游结果
> - **批间串行**：第 2 批必须等第 1 批全部完成才启动

**Q4：消息总线和发布订阅区别？**

> 本质就是发布订阅，但加了"信箱"设计：
> - 每条消息有"收件人"字段，不是纯广播
> - `inbox(agent_name)` 只拿发给自己的
> - 加了治理（限流/配额/防环/审计）

**Q5：Worker 没有记忆/RAG/Skills，怎么办？**

> 三种解法：
> 1. **主管集中准备**：主管在 `plan()` 时先做记忆召回+RAG，把结果塞进工单指令
> 2. **共享记忆层**：所有 Worker 共享一个向量数据库，执行前查相关记忆
> 3. **后处理增强**：Worker 交回结果后，主管汇总时再做一次 RAG/记忆召回补全
>
> 本项目目前用**解法 1 的简化版**（主管拆任务，但拆之前没做记忆召回）。

---

## 五、一句话总结

> 多 Agent 协作的核心不是"每个 Worker 多强"，而是"怎么把一群简单 Worker 组织成能处理复杂问题的团队"。模式 A（主管-Worker）是工程上最平衡的方案，但要知道它的边界：任务无法拆解、需要深度上下文、需要标准答案命中时，单 Agent 反而更好。

---

## 六、代码位置速查表

| 概念 | 文件 | 行号 |
|---|---|---|
| 主管拆任务 | `modules/multiagent/agents.py` | 457-476 |
| Worker 执行 | `modules/multiagent/agents.py` | 303-381 |
| 上下文隔离 | `modules/multiagent/agents.py` | 240-243 |
| 依赖等待 | `modules/multiagent/harness.py` | 836-851 |
| 消息总线 | `modules/multiagent/coordination.py` | 132-144 |
| 上游产出透传 | `modules/multiagent/agents.py` | 398-406 |
| 汇总 | `modules/multiagent/agents.py` | 511-544 |
| 审核 | `modules/multiagent/agents.py` | 545-569 |
| 委派 | `modules/multiagent/delegation.py` | 132-160 |
| 五道中间件 | `modules/harness/middleware.py` | 62-302 |
