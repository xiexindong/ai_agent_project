# 多 Agent 协作：一行提问如何变成团队答案

> 写给第一次看 `modules/multiagent/agents.py` 的人。
> 用"一家小公司接客户订单"的比喻，把每个类、每个方法、每行关键代码讲清楚。
> 读完你能自己对着源码走一遍完整流程。

---

## 一、先建立整体画面：这不是聊天，是"接订单"

单 Agent（`/chat`）像**一个全能客服**：你问什么，他一个人从头查到尾。

多 Agent（`/agent/team/run`）像**一家小公司接订单**：

```
客户（你）提需求
    │
    ▼
┌─────────────┐   拆成几张工单    ┌──────────────────────────┐
│  主管 Supervisor │ ───────────────→ │ 工单 t1 → 研究员 researcher_1 │
│ （不干活，只派活）│                  │ 工单 t2 → 分析员 analyst_1   │
└─────────────┘                  └──────────────────────────┘
    │                                        │
    │            各自干完，交回工单            │
    │ ←──────────────────────────────────────┘
    ▼
┌─────────────┐   草稿答案    ┌─────────────┐
│  主管汇总答案  │ ──────────→ │ 审核员 reviewer_1 │
│  synthesize   │             │  打分、挑毛病      │
└─────────────┘             └─────────────┘
    │
    ▼
最终答案返回给客户
```

**核心区别**：单 Agent 是"一个人边想边做"；多 Agent 是"先拆、再并行做、最后收口"。

---

## 二、文件里到底有哪几个"人"

`agents.py` 定义了 **4 个角色类 + 1 个基类**，用 `build_agent()` 工厂按角色实例化：

| 类 | 角色 | 干什么 | 对应现实岗位 |
|---|---|---|---|
| `BaseAgent` | 基类 | 封装"拼提示词 → 调 LLM/工具 → 产出结果"的通用流程 | 所有员工的通用工位 |
| `SupervisorAgent` | `AgentRole.SUPERVISOR` | 拆任务（`plan`）+ 汇总答案（`synthesize`） | 项目经理 |
| `WorkerAgent` | `AgentRole.RESEARCHER` / `ANALYST` / `GENERAL` | 接一个子任务，用工具干活，产出结论 | 研究员 / 数据分析师 |
| `ReviewerAgent` | `AgentRole.REVIEWER` | 对最终草稿打分、挑毛病 | 质检员 |

**工厂函数**（[agents.py:609](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L609)）：

```python
def build_agent(spec, llm=None, tool_registry=None, gate=None) -> BaseAgent:
    if spec.role == AgentRole.SUPERVISOR:
        return SupervisorAgent(spec, llm, tool_registry, gate)   # 主管
    if spec.role == AgentRole.REVIEWER:
        return ReviewerAgent(spec, llm, tool_registry, gate)     # 审核员
    return WorkerAgent(spec, llm, tool_registry, gate)           # 其余都是执行者
```

**默认团队**（[agents.py:573](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L573)）只配 4 个人，因为"每多一个成员，主管的提示词就长一截，规划质量反而下降"：

```python
def build_default_team() -> list[AgentSpec]:
    return [
        AgentSpec(name="supervisor",   role=AgentRole.SUPERVISOR, temperature=0.2),
        AgentSpec(name="researcher_1", role=AgentRole.RESEARCHER, tools=["data_query"], multimodal=True),
        AgentSpec(name="analyst_1",    role=AgentRole.ANALYST,    tools=["calculator"]),
        AgentSpec(name="reviewer_1",   role=AgentRole.REVIEWER,   temperature=0.1),
    ]
```

---

## 三、核心数据：任务和结果长什么样

### 3.1 子任务 `SubTask`（[models.py:88](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py#L88)）

就是一张**工单**：

```python
class SubTask(BaseModel):
    id: str            # 工单号，如 "t1"
    assignee: str      # 派给谁，如 "researcher_1"
    instruction: str   # 具体指令，如 "查 2024 Q3/Q4 营收"
    depends_on: list[str]  # 依赖哪些上游工单（构成 DAG）
    priority: int      # 1~5，同批内排序用
```

### 3.2 规划结果 `PlanResult`（[models.py:100](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py#L100)）

主管拆完任务的**派单表**，**两种形态二选一**：

```python
class PlanResult(BaseModel):
    thought: str           # 主管的思考过程（排查用）
    subtasks: list[SubTask]        # 形态 A：拆出来的任务清单
    direct_answer: str     # 形态 B：简单问题直接答（不拆）
    needs_review: bool     # 是否需要审核员终审
```

关键设计：**强制二选一**（[models.py:120-124](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py#L120-L124)）：

```python
@model_validator(mode="after")
def _must_have_plan_or_answer(self):
    if not self.subtasks and not self.direct_answer.strip():
        raise ValueError("PlanResult 必须给出 subtasks 或 direct_answer 之一")
    return self
```

防止模型两边都返回空，导致"派了 0 个任务却继续等汇总"的静默失败。

### 3.3 执行结果 `AgentOutput`（[models.py:145](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py#L145)）

每个员工交回来的**工作成果单**：

```python
class AgentOutput(BaseModel):
    agent: str           # 谁干的，如 "researcher_1"
    role: AgentRole
    task_id: str         # 对应哪张工单
    content: str         # 产出正文（成功时）
    success: bool        # 成败
    error: str           # 失败原因（success=False 时）
    steps: int           # 实际推理步数
    tool_calls: list[dict]   # 工具调用日志（账单留痕）
    duration_ms: int     # 耗时
    delegations: list    # 我转出去的活（委派账）
```

**关键**：成功和失败用**同一个结构**表达，上游只看 `success` 和 `content`/`error`，不用猜返回格式。

---

## 四、主管拆任务：`SupervisorAgent.plan()` 详解

这是你光标停的地方。分 4 步：

### 第 1 步：组装"花名册"和"输入形态"（[agents.py:467-475](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L467-L475)）

```python
roster_text = "\n".join(
    f"- {a.name}（{a.role.value}）: {a.description or '通用执行者'}"
    f"{'，可用工具: ' + ','.join(a.tools) if a.tools else ''}"
    for a in roster
) or "（无可用成员）"

modality = (
    f"文本 + {ctx.mm.image_count} 张图片" if ctx.mm and ctx.mm.image_count
    else "纯文本"
)
```

把团队里有哪些人、各会什么、这次输入带不带图，写成一段文字，塞进提示词。

### 第 2 步：填充提示词模板（[agents.py:476](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L476)）

```python
prompt = self.PLAN_PROMPT.format(
    query=ctx.query,        # 用户原始问题
    modality=modality,      # 输入形态
    roster=roster_text,     # 可用成员
    schema=json_schema_hint(PlanResult),  # 输出结构说明
)
```

模板长这样（[agents.py:435-443](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L435-L443)）：

```
## 用户问题
{query}

## 输入形态
{modality}

## 可用成员
{roster}

## 成员产出规则
- 能并行的任务不要串行（不要写无谓的 depends_on）
- 一个任务只派一个人
- 简单问题直接给 direct_answer

## 输出结构（严格遵守）
{schema}

只输出一个 JSON 对象，不要多余文字。
```

**`json_schema_hint(PlanResult)`** 把 `PlanResult` 的字段结构转成 LLM 能看懂的说明，让模型"只在格子里填空"，而不是自由发挥写小作文。

### 第 3 步：调 LLM + 校验（[agents.py:483-484](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L483-L484)）

```python
data = self.llm.invoke_structured(prompt, system_prompt=self.system_prompt)
return PlanResult.model_validate(data)
```

- `invoke_structured`：让 LLM 输出 JSON
- `model_validate`：用 pydantic 当场校验，缺字段立刻报错

### 第 4 步：失败重试 + 降级（[agents.py:485-507](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L485-L507)）

```python
except Exception as first_err:
    # 把校验错误原文回喂，让模型修正后重试一次
    retry_prompt = f"{prompt}\n\n上一次输出不合法，错误是：{first_err}\n请修正后重新输出。"
    data = self.llm.invoke_structured(retry_prompt, ...)
    return PlanResult.model_validate(data)
except Exception as second_err:
    # 二次失败 → 降级为单任务直答
    return PlanResult(
        thought="规划失败，降级为单任务执行",
        subtasks=[SubTask(id="t1", assignee=fallback, instruction=ctx.query)],
        needs_review=False,
    )
```

**为什么重试一次**：模型最常见的错不是乱写，而是漏字段（如 `subtasks` 里少 `assignee`），把错误原文回喂通常一次就补齐。

**为什么降级而不是抛错**：规划失败不该让整个请求挂掉，退化成"单 Agent 硬答"至少能给出答案。

---

## 五、执行者干活：`BaseAgent.run()` 详解

主管派完活，每个 Worker 走 `run()` 方法。分 5 步：

### 第 1 步：收集上下文（[agents.py:306-312](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L306-L312)）

```python
instruction = task.instruction if task else ctx.query   # 我的具体指令
upstream_text = self._upstream_text(ctx)                # 上游产出（截断 1200 字）
vision_text = self.describe_images(ctx)                 # 图片翻译成文字（只发生一次）
peer_text = ctx.bus.inbox_text(self.name) if ctx.bus else ""  # 同伴消息（最近 20 条）
```

### 第 2 步：判断要不要委派（[agents.py:323-342](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L323-L342)）

```python
if dex is not None and dex.enabled and self.spec.can_delegate and task:
    decision = decide_delegation(self.llm, instruction, dex.candidates(self.name), ...)
    if decision and decision.should_delegate:
        rec = dex.delegate(from_agent=self.name, to_agent=decision.to_agent, ...)
        delegated.append(rec)
```

**三个前置条件缺一不可**：委派器已装配、本 Agent 有委派资格、有具体任务。开关默认关，关闭时零额外开销。

### 第 3 步：组装最终提示词（[agents.py:345](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L345)）

```python
prompt = self._build_prompt(instruction, upstream_text, vision_text, peer_text, delegation_text)
```

把任务、上游产出、图片解析、同伴消息、委派产出按顺序拼成一段话。顺序有讲究：委派产出紧跟上游产出（同级），同伴消息放最后（避免冲淡"直接给结论"的指令）。

### 第 4 步：执行（两条路径）（[agents.py:352-359](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L352-L359)）

```python
schemas = self._tool_schemas()
if schemas:
    content, steps, logs = self._run_with_tools(prompt, self.spec.max_steps, ctx.session_id)
else:
    content = self.llm.invoke(prompt, system_prompt=self.system_prompt, ...)
    steps, logs = 1, []
```

- **有工具** → 走 `_run_with_tools`：Function Calling 循环，最多 `max_steps` 轮
- **无工具** → 直接 `invoke`：一次 LLM 调用出结论

### 第 5 步：包装成 `AgentOutput`（[agents.py:360-381](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L360-L381)）

```python
return AgentOutput(
    agent=self.name, role=self.spec.role, task_id=task.id if task else "",
    content=(content or "").strip(), success=True,
    steps=steps, tool_calls=logs, duration_ms=..., multimodal=bool(vision_text),
    delegations=delegated, delegated_by=getattr(ctx, "delegated_by", ""),
)
```

**关键**：异常被 catch 住，包装成 `AgentOutput.failed(...)`，**不外抛**。一个 Worker 挂了不影响其他 Worker 和主管汇总。

---

## 六、工具循环：`_run_with_tools()` 详解

这是 Worker 真正"边想边做"的地方。和单 Agent 的 ReAct 类似，但更精简。

### 6.1 上下文隔离的命门（[agents.py:240-243](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L240-L243)）

```python
messages = [
    {"role": "system", "content": self.system_prompt},
    {"role": "user", "content": prompt},
]
```

**为什么每次新建 messages 而不是继承主对话**：
- 主 Agent 的 messages 随轮次只增不减，迟早撞 token 上限
- 子 Agent 共享这份历史 = 把膨胀复制 N 份，"省上下文"成空话
- 隔离必须做在**数据结构层**（新建列表），而不是提示词约定层

### 6.2 Function Calling 循环（[agents.py:248-299](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L248-L299)）

```python
while step < steps_budget:
    step += 1
    res = self.llm.invoke_with_tools(messages, schemas, temperature=self.spec.temperature)
    last_text = res.get("content") or ""
    calls = res.get("tool_calls") or []
    if not calls:                          # 模型不再调工具 → 收敛
        return last_text, step, logs

    for call in calls[:1]:                 # 一次只处理一个调用，避免并行结果错序
        name, args = call.get("name", ""), call.get("arguments", {})

        # HITL：敏感工具先过人工批准
        if self.gate is not None:
            approved, args = self.gate.ask_tool(name, args, agent=self.name, ...)
            if not approved:
                messages.append({"role": "user", "content": f"工具 {name} 未获批准，请改用其它方式。"})
                continue

        result = self.tools.execute_tool(name, args, session_id=session_id)
        observe = f"工具 {name} 返回：{result.result[:1000]}" if result.success else f"工具 {name} 失败：{result.error}"
        messages.append({"role": "user", "content": observe})

return last_text or "已达到步数上限，以下是已有信息下的结论。", step, logs
```

**和单 Agent ReAct 的区别**：
- 单 Agent 的 ReAct 在 `react_agent.py`，走 LangGraph 节点循环，有 6 道终止闸
- 这里的循环更简单：只看"模型是否还调工具"，步数上限 `max_steps`（默认 3）

---

## 七、主管汇总：`synthesize()` 详解

所有 Worker 交回工单后，主管把产出合成一份答案（[agents.py:511-523](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L511-L523)）：

```python
def synthesize(self, ctx: RunContext, outputs: list[AgentOutput]) -> str:
    blocks = []
    for o in outputs:
        status = "成功" if o.success else "失败"
        body = o.content if o.success else f"（失败）{o.error}"
        blocks.append(f"【{o.agent} · {status}】{body}")

    prompt = self.SYNTH_PROMPT.format(query=ctx.query, outputs="\n\n".join(blocks))
    try:
        return self.llm.invoke(prompt, system_prompt=self.system_prompt)
    except Exception as e:
        # 汇总失败 → 回退拼接所有成功产出
        return "\n\n".join(o.content for o in outputs if o.success)
```

**设计要点**：
- 失败的产出也写进提示词，如实标注"该部分未能完成"
- 汇总本身失败时，回退到简单拼接，不让整个请求挂掉

---

## 八、审核员把关：`review()` 详解

汇总草稿出来后，审核员打分挑毛病（[agents.py:545-568](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L545-L568)）：

```python
def review(self, ctx: RunContext, draft: str) -> AgentOutput:
    prompt = self.REVIEW_PROMPT.format(
        query=ctx.query, draft=draft[:4000],  # 草稿截断到 4000 字
        schema=json_schema_hint(ReviewVerdict)
    )
    try:
        data = self.llm.invoke_structured(prompt, ...)
        verdict = ReviewVerdict.model_validate(data)
        content = f"评审{'通过' if verdict.passed else '不通过'}（{verdict.score:.2f}）\n问题：...\n建议：..."
        return AgentOutput(agent=self.name, content=content, success=True, ...)
    except Exception as e:
        # 解析失败不阻断主流程，按放行处理
        return AgentOutput(agent=self.name, content=f"评审未生效（解析失败: {e}），默认放行", success=True, ...)
```

**关键**：审核员**只留意见，不改答案**。解析失败时默认放行，不让审核环节成为单点故障。

---

## 九、完整调用链（从 HTTP 请求到答案）

```
POST /agent/team/run                          server.py:713
    │
    ▼
runner.run(query, mm, session_id)             server.py:753
    │
    ▼
AgentHarness.run()                            harness.py（主方法）
    │
    ├─【第 1 步·规划】─────────────────────────────
    │  supervisor.plan(ctx, roster)               harness.py:430
    │      │
    │      ▼
    │  SupervisorAgent.plan()                     agents.py:457
    │      ├─ 组装 roster_text / modality           agents.py:467-475
    │      ├─ PLAN_PROMPT.format(...)              agents.py:476  ← 你光标停的地方
    │      ├─ llm.invoke_structured(...)           agents.py:483
    │      ├─ PlanResult.model_validate(...)       agents.py:484
    │      └─ 失败重试 → 降级                      agents.py:485-507
    │
    ├─【第 2 步·派活】─────────────────────────────
    │  按 depends_on 拓扑分批，批内线程池并行
    │  每个 SubTask → WorkerAgent.run()            agents.py:303
    │      ├─ 收集上下文（上游/图片/同伴消息）        agents.py:306-312
    │      ├─ 判断要不要委派                        agents.py:323-342
    │      ├─ _build_prompt(...)                   agents.py:345
    │      ├─ _run_with_tools(...) 或 invoke(...)  agents.py:352-359
    │      └─ 包装成 AgentOutput                    agents.py:360-381
    │
    ├─【第 3 步·一致性检测 / 仲裁 / 重规划】─────────
    │  （默认关，见 coordination.py）
    │
    ├─【第 4 步·汇总】─────────────────────────────
    │  supervisor.synthesize(ctx, outputs)         harness.py:611
    │      └─ SYNTH_PROMPT.format(...)             agents.py:518
    │
    ├─【第 5 步·终审】─────────────────────────────
    │  reviewer.review(ctx, draft)                 harness.py（条件触发）
    │      └─ REVIEW_PROMPT.format(...)            agents.py:548
    │
    └─【收口】────────────────────────────────────
        返回 RunResult（含 plan/outputs/final_answer/trace/...）
```

---

## 十、和单 Agent 的关键区别（一张表）

| 维度 | 单 Agent（`/chat`） | 多 Agent（`/agent/team/run`） |
|---|---|---|
| 入口 | `POST /chat` | `POST /agent/team/run` |
| 执行器 | `ReActAgent.run()` | `AgentHarness.run()` |
| 任务拆解 | 无（自己边想边做） | `SupervisorAgent.plan()` 先拆成子任务 |
| 执行方式 | LangGraph 节点循环（reason→act→...） | 线程池并行，每个 Worker 独立跑 |
| 上下文 | 主对话 messages 只增不减 | 每个 Worker 新建 messages，用完即弃 |
| 工具循环 | 6 道终止闸（预算/循环/无进展/...） | 只看"是否还调工具"+步数上限 |
| 结果汇总 | 无（自己给自己答案） | `SupervisorAgent.synthesize()` 汇总 |
| 质量把关 | 无 | `ReviewerAgent.review()` 打分 |
| 委派 | 无 | 可选，Worker 可把活转给别人 |

---

## 十一、怎么在代码里自己走一遍

1. 打开 [server.py:713](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py#L713)，找到 `POST /agent/team/run` 端点
2. 点进 `runner.run(...)`（[server.py:753](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py#L753)），跳到 `AgentHarness.run()`
3. 在 `harness.py:430` 找到 `supervisor.plan(ctx, self.roster)`，点进 `plan()`
4. 你就在 [agents.py:457](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L457)，往下读就是本文第四节的每一步

**调试技巧**：在 `agents.py:476` 打断点，用 `curl` 发一个 `POST /agent/team/run` 请求，能看到 `prompt` 变量的真实内容——那就是主管收到的完整提示词。

---

## 十二、一句话记住

多 Agent = **主管先拆单 → Worker 并行干活（各自隔离上下文）→ 主管汇总 → 审核员把关**。你光标停的 `agents.py:476` 就是**主管拆单时组装提示词**的那一行。

---

## 十三、多 Agent vs 单 Agent：什么时候用哪个

这是实际选型时最常被问到的问题。先说结论：**多 Agent 不是"更高级"，是"另一种场景的工具"**。

### 13.1 两套系统的"装备"对比

| 能力 | 单 Agent（`/chat`） | 多 Agent Worker |
|---|---|---|
| 记忆召回（短期+长期） | ✅ `prepare` 节点里做 | ❌ 没有 |
| RAG 知识库检索 | ✅ `prepare` 节点里做 | ❌ 没有 |
| MCP 工具调用 | ✅ `prepare`/`act` 里做 | ❌ 没有 |
| Skills 命中（标准答案） | ✅ `router.py` 三级路由 | ❌ 没有 |
| 上下文预算裁剪 | ✅ `_fit_context()` 贪心装包 | ❌ 没有 |
| 输出格式约束 | ✅ `output_format` 指令 | ❌ 没有 |
| 工具调用 | ✅ 走五道中间件 | ✅ 也走五道中间件 |

**关键区别**：单 Agent 的 `prepare` 节点是"全副武装"——记忆、RAG、MCP、Skills、权限、预算裁剪全上；多 Agent 的 Worker 只有**主管给的指令 + 上游产出 + 工具白名单**，其他一概没有。

### 13.2 为什么多 Agent 不给 Worker 配这些

不是技术做不到，是**设计取舍**：

```
如果给每个 Worker 也配 prepare 节点：
  researcher_1 查 Q3 营收 → 先走记忆召回 + RAG + Skills + ...
  researcher_2 查 Q4 营收 → 又走一遍记忆召回 + RAG + Skills + ...
  analyst_1 算环比       → 再走一遍记忆召回 + RAG + Skills + ...

  问题：
  1. 每个 Worker 重复召回同样的记忆/RAG，token 爆炸
  2. 每个 Worker 上下文变重，并行优势被抵消
  3. 主管拆任务时已经给了明确指令，Worker 不需要"自己找上下文"
```

**设计哲学**：多 Agent 的上下文由**主管集中准备**，Worker 只负责执行。就像项目经理把资料打印好发给员工，而不是让每个员工自己去档案室翻。

### 13.3 什么时候多 Agent 确实不如单 Agent

如果任务需要**深度上下文**才能答好，多 Agent 的 Worker 恰恰没有这些召回能力：

| 场景 | 为什么单 Agent 更好 |
|---|---|
| "根据我上周问的报销政策，这次出差能报多少" | 需要**记忆召回**知道"上周问了什么" |
| "结合公司最新财报和行业报告，分析竞争力" | 需要**RAG 检索**内部资料 |
| "按标准答案格式，回答客户关于产品 X 的售后政策" | 需要**Skills 命中**标准答案 |

这些场景的共同点是：**答案质量依赖"历史/资料/标准"的召回质量**。

### 13.4 什么时候多 Agent 确实比单 Agent 强

| 特征 | 例子 |
|---|---|
| 任务可**清晰拆解**为独立子任务 | "查 A 数据 + 查 B 数据 + 算对比" |
| 子任务之间**不需要共享上下文** | 查 Q3 和查 Q4 是独立的 |
| 需要**并行提速** | 3 个查询同时跑，总耗时 ≈ 最慢的那个 |
| 需要**失败隔离** | 一个子任务挂了不影响其他 |
| 需要**质量把关** | 审核员挑毛病，单 Agent 没有"第二双眼睛" |

### 13.5 一句话选型

> 任务能拆成"查 A、查 B、算对比"这种**独立子任务** → 用多 Agent；
> 任务需要"结合我之前说的、查资料、按标准格式"这种**深度上下文** → 用单 Agent。

---

## 十四、多 Agent 和工具调用到底差在哪

这是另一个高频困惑："工具调用也能查 A 数据、查 B 数据，为什么要多 Agent？"

### 14.1 本质区别：谁在做决策

```
工具调用（单 Agent 的 act 节点）：
  一个大脑（LLM）决定"现在该查 A 了" → 调工具 → 拿到结果 → 再决定"现在该查 B 了" → ...
  所有决策在同一个上下文里串行发生

多 Agent：
  主管一次性拆好"查 A 给 researcher_1，查 B 给 researcher_2"
  两个 Worker 同时独立执行，互不干扰
  主管最后汇总
```

**核心差异**：工具调用是**串行决策链**，多 Agent 是**并行分工**。

### 14.2 一张表看懂

| 维度 | 工具调用（单 Agent） | 多 Agent |
|---|---|---|
| 决策者 | 一个 LLM 大脑 | 主管拆任务 + Worker 各自执行 |
| 执行方式 | 串行：调完 A 再调 B | 并行：A 和 B 同时跑 |
| 上下文 | 所有工具结果进同一个 messages | 每个 Worker 独立 messages，用完即弃 |
| 失败影响 | 一个工具失败，整个链条中断 | 一个 Worker 失败，其他照常 |
| 结果汇总 | LLM 自己整理工具结果 | 主管专门汇总，审核员把关 |
| 适用场景 | 步骤有依赖（先查 A 才能查 B） | 步骤独立（A 和 B 无依赖） |

### 14.3 什么时候工具调用就够了

如果子任务之间有**依赖关系**，工具调用更合适：

```
"先查 Q3 营收 → 再根据 Q3 查同比增长率 → 最后算 Q4 预测"
  → 第 2 步依赖第 1 步的结果，第 3 步依赖第 2 步
  → 必须串行，工具调用链更自然
```

### 14.4 什么时候多 Agent 更合适

如果子任务之间**无依赖、可并行**，多 Agent 更合适：

```
"查 Q3 营收 + 查 Q4 营收 + 查行业平均 → 算对比"
  → 前三个查询互相独立，可同时跑
  → 多 Agent 并行总耗时 ≈ 最慢的那个，工具调用串行总耗时 = 三个相加
```

### 14.5 一句话

> 工具调用 = **一个人按顺序做几件事**；多 Agent = **几个人同时各做一件事，最后汇总**。
> 步骤有依赖用工具调用，步骤独立要并行用多 Agent。

