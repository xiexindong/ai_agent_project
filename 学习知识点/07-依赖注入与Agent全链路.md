# 07 · 依赖注入与 Agent 全链路

> 对应代码：`main.py:133-142`（装配）、`modules/agent/react_agent.py`、`modules/agent/graph.py`
> 一句话：**这 7 行不执行任何逻辑，只是把 5 个已造好的组件"接线"给 ReActAgent；真正的全流程跑在 `AgentGraph.run()` 那张 5 节点图里。**

---

## 1. 这 7 行在干什么

```133:142:main.py
    react_agent = ReActAgent(
        llm_client=llm,
        tool_registry=tool_registry,
        rag_chain=rag,
        memory_manager=memory,
        skill_registry=skill_registry,
    )
    # AgentGraph 把 ReActAgent 的四步（prepare→reason↔act→answer）编排成 LangGraph 状态图
    agent_graph = AgentGraph(react_agent)
```

**依赖注入（DI）**：组件全部在外部造好（`main.py:83-124`），这里只是传进来。
ReActAgent 自己**不 new 任何依赖**（除了 LLM 的兜底），好处是：

| 好处 | 说明 |
|---|---|
| 可替换 | 想换 LLM / 换向量库，只改调用方，不动 Agent |
| 可测试 | 单测时传个假 `tool_registry` 就能跑 |
| 可降级 | 传 `None` 就等于关闭该能力（见第 3 节） |

构造时做了什么（`react_agent.py:53-68`）：

```python
self.llm = llm_client or LLMClient()   # 唯一有兜底的：没传就自己 new 一个
self.tool_registry = tool_registry     # 其余 4 个原样存，None 就是 None
self.rag = rag_chain
self.memory = memory_manager
self.skill_registry = skill_registry
self._max_iter = get_settings().max_react_iterations   # 注意：不来自参数，直接读配置
```

> 命名变化：`rag_chain` → `self.rag`、`memory_manager` → `self.memory`、`skill_registry` → 同名。
> 后面 `_save_node` 里的 `self.agent.memory.pg` 就是这个 memory。

---

## 2. 五个参数逐个解析

| 参数 | 在哪造的 | 内部存成 | 承担的工作 |
|---|---|---|---|
| `llm_client` | `main.py:83` `LLMClient()` | `self.llm` | 所有 LLM 调用：Reason 决策、生成答案、生成摘要 |
| `tool_registry` | `main.py:91-92` 本地工具 + `112` MCP 工具 | `self.tool_registry` | 提供工具清单给 LLM、执行工具（本地/MCP 统一调度） |
| `rag_chain` | `main.py:124` `RAGChain()` | `self.rag` | 知识库召回，注入 `rag_context` |
| `memory_manager` | `main.py:87` `MemoryManager(llm_client=llm)` | `self.memory` | 记忆召回 + 落库（内部持有 `pg`） |
| `skill_registry` | `main.py:97-98` | `self.skill_registry` | 意图匹配，命中则注入 SOP 流程 |

### 它们在 5 个节点里分别被谁用

| 组件 | prepare | reason | act | answer | save |
|---|---|---|---|---|---|
| `llm` | | ✅ `invoke_with_tools` | | ✅ `invoke` | |
| `tool_registry` | | ✅ 生成 tools 描述 | ✅ `execute_tool` | | |
| `rag` | ✅ `build_context` + `retrieve` | | | | |
| `memory` | ✅ `build_context` | | | | ✅ `save_turn` + `pg.update_session_state` |
| `skill_registry` | ✅ `build_skill_context` | | | | |

**注意分工**：
- `prepare` 是唯一用到 3 个组件（skill / memory / rag）的节点——**召回全在这，后面节点只读现成的**
- `act` 只用 `tool_registry`，`reason` 只用 `llm` + `tool_registry`（后者只为拼描述）

---

## 3. 降级：5 个依赖里 4 个可以缺席

每个组件在使用前都有 `if self.xxx:` 判空 + `try/except`：

```114:116:modules/agent/react_agent.py
        if self.memory:
            try:
                ...
```

| 缺了谁 | 表现 |
|---|---|
| `skill_registry` | 跳过技能匹配，走通用助手流程 |
| `memory` | 无历史记忆；`save` 节点整段跳过（`graph.py:181`） |
| `rag` | `rag_context = ""`，提示词显示"（知识库无相关内容）" |
| `tool_registry` | `tools_desc = "（无可用工具）"`；act 时 `observation = "工具注册器未初始化…"` |
| `llm` | **不能缺**——构造函数里 `llm_client or LLMClient()` 兜底新建 |

配合 `main.py` 的 try/except（MCP 挂、Milvus 挂），构成项目的「全链路降级」。

---

## 4. 完整执行链路

入口：`agent_graph.run(query, session_id, user_id)` → `graph.py:256`

```
run()
 └─ initial_state = {session_id, user_id, user_query, prepared:False}   ← 只有 4 个字段
 └─ graph.invoke(initial_state, {"recursion_limit": 30})
     │
     ├─[prepare]──────────────────────────────────────────────────────
     │   if not state["prepared"]:
     │       agent.prepare_state(query, session_id, user_id)
     │         ① check_prompt_injection(query)   → 命中则直接返回 finished=True 的 state
     │         ② sanitize_input(query)           → 清洗
     │         ③ skill_registry.build_skill_context()  → active_skill / skill_context
     │         ④ get_embedding_service().embed(clean_query)   ← 算一次问句向量，复用给 save
     │         ⑤ memory.build_context(...)       → memory_context / 长短期记忆
     │         ⑥ rag.build_context(clean_query)  → rag_context + rag_sources
     │   返回 full_state.to_dict() + prepared=True
     │
     ├─[reason]───────────────────────────────────────────────────────
     │   AgentState.from_dict(state) → agent.reason_step()
     │     ① iteration += 1
     │     ② tool_registry.get_openai_tools_schema()  → 工具清单
     │     ③ 拼 REASON_PROMPT（query+技能+知识库+记忆+工具历史+observation+工具清单）
     │     ④ llm.invoke_with_tools(messages, tools=tools_schema)
     │     ⑤ 有 tool_calls 且 iteration < max → need_tool_call=True, action=tool_calls[0]
     │        否则 → need_tool_call=False, final_answer=content
     │     ⑥ iteration >= max 且还要调工具 → 强制终止（边界3）
     │   只回写 6 个字段：iteration/reasoning/action/need_tool_call/final_answer/finished
     │
     ├─[should_continue 条件边]───────────────────────────────────────
     │   finished=True                          → "answer"
     │   need_tool_call=True 且 error 为空       → "act"
     │   其它（含 need_tool_call 但有 error）     → "answer"   ← 兜底防死循环
     │
     ├─[act]（可能循环多次）───────────────────────────────────────────
     │   tool_registry.execute_tool(name, params, session_id)
     │     └─ BaseTool.run → validate_params → execute → ToolResult
     │        （本地直接算 / MCP 转发子进程，对这里完全透明）
     │   成功 → observation = str(result)
     │   失败 → observation = "工具执行失败: ..."   ← 不中断，回喂给 LLM 让它自己调整
     │   追加 tool_calls_history + tool_call_logs，清空 action
     │   └─ 固定边回到 reason（这就是 ReAct 循环）
     │
     ├─[answer]───────────────────────────────────────────────────────
     │   若已有 final_answer 且不再调工具 → 直接用（省一次 LLM）
     │   否则 → llm.invoke(综合 prompt) 重新生成
     │   generate_summary()：优先从【摘要】标签正则提取，否则调 LLM 生成
     │
     └─[save]─────────────────────────────────────────────────────────
         if agent.memory and session_id and user_id:
             ① embed(summary)          → 摘要向量
             ② 复用 prepare 阶段暂存的问句向量（state.summary_embedding）
             ③ memory.save_turn(原文 + 摘要 + 两个向量 + 工具日志 + RAG 来源)
             ④ memory.pg.update_session_state(session_id, agent_state.to_dict())
         → END
```

### 状态怎么在节点间流动

**两套表示**，每个节点都在做转换：

```
graph 内部（dict / AgentStateDict）  ──from_dict──▶  AgentState 对象  ──▶  ReActAgent 方法
                                     ◀──to_dict────                   ◀──
```

- 图层面是 **`dict`**（LangGraph 要求，用 `TypedDict` 标注）
- ReActAgent 层面是 **`AgentState` 对象**（方便改字段）
- 节点返回值是 **`dict` 增量**（LangGraph 自动 merge 回 state）

⚠️ 所以每个节点返回的都是**字段白名单**，不是全量：

```135:142:modules/agent/graph.py
        updates = {
            "iteration": updated.iteration,
            "reasoning": updated.reasoning,
            "action": updated.action,
            "need_tool_call": updated.need_tool_call,
            "final_answer": updated.final_answer,
            "finished": updated.finished,
        }
```

> **改了 AgentState 的字段但节点没回写 → 改动会丢。** 加新字段时两边都要动：`state.py` 加字段 + 对应节点 updates 里补上。

---

## 5. 一次完整轨迹（文档中的例子）

用户问「3\*7 等于多少」：

| 步 | 节点 | 关键动作 | 结果 |
|---|---|---|---|
| 1 | prepare | 技能匹配命中「数学计算」，召回记忆/知识库 | `skill_context` 注入 SOP |
| 2 | reason | LLM 决策要调工具 | `need_tool_call=True`, `action={name:"calculator", args:{expression:"3*7"}}` |
| 3 | 条件边 | `need_tool_call=True` 且无 error | → `act` |
| 4 | act | 执行 calculator | `observation="21"` |
| 5 | reason | LLM 看到 21，决策直接答 | `need_tool_call=False`, `final_answer="3*7 等于 21"` |
| 6 | 条件边 | 不再调工具 | → `answer` |
| 7 | answer | 已有答案直接用 + 抽摘要 | `llm_summary="数学计算:3*7=21"` |
| 8 | save | 落 PG | END |

`recursion_limit=30` 是图的硬保险丝：ReAct 每轮消耗 2 步（reason+act），`max_react_iterations=10` 会先触发强制终止。

---

## 6. 速查表

| 问题 | 答案 |
|---|---|
| 这 7 行执行逻辑吗 | 不执行，只是装配（依赖注入） |
| 5 个依赖哪个必填 | 只有 `llm`（其它传 None 就是降级） |
| `max_iterations` 从哪来 | `get_settings().max_react_iterations`，不由参数传入 |
| 工具清单怎么给 LLM 的 | `tool_registry.get_openai_tools_schema()` → OpenAI Function Calling 格式 |
| 工具失败会中断吗 | 不会，转成 `observation` 文本回喂 LLM |
| 循环怎么停的 | 三重：`need_tool_call=False` / `iteration>=max` / `recursion_limit=30` |
| 节点返回值是什么 | 字段增量 dict，LangGraph 自动 merge |
| 状态存哪 | `save` 节点 → `pg.update_session_state()`（**只存不读**，见文档 01） |
| 记忆向量存几个 | 2 个：摘要向量 + **问句向量**（后者是长期记忆主检索列） |

---

## 7. 两个值得注意的地方

### ① Prompt 注入拦截后，仍会进 reason 调一次 LLM

`prepare_state` 检测到注入时返回 `finished=True` + 拦截文案（`react_agent.py:82-90`），
但 `prepare → reason` 是**固定边**，不检查 `finished`，所以：

```
prepare(拦截，finished=True)
   ↓ 固定边，无条件
reason  ← 仍会用 state.user_query（原始注入文本）拼 prompt 调 LLM
   ↓ LLM 返回的 content 会覆盖 state.final_answer
answer  ← 用的是被覆盖后的答案，拦截文案没了
```

**现状影响**：拦截文案会被 LLM 输出覆盖，且注入内容仍被送进模型。
好消息是 `should_continue` 里 `finished` 优先级最高，所以**不会走到 act 去执行工具**。

要修的话：在 `prepare` 和 `reason` 之间加条件边（finished → 直接跳 answer），或在 `_reason_node` 开头判断 `state.get("finished")` 直接 return。

### ② `summary_embedding` 字段名是历史包袱

它在 `prepare` 阶段存的是**问句向量**（`react_agent.py:131` 注释里也写了），`save` 节点复用它作为 `query_embedding` 落库（`graph.py:194`）。
读代码时别被名字骗了——它不是摘要的向量。
