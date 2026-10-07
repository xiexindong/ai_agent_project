# 11 Prompt 拼接与循环控制权

> **本文定位**：回答连着问出来的三个问题——
> ① 是不是所有组件（tools / MCP / skills / memory / RAG / LLM）最终都只是为了拼成一个完整 prompt？
> ② 谁控制 reason 第一步、第二步的结果？
> ③ 是不是 ReAct 控制这个循环？
>
> **三句话结论**：
> - 拼 prompt 只覆盖「读」这一半，真正的终点是 **LLM 的上下文窗口**，不是任何数据库；
> - 每步结果由**三个角色分工控制**：LLM 出决策、`reason_step` 的 if 翻译字段、`_should_continue` 决定跳转；
> - **ReAct 是设计模式，不是控制器**。循环转不转由 LangGraph 的那条 `add_edge("act", "reason")` 决定。

---

## 〇、先纠两个高频误称

| 常见说法 | 实际情况 | 证据 |
|---|---|---|
| 「一切最终都是 Milvus」 | Milvus 只存**知识库文档向量**；对话记忆存在 **PostgreSQL + pgvector** | 知识库：`rag_chain.py:68` `self.vector_store.search()`；记忆：`memory_manager.py:250` `self.pg.search_memory_by_vector()` |
| 「ReAct 控制循环」 | ReAct 只决定「建哪些节点、叫什么」，节点叫 `reason`/`act` 是它的影响 | 循环的连线在 `graph.py:97` |

这两条如果不纠正，后面所有推理都会偏。

---

## 一、「一切为了拼 Prompt」的真相：有三条出口，不是一条

最直观的证据是 `reason_step` 里真正"拼 prompt"的那一段：

```187:209:modules/agent/react_agent.py
        tools_schema = []
        tools_desc = "（无可用工具）"
        if self.tool_registry:
            tools_schema = self.tool_registry.get_openai_tools_schema()
            tools_desc = "\n".join(
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in tools_schema
            )
        ...
        prompt = self.REASON_PROMPT.format(
            query=state.user_query,
            skill_context=state.skill_context or "（未匹配到专门技能，按通用企业助手流程处理）",
            rag_context=state.rag_context or "（知识库无相关内容）",
            memory_context=state.memory_context or "（无历史记忆）",
            tool_history=tool_history,
            observation=state.observation or "（初始状态，暂无工具观察）",
            tools_desc=tools_desc,
        )
```

`REASON_PROMPT` 一共 7 个槽位，填满就是那个"非常完整的 prompt"。

### 1.1 出口 A：进 prompt 的（都是字符串拼接）

| 组件 | 槽位 | 数据来源 | 上游生成位置 |
|---|---|---|---|
| 技能 SOP | `skill_context` | `SkillRegistry.build_skill_context()` | `react_agent.py:104` |
| 知识库召回 | `rag_context` | **Milvus** 粗召回 → BGE 精排 | `rag_chain.py:78` `build_context()` |
| 记忆 | `memory_context` | **PG + pgvector** 长期摘要 + 最近 3 轮原文 | `memory_manager.py:159` `build_context()` |
| 历史工具调用 | `tool_history` | 前几轮的 `tool_calls_history` | `react_agent.py:196` |
| 工具观察结果 | `observation` | 上一轮 `act_step` 写入的工具返回 | `react_agent.py:274` |
| 工具描述 | `tools_desc` | `ToolRegistry.get_openai_tools_schema()` 转文本 | `react_agent.py:191` |

### 1.2 出口 B：不进 prompt、不落库的（控制流）

| 组件 | 它控制什么 | 代码位置 |
|---|---|---|
| 图的节点与边 | 流程走 `act` 还是 `answer` | `graph.py:76-99` |
| `max_iterations` | 死循环熔断（默认 10） | `react_agent.py:234` |
| `recursion_limit` | 引擎层总步数熔断（30） | `graph.py:273` |
| `SkillRegistry.match()` 匹配算法 | 决定注入**哪段** SOP，算法本身模型看不到 | `react_agent.py:104` |
| `tool_registry.reset_session()` | 每轮清空工具去重账本 | `main.py:184` |

### 1.3 出口 C：不进 prompt、但落库的（喂给下一轮）

| 数据 | 落到哪 | 代码位置 |
|---|---|---|
| 原文 + 摘要 + 摘要向量 + 问句向量 | PG `chat_history` | `graph.py:199` → `pg_manager.py:479` |
| Agent 状态快照 | PG `chat_session.agent_state` | `graph.py:217` → `pg_manager.py:400` |
| 工具调用日志 | PG（随 `save_turn` 一并写） | `graph.py:207` |

**出口 C 才是关键**：这一轮的产出，下一轮会被 `prepare_state` 重新召回，拼进 prompt。所以「拼 prompt」不是单向汇聚，而是**吃自己输出的闭环**。

### 1.4 一个容易被漏掉的第二通道：工具 Schema

工具其实有**两份**，走两个不同的通道：

```212:218:modules/agent/react_agent.py
        result = self.llm.invoke_with_tools(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT.format(max_iter=self._max_iter)},
                {"role": "user", "content": prompt},
            ],
            tools=tools_schema,
        )
```

| 形式 | 走哪 | 给谁看 |
|---|---|---|
| `tools_desc`（纯文本） | 拼进 `prompt` 字符串 | 让模型"读懂有哪些工具" |
| `tools_schema`（JSON Schema） | 单独的 `tools=` 参数 | 让接口层校验参数格式，`tool_choice="auto"` 靠它决策 |

---

## 二、拼出来的 Prompt 长什么样（输入 → 输出对照）

场景：用户问 `3*7 等于多少`，第 1 轮 reason 时 state 各字段的取值：

```python
state.skill_context       = "【数学计算】...优先使用 calculator 工具"
state.rag_context         = ""          # Milvus 没命中
state.memory_context      = ""          # 首次会话无记忆
state.observation         = ""
state.tool_calls_history  = []
state.user_query          = "3*7 等于多少"
```

`REASON_PROMPT.format(...)` 填完后的真实文本（渲染后是 8 段）：

```
## 用户提问
3*7 等于多少

## 激活技能（SOP作业流程）
【数学计算】...优先使用 calculator 工具

## 知识库参考
（知识库无相关内容）

## 历史记忆
（无历史记忆）

## 历史工具调用
（无）

## 工具观察结果
（初始状态，暂无工具观察）

## 可用工具（Function Calling）
- calculator: 计算数学表达式
- data_query: 查询业务数据
- web_search: 联网搜索
- get_current_time: 获取当前时间

请分析当前情况，决定下一步行动：
...
```

模型看完返回的不是文本，而是工具调用：

```python
{"content": "", "tool_calls": [{"name": "calculator",
                                "arguments": {"expression": "3*7"},
                                "id": "call_abc123"}]}
```

第 2 轮 reason **用同一个模板重新拼一遍**，但两个槽位被填上了：

```python
state.observation = "21"                                    # act 节点写入
tool_history      = "  - calculator({'expression': '3*7'}) → 21"
```

模型看到 `21`，这次 `tool_calls=None`，直接把 `content` 当最终答案。

**注意**：同一个 `reason` 节点跑了两遍，代码里没有"第一步做什么、第二步做什么"的硬编码剧本。

---

## 三、谁控制每一步的结果

控制权分散在四个角色手里：

| 角色 | 控制什么 | 代码位置 |
|---|---|---|
| **LLM** | 决策内容：调不调工具、调哪个、参数是啥 | 返回 `result["tool_calls"]` |
| **`reason_step` 的 if** | 把 LLM 原始输出翻译成 state 字段 | `react_agent.py:225-231` |
| **`iteration` 计数器** | 超限时强制掐断，模型说什么都不好使 | `react_agent.py:234-237` |
| **`_should_continue`** | 拿 state 字段当红绿灯，决定走 `act` 还是 `answer` | `graph.py:248-252` |
| **`act_step`** | 第二步的输入 `observation` 由它写入 | `react_agent.py:273-276` |

### 3.1 第一步（第 1 次 reason）的三层控制

```220:237:modules/agent/react_agent.py
        state.reasoning = result["content"] or ""
        tool_calls = result.get("tool_calls")
        breakpoint()  # 【调试断点】模型决策结果：可看 tool_calls（模型决定调不调工具、调哪个）

        # 判断是否需要工具
        if tool_calls and state.iteration < state.max_iterations:
            state.need_tool_call = True
            state.action = tool_calls[0]  # 取第一个工具调用
        else:
            # 不需要工具，直接作为最终答案
            state.need_tool_call = False
            state.final_answer = result["content"] or "抱歉，我暂时无法回答这个问题。"

        # 最大循环次数拦截（边界3）
        if state.iteration >= state.max_iterations and state.need_tool_call:
```

拆开看：

| 层 | 谁在做 | 产出 |
|---|---|---|
| 输入内容 | `prepare_state` 填的 7 个槽位 | `prompt` 文本 |
| 决策内容 | LLM | `tool_calls` |
| 字段落定 | `reason_step` 的 `if` | `need_tool_call` / `action` / `final_answer` |
| 跳转 | `_should_continue` | `"act"` 或 `"answer"` |

### 3.2 第二步（第 2 次 reason）的特殊之处

**第二步的输入不由 reason 节点控制，是 `act_step` 写的**：

```273:281:modules/agent/react_agent.py
        if result.success:
            state.observation = str(result.result)
        else:
            state.observation = f"工具执行失败: {result.error}"
            _logger.warning(f"工具执行失败: {tool_name} | {result.error}")

        # 重置 action，准备下一轮
        state.action = {}
        return state
```

`observation` 的源头是 `ToolRegistry.execute_tool()` 的真实执行结果（本地函数直接跑，MCP 走 JSON-RPC 子进程）。

`state.action = {}` 这行不能少：不清空的话，下一轮 reason 会把**旧 action** 误当成"还要再调一次"。

### 3.3 跳转权：LLM 完全没有发言权

```248:252:modules/agent/graph.py
        if state.get("finished"):
            return "answer"
        if state.get("need_tool_call") and not state.get("error"):
            return "act"
        return "answer"
```

三种情况：

| 情况 | state 特征 | 返回值 |
|---|---|---|
| A | `finished=True` | `"answer"` |
| B | `need_tool_call=True` 且 `error` 为空 | `"act"` |
| C | `need_tool_call=True` 但 `error` 有值 | `"answer"`（兜底防死循环） |

### 3.4 完整字段归属表

| state 字段 | 谁写进去的 | 阶段 |
|---|---|---|
| `skill_context` / `rag_context` / `memory_context` | `prepare_state` | prepare |
| `reasoning` / `need_tool_call` / `action` / `final_answer` | `reason_step` | reason |
| `observation` / `tool_calls_history` | `act_step` | act |
| 路由结果（`"act"` / `"answer"`） | `_should_continue` | 边 |

### 3.5 具体数值的输入 → 输出

用户问 `3*7 等于多少`，`max_iterations=10`：

```
第 1 次 reason
  输入 observation  = ""
  输入 tool_history = "（无）"
  LLM 返回          = {"content":"", "tool_calls":[{"name":"calculator","arguments":{"expression":"3*7"},"id":"call_abc123"}]}
  if 判断           = tool_calls 非空 且 iteration=1 < 10 → True 分支
  结果              = {need_tool_call: True, action: {name:"calculator",...}, final_answer: ""}

  _should_continue  = 读 state → 返回 "act"
act 节点
  输入 action       = {name:"calculator", arguments:{expression:"3*7"}}
  输出 observation  = "21"
  tool_calls_history= [{"name":"calculator",...,"result":"21"}]

第 2 次 reason
  输入 observation  = "21"          ← 上一步 act 写的
  输入 tool_history = "  - calculator({'expression': '3*7'}) → 21"
  LLM 返回          = {"content":"3*7 等于 21", "tool_calls": None}
  if 判断           = tool_calls 为空 → else 分支
  结果              = {need_tool_call: False, action: {}, final_answer: "3*7 等于 21"}

  _should_continue  = 返回 "answer"
```

---

## 四、是不是 ReAct 控制循环？不是

### 4.1 四个角色的分工

| 角色 | 是思想还是能执行的东西 | 控制什么 | 代码位置 |
|---|---|---|---|
| **ReAct** | 思想（Reason→Action→Observation） | 决定**要不要**有 reason/act 节点、叫什么 | `react_agent.py` 的类名 + `SYSTEM_PROMPT` 里的文字 |
| **LangGraph** | 能执行的引擎 | 循环**能不能转** | `graph.py:97` |
| **`_should_continue`** | 纯函数 | 每圈往哪拐 | `graph.py:225` |
| **`max_iterations` + `recursion_limit`** | 参数 | 什么时候强制停 | `react_agent.py:234` / `graph.py:273` |

真正"连成环"的那条线：

```96:99:modules/agent/graph.py
        # act 执行后回到 reason（ReAct 循环）
        workflow.add_edge("act", "reason")
        workflow.add_edge("answer", "save")
        workflow.add_edge("save", END)
```

岔路口：

```88:95:modules/agent/graph.py
        workflow.add_conditional_edges(
            "reason",
            self._should_continue,
            {
                "act": "act",
                "answer": "answer",
            },
        )
```

**一个易忽略的点**：`ReActAgent.SYSTEM_PROMPT` 里确实写着"## 工作流程（ReAct）1. Reason 2. Action 3. Observation"，但那只是**告诉模型"你在做 ReAct"**，是给模型看的文字，不产生任何控制力。模型只是照着要求输出 `tool_calls`。

### 4.2 反证：改一行边，循环消失

```python
# 情况 A：现状（act 回到 reason）→ 循环存在
workflow.add_edge("act", "reason")

# 用户问 "3*7 等于多少"，实际执行轨迹：
#   iteration=1  reason → _should_continue 返回 "act"
#   act: calculator(3*7) → observation="21"
#   ↓ 沿 act→reason 这条边回到 reason
#   iteration=2  reason → _should_continue 返回 "answer"
#   answer → final_answer="3*7 等于 21"
# 结果：{iteration: 2, tool_call_logs: [calculator], final_answer: "3*7 等于 21"}
```

```python
# 情况 B：把边改成 act → answer（切断回边）→ 循环消失
workflow.add_edge("act", "answer")

# 同样问 "3*7 等于多少"，实际执行轨迹：
#   iteration=1  reason → 要调工具 → act 执行 calculator
#   ↓ 沿 act→answer 直接去收尾，不再回 reason
#   answer 节点进场时 observation="21"，但 LLM 没机会看第二眼
# 结果：{iteration: 1, final_answer: <第一次 reason 基于空 observation 生成的答案>}
```

**ReAct 这个名词一个字都没改，循环行为完全变了。** 这就证明控制循环的是边，不是 ReAct。

### 4.3 等价写法：把图手写成 while

```python
# LangGraph 图 ≡ 下面这个手写循环
state = prepare_state(query)                       # 节点
while True:
    state = reason_step(state)                     # 节点 ← ReAct 的 "Reason"
    nxt = _should_continue(state)                  # 条件边
    if nxt == "answer":
        break                                      # 跳出 → answer 节点
    state = act_step(state)                        # 节点 ← ReAct 的 "Action"
    # 循环末尾回到 while 顶部 ← 等价于 add_edge("act", "reason")
```

最后那行注释是重点：`while` 天然回顶 = 图里那条 `add_edge("act","reason")`。所以"循环"这件事是**执行结构**给的，ReAct 只是给几个位置贴了 Reason/Action/Observation 的标签。

---

## 五、为什么要这么分层

| 做法 | 原因（不这么写会怎样） |
|---|---|
| 全部塞进一个 `user` 消息，而不是多轮 messages | 模型只看一段文字，无 role 歧义；代价是没法用原生 `assistant/tool` 消息回填工具结果 |
| 工具留两份（文本 + Schema） | 文本版让模型"读懂"，Schema 版让接口"校验"。少了 Schema，`tool_choice="auto"` 无从决策 |
| 空值填 `"（知识库无相关内容）"` 而非留空 | 留空会让槽位空缺，模型可能误以为上下文被截断而乱编 |
| 记忆和知识库分两条检索链 | 混用会让"用户自己的历史"和"公司文档"互相污染召回 |
| LLM 只出 `tool_calls`，不让它控制跳转 | 模型输出是自由文本，格式可能不对；跳转必须基于**结构化字段**判断才稳定 |
| `iteration < max_iterations` 写在代码里 | 只靠 prompt 里写"最多循环10次"不可靠，模型会无视，必须代码硬拦 |
| `_should_continue` 加 `not error` 分支 | 出错时强制收尾，避免"出错→还想调工具→再出错"的空转 |
| `act_step` 末尾 `state.action = {}` | 不清空会导致旧 action 被当成新一轮调用，重复执行工具 |
| `prepare` 只在 `not state.get("prepared")` 时执行 | 第 2 次 reason 不重跑记忆/RAG 召回，省 token 和 embedding |
| 落库放在 `answer` 之后的 `save` 节点 | 先算完答案才存，保证摘要包含完整回答；存完成为下一轮 prompt 素材 |

---

## 六、一句话总结

> **「读」的组件（技能 SOP / RAG / 记忆 / 工具历史 / 观察结果）被字符串拼成 7 个槽位，汇进 `REASON_PROMPT`；
> 「写」的组件（`save` 节点）把这轮产出落库，变成下一轮的输入——这才是闭环，Milvus 只是其中一个数据源，不是终点。
> 每步结果由 LLM 出决策、`reason_step` 的 if 翻译字段、`_should_continue` 决定跳转，三者分工；
> ReAct 画图纸，LangGraph 盖房子，`_should_continue` 当红绿灯，`max_iterations` 当保险丝。**

关键数字与边界再钉一遍：

1. `REASON_PROMPT` 有 **7 个槽位**，填满就是那个"完整 prompt"；
2. 工具存在**两个通道**：`prompt` 里的 `tools_desc` 文本 + 独立的 `tools=Schema` 参数；
3. 记忆在 **PG + pgvector**，知识库在 **Milvus**，两者不能混称；
4. 循环由 **`graph.py:97` 的 `add_edge("act","reason")`** 控制，删掉即变成一次性流程；
5. 两道刹车：**`max_iterations=10`**（业务层）+ **`recursion_limit=30`**（引擎层）。
