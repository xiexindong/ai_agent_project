# 08 · Skill 与 Tool：像在哪、差在哪

> 对应代码：`modules/skills/`（base_skill.py / builtin_skills.py / skill_registry.py）、`modules/tools/`（base_tool.py / calculator.py / data_query.py / tool_registry.py）
> 一句话：**Skill 是「作业指导书」（一段文本，告诉 LLM 该按什么流程做），Tool 是「螺丝刀」（一段代码，真正干活并返回结果）。Skill 只能推荐 Tool，不能替它干活；Tool 不知道 Skill 存在，但可以被 Skill 反复点名。**

---

## 1. 为什么看着像

因为两者的**骨架几乎一模一样**，都是「声明 → 注册 → 被 Agent 取用」：

| 环节 | Skill | Tool |
|---|---|---|
| 基类 | `BaseSkill(ABC)`（类属性声明） | `BaseTool(ABC)`（构造参数声明） |
| 唯一标识 | `name = "math_calculator"` | `name = "calculator"` |
| 说明文字 | `description` | `description` |
| 注册表 | `SkillRegistry._skills: Dict[str, BaseSkill]` | `ToolRegistry._tools: Dict[str, BaseTool]` |
| 注册方法 | `register()` / `register_batch()` | `register()` / `register_batch()` |
| 重名策略 | 覆盖 + warning | 覆盖 + warning |
| 装配位置 | `main.py:97-98` | `main.py:91-92` + `112`（MCP） |
| 注入 Agent | `ReActAgent(skill_registry=...)` | `ReActAgent(tool_registry=...)` |

**所以"像"的感觉是对的**——它们是同一套插件体系的两个变体。区别全在**运行时干什么**。

---

## 2. 核心差异一览

| 维度 | Skill（技能） | Tool（工具） |
|---|---|---|
| **本质** | **一段提示词文本**（SOP） | **一段可执行代码** |
| 有没有执行动作 | ❌ 不执行任何东西 | ✅ `execute()` 真正跑逻辑 |
| 有没有返回值 | ❌ 无 | ✅ `ToolResult` |
| **谁决定启用** | **Python 代码**（关键词打分 `match_score`） | **LLM**（Function Calling 决策） |
| 是否确定性 | ✅ 确定性（同输入必同结果） | ❌ 概率性（模型决定调不调） |
| 触发依据 | `trigger_keywords` + 阈值 0.35 | `inputSchema`（JSON Schema） |
| 生效时机 | `prepare` 节点，一次 | `reason` 决策 + `act` 执行，**可循环多次** |
| 影响 LLM 的方式 | 塞进 prompt 的 `skill_context` | 作为 `tools` 参数 + 结果进 `observation` |
| 失败表现 | 不匹配 → 走通用流程 | 报错 → `error` 文本回喂 LLM |
| 独有字段 | `trigger_keywords` / `priority` / `bound_tools` / `instructions` | `parameters_schema` / `tool_type`(local\|mcp) |
| 抽象方法 | 无（`BaseSkill` 没有 abstractmethod） | 有 `execute()` 必须实现 |
| 当前数量 | 4 个 | 5 个（2 本地 + 3 MCP） |

---

## 3. 各自的完整调用链对比

### Skill：代码匹配 → 注入文本 → 结束

```
main.py:97    skill_registry.register_batch(get_default_skills())
   ↓
prepare 节点  skill_registry.build_skill_context(clean_query)
                 ├─ 每个技能 match_score(query)
                 │     命中词按长度加权：短语(≥3字)0.6 / 双字词0.4 / 单字0.15
                 │     得分累加，封顶 1.0
                 ├─ 排序：score 降序，同分 priority 降序
                 └─ 最高分 < 0.35 → 返回 None（不激活）
   ↓
state.skill_context = format_context(match)
   = "【激活技能】数学计算\n技能说明:…\n推荐工具: calculator\n作业流程(SOP):\n1. …"
   ↓
reason 节点   拼进 REASON_PROMPT 的 {skill_context} 槽位
   ↓
             —— Skill 的使命到此结束，后面不再出现 ——
```

### Tool：LLM 决策 → 执行 → 结果回喂 → 可循环

```
main.py:91/112  tool_registry.register(...)      ← 本地 + MCP 混在一张表
   ↓
reason 节点    tools_schema = get_openai_tools_schema()   ← 全量 5 个，不过滤
               llm.invoke_with_tools(messages, tools=tools_schema)
   ↓           LLM 返回 tool_calls → state.action = {name, arguments}
条件边         need_tool_call=True → act
   ↓
act 节点       tool_registry.execute_tool(name, params, session_id)
                  └─ BaseTool.run: validate_params → execute → ToolResult
   ↓           observation = "21"
固定边         回到 reason → LLM 看到结果，决定继续调 or 直接答（可多轮）
```

**关键对比**：Skill 是一条**单向注入**链路；Tool 是**闭环反馈**链路。

---

## 4. 两者的关系：多对多、可独立

### Skill 可以脱离 Tool（`bound_tools = []`）

`KnowledgeQASkill` 就是例子：

```82:82:modules/skills/builtin_skills.py
    bound_tools = []  # 走 RAG 通道，不绑定函数工具
```

知识召回在 `prepare` 阶段由 RAG 完成、内容已进上下文，LLM 读着 SOP 直接作答。**证明 Skill 不需要依附 Tool 也能存在。**

### Tool 也可以脱离 Skill

没有任何技能命中时（走通用助手流程），LLM 照样能看到全部 5 个工具并自由调用。

### 绑定关系现状

| Skill | `bound_tools` | priority |
|---|---|---|
| `math_calculator` 数学计算 | `["calculator"]` | 10 |
| `business_data_query` 企业数据查询 | `["data_query"]` | 10 |
| `knowledge_qa` 知识库问答 | `[]`（走 RAG） | 5 |
| `web_search` 联网信息检索 | `["web_search", "get_current_time"]` | 8 |

一个 Skill 可绑多个 Tool（`web_search` 绑了 2 个）；一个 Tool 也可被多个 Skill 绑（设计上支持，当前无重叠）。

---

## 5. ⚠️ 关键认知：`bound_tools` 是「软引导」，不是硬限制

这是最容易误解的一点。

**代码证据 1** —— `skill_registry.py` 的类注释：

```
- 技能只做「软引导」：注入 SOP 与推荐工具，不硬屏蔽其他工具，避免误匹配导致无法作答。
```

**代码证据 2** —— `reason_step` 拿的是**全量工具**，没有按 `bound_tools` 过滤：

```189:194:modules/agent/react_agent.py
        if self.tool_registry:
            tools_schema = self.tool_registry.get_openai_tools_schema()
            tools_desc = "\n".join(
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in tools_schema
            )
```

所以：命中「数学计算」技能时，`web_search`、`data_query` 等**依然在工具清单里**，LLM 想调还是能调。`bound_tools` 只是以「推荐工具: calculator」这行字的形式出现在提示词里。

> **是有意为之，不是 bug。** 硬屏蔽有风险：一旦技能误匹配（比如用户说"计算一下库存"被判成数学计算），把 `data_query` 屏蔽掉就彻底答不了了。软引导最多引导错，不会堵死。

---

## 6. 为什么不干脆只用 Tool？

让 LLM 自己看着工具清单选不就行了？Skill 解决的是三件 Tool 解决不了的事：

| 价值 | 说明 |
|---|---|
| **固化流程** | 把"老员工怎么干"写成 SOP 文本，改流程只改文字，不改代码 |
| **省成本 / 提速** | 意图路由用 Python 关键词打分（0 次 LLM 调用），不用让模型每次先做一遍意图分类 |
| **可预测 + 可审计** | 命中记录进 `state.active_skill`，能回溯"这轮走了哪个流程"；关键词匹配结果稳定可复现 |
| **约束模型行为** | 强制要求用工具而非心算——见下面的例子 |

**最典型的一条 SOP**（`MathCalculatorSkill.instructions`）：

```
2. 必须调用 calculator 函数完成计算，禁止凭记忆心算给出数值结果；
```

这句话说明了两者的互补关系：**Skill 负责"不许你心算"，Tool 负责"精确算出 21"**——
Skill 是纪律，Tool 是能力。

---

## 7. `calculator.py` 顺带一个安全点

Tool 是**真代码**，所以有真实的执行风险。`calculator` 没用 `eval()`：

```105:106:modules/tools/calculator.py
            tree = ast.parse(expr, mode="eval")
            result = self._eval_node(tree)
```

用 `ast` 解析成语法树，再按 `_SAFE_OPS` / `_SAFE_FUNCS` **白名单**逐节点求值：

| 表达式 | 结果 |
|---|---|
| `"3*7"` | `21` |
| `"sqrt(16)"` | `4` |
| `"__import__('os')"` | `ValueError: 不支持的函数: __import__` |
| `"open('/etc/passwd')"` | `ValueError: 不支持的函数: open` |

> **Skill 永远不会带来这种风险**——它只是文本。这也是二者"本质不同"最实在的体现：一个要防代码注入，一个不用。

---

## 8. 速查表

| 问题 | 答案 |
|---|---|
| Skill 会执行代码吗 | 不会，纯文本注入 |
| Tool 会执行代码吗 | 会，`execute()` 是抽象方法必须实现 |
| 谁决定用哪个 Skill | Python 代码（关键词打分，阈值 0.35） |
| 谁决定用哪个 Tool | LLM（Function Calling） |
| `bound_tools` 会屏蔽其它工具吗 | **不会**，只是提示词里的"推荐工具" |
| Skill 能没有 Tool 吗 | 能（`knowledge_qa`，走 RAG） |
| Tool 能没有 Skill 吗 | 能（未命中技能时 LLM 自由调用） |
| 一个 Skill 能绑几个 Tool | 0~N（`web_search` 绑了 2 个） |
| Skill 在哪生效 | `prepare` 节点，一次性注入 `skill_context` |
| Tool 在哪生效 | `reason` 决策 + `act` 执行，可循环多次 |
| 匹配失败会怎样 | Skill：走通用流程；Tool：error 文本回喂 LLM |
| 两者唯一必填标识 | 都是 `name`（注册表键） |
