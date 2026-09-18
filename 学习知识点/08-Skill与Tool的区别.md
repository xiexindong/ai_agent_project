# 08 Skill 与 Tool：作业指导书 vs 干活的手

> 对应代码：`modules/skills/`（base_skill.py / builtin_skills.py / skill_registry.py）、`modules/tools/`（base_tool.py / calculator.py / data_query.py / tool_registry.py）
>
> 一句话：**Skill 是「作业指导书」（一段文字，告诉 LLM 该按什么流程做），Tool 是「干活的手」（一段代码，真的把活干完并返回结果）。**
> 指导书不能替你干活，手也不会自己知道该干什么活——所以两者必须配合，但**互不依赖**。

---

## 〇、30 秒看懂：跟着一个例子走一遍

用户输入：

```
[cli-user | a1b2c3d4] > 3乘7等于几
```

接下来发生 5 件事，**注意每一步是 Skill 在干活、还是 Tool 在干活**：

### 第 1 步：`prepare` 节点，Python 关键词打分（Skill 被选中）

把 4 个技能的触发词都拿用户的句子去碰一遍：

```python
MathCalculatorSkill.match_score("3乘7等于几")
# 命中词：["等于几", "乘"]
#   等于几 → 3 个字，权重 0.6
#   乘     → 1 个字，权重 0.15
# 得分 = 0.75  →  0.75 >= 阈值 0.35  →  激活
```

**这一步全程没有调用大模型**，纯 Python 字符串匹配，花 0 元、0 毫秒。

### 第 2 步：把这个技能的 SOP 文本塞进 prompt（Skill 的唯一产出）

```python
skills.build_skill_context("3乘7等于几")
# 返回格式化的文本 ↓
```

```
【激活技能】数学计算
命中意图词: 等于几, 乘
技能说明: 处理四则运算、幂运算、三角函数、开方、取模等各类数学计算问题，保证结果精确。
推荐工具: calculator
作业流程(SOP):
1. 从用户问题中识别已知数值与运算关系，转换为合法数学表达式（如 '2加3' → '2+3'）；
2. 必须调用 calculator 函数完成计算，禁止凭记忆心算给出数值结果；
3. 向用户展示「计算式 → 结果」，带单位的问题需注明单位；
4. 若表达式无法唯一确定（如缺少必要数值），先向用户追问澄清，不要猜测参数。
```

这段文字被拼进 `REASON_PROMPT` 的 `{skill_context}` 槽位。**Skill 的使命到这里就结束了**——它全程只干了"提供一段文字"这一件事。

### 第 3 步：`reason` 节点，LLM 读完 SOP，决定动手（Tool 被点名）

```python
# LLM 返回的不是文字，而是工具调用请求
{"tool_calls": [{"name": "calculator", "arguments": {"expression": "3*7"}}]}
```

注意 LLM 把 `3乘7` 换算成了 `3*7`——这正是 SOP 第 1 条要求的。

### 第 4 步：`act` 节点，真的执行（Tool 干活）

```python
# act 节点 → tool_registry.execute_tool("calculator", {"expression": "3*7"})
ToolResult(success=True, result=21, duration_ms=12)
# 写进 state.observation = "21"
```

**这才是"干活"**。没有 Tool，LLM 只能说"大约是 21"，说错了也没人知道。

### 第 5 步：循环回 `reason`，LLM 看到 `21` 后组织回答

```
3 × 7 = 21
```

### 复盘：谁负责什么？

| 环节 | 谁在干 | 干了什么 |
|---|---|---|
| 判断"这是数学问题" | **Skill 机制**（Python 打分） | 选出技能，不产生答案 |
| 提供"必须用工具、先把中文转成表达式"的规矩 | **Skill**（一段文本） | 约束 LLM 的行为 |
| 把 `3乘7` 转成 `3*7` | **LLM** | 理解语义 |
| 算出 `21` | **Tool**（真代码） | 保证结果精确 |
| 决定什么时候调、用什么参数 | **LLM** | 概率性决策 |
| 组织成人类能读的回答 | **LLM** | 自然语言生成 |

---

## 一、先用生活类比记住它俩

| 类比物 | 对应本项目 | 特点 |
|---|---|---|
| **菜谱**（写在纸上的步骤） | **Skill** | 只是文字，不能替你炒菜；但没它，厨师可能乱放调料 |
| **灶台 / 锅铲** | **Tool** | 真家伙，能把菜做出来；但没菜谱，它不知道要炒什么 |
| **厨师** | **LLM** | 读了菜谱，动手用锅铲，最后端盘子 |

另一个更好记的类比：

> **Skill = 医院的诊疗方案**（写清楚"先验血、再拍片、最后开药"）
> **Tool = 手术刀和化验机**（真能验、真能切）
> **LLM = 主治医生**（按方案决定用哪个器械）

方案写错了顶多治疗不规范；器械用错了（比如 `eval()` 被执行注入）会出真事故——这就是第 8 节要讲的。

---

## 二、为什么容易混：它俩的"骨架"一模一样

第一眼看代码，两者确实像双胞胎——都是**声明 → 注册 → 被 Agent 取用**：

| 环节 | Skill | Tool |
|---|---|---|
| 基类 | `BaseSkill(ABC)`，用**类属性**声明 | `BaseTool(ABC)`，用**构造参数**声明 |
| 唯一标识 | `name = "math_calculator"` | `name = "calculator"` |
| 说明文字 | `description`（给匹配器和人看） | `description`（**给 LLM 看**，影响它选不选） |
| 注册表 | `SkillRegistry._skills: Dict[str, BaseSkill]` | `ToolRegistry._tools: Dict[str, BaseTool]` |
| 注册方法 | `register()` / `register_batch()` | `register()` / `register_batch()` |
| 重名策略 | 覆盖 + warning | 覆盖 + warning |
| 装配位置 | `main.py:97-98` | `main.py:91-92` + `112`（MCP 并入） |
| 注入 Agent | `ReActAgent(skill_registry=...)` | `ReActAgent(tool_registry=...)` |

所以"看着像"的感觉是对的：**它们是同一套插件体系的两个变体**。区别全在**运行时到底干了什么**。

---

## 三、核心差异（每条都带例子）

| 维度 | Skill（技能） | Tool（工具） | 一句话例子 |
|---|---|---|---|
| **本质** | 一段**提示词文本**（SOP） | 一段**可执行代码** | Skill 是 `"第2条：禁止心算"`；Tool 是 `ast.parse("3*7")` |
| **会执行吗** | 不会，只产出一段字符串 | 会，`execute()` 真跑逻辑 | 调用 Skill 0 次副作用；调用 Tool 会真的算出 21 |
| **有返回值吗** | 没有 | 有 `ToolResult` | `ToolResult(success=True, result=21, duration_ms=12)` |
| **谁决定启用** | **Python 代码**（关键词打分） | **LLM**（Function Calling） | 说"订单"必激活数据查询；但模型可以不理它 |
| **确定性** | 确定（同输入必同结果） | 概率（模型可能忘了调） | Python 打分可复现；LLM 决策不能保证 |
| **触发依据** | `trigger_keywords` + 阈值 0.35 | `parameters_schema`（JSON Schema） | `"订单"(0.4)`；`{"expression": "string"}` |
| **生效时机** | `prepare` 节点，**一次** | `reason` 决策 + `act` 执行，**可循环多次** | Skill 只在开头出现；Tool 可能被调 3 轮 |
| **怎么影响 LLM** | 塞进 prompt 的 `skill_context` 槽位 | 作为 `tools` 参数 + 结果进 `observation` | 一个是"读到的字"，一个是"能按的按钮" |
| **失败会怎样** | 不匹配 → 悄悄走通用流程 | 报错 → `error` 文本回喂 LLM | Skill 失败=没有指导书；Tool 失败=模型读到报错后换招 |
| **独有字段** | `trigger_keywords` / `priority` / `bound_tools` / `instructions` | `parameters_schema` / `tool_type`(local\|mcp) | — |
| **抽象方法** | 无（`BaseSkill` 没有 `abstractmethod`） | 有 `execute()`，子类必须实现 | 所以 Skill 是"配置"，Tool 是"实现" |
| **当前数量** | 4 个 | 5 个（2 本地 + 3 MCP） | 本地 `calculator`/`data_query`；MCP `web_search`/`get_current_time`/`knowledge_stats` |

**最关键的一行**：`谁决定启用`。
- Skill 是 **Python 说了算**（`if score >= 0.35`），所以它是**确定性**的；
- Tool 是 **LLM 说了算**，所以它是**概率性**的。

---

## 四、两条链路的形状完全不同

### Skill：单向注入，用完即弃

```
用户提问 "3乘7等于几"
   ↓
prepare 节点   skill_registry.build_skill_context(clean_query)
                 ├─ 每个技能 match_score(query)
                 ├─ 排序（score 降序，同分看 priority）
                 └─ 最高分 0.75 >= 0.35 → 激活数学计算技能
   ↓
state.skill_context = "【激活技能】数学计算\n命中意图词: 等于几, 乘\n…"
   ↓
reason 节点    拼进 REASON_PROMPT 的 {skill_context} 槽位
   ↓
        —— Skill 的使命到此结束，后面所有轮次都不再出现 ——
```

形状：**一条直线，箭头只朝一个方向，没有回头路。**

### Tool：闭环反馈，可以转好几圈

```
main.py:91/112  tool_registry.register(...)     ← 本地 2 个 + MCP 3 个，混在同一张表
   ↓
reason 节点     tools_schema = get_openai_tools_schema()   ← 全量 5 个，不按技能过滤
               llm.invoke_with_tools(messages, tools=tools_schema)
   ↓           LLM 返回 tool_calls → state.action = {"name": "calculator", ...}
条件边         need_tool_call=True → 走 act
   ↓
act 节点       tool_registry.execute_tool("calculator", {"expression": "3*7"})
                 └─ BaseTool.run: validate_params → execute → ToolResult
   ↓           state.observation = "21"
固定边         回到 reason → LLM 看到 21，决定"还要不要再调"（可多轮）
```

形状：**一个圈，结果会反喂给 LLM，LLM 看到结果后再决定下一步。**

> 形状差异直接决定了：Skill 只能"影响一次"，Tool 能"纠错多次"。
> Tool 算错了，LLM 看到 observation 还能重调；Skill 注入错了，那轮就错到底。

---

## 五、Skill 怎么被选中：打分实例

打分规则（`base_skill.py:56-59`）：

| 关键词长度 | 权重 | 设计意图 |
|---|---|---|
| >= 3 字（如 `等于多少`、`查一下`） | **0.6** | 长短语，意图强 |
| 2 字（如 `订单`、`年假`） | **0.4** | 双字词，意图中等 |
| 1 字（如 `乘`、`除`） | **0.15** | 弱词，单独出现不足以激活 |

得分 = 命中词权重**累加**（封顶 1.0）；阈值 **0.35**（`skill_registry.py:26`）。

实际算几个例子：

| 用户输入 | 命中技能 | 命中的词（权重） | 总分 | 结果 |
|---|---|---|---|---|
| `订单` | 企业数据查询 | 订单(0.4) | **0.40** | 激活（刚过线） |
| `除` | 数学计算 | 除(0.15) | **0.15** | **不激活**（单个弱词不够，防闲聊误命中） |
| `3乘7等于几` | 数学计算 | 等于几(0.6) + 乘(0.15) | **0.75** | 激活 |
| `公司规定年假几天` | 知识库问答 | 公司规定(0.6) + 年假(0.4) | **1.00** | 激活 |
| `你好呀` | 无 | 无 | **0.00** | 不激活 → 走通用助手流程 |

阈值 0.35 的取值逻辑就在表里：**命中 1 个双字词（0.4）就能过，只有 1 个单字词（0.15）就过不了。**

还有一种情况是"平局"：两个技能得分相同（比如都 1.0），这时看 `priority`，数字大的赢。所以知识库问答的 `priority` 故意设成 **5**（低于计算/数据查询的 **10**）——它的触发词太宽泛（"制度""政策"），让语义更明确的技能优先拿走请求。

---

## 六、Skill 注入的原文长什么样

`BaseSkill.format_context()`（`base_skill.py:88-101`）把技能渲染成文本：

```python
# 状态：命中数学计算技能，命中词 ["等于几", "乘"]
skill.format_context(match)
```

输出：

```
【激活技能】数学计算
命中意图词: 等于几, 乘
技能说明: 处理四则运算、幂运算、三角函数、开方、取模等各类数学计算问题，保证结果精确。
推荐工具: calculator
作业流程(SOP):
1. 从用户问题中识别已知数值与运算关系，转换为合法数学表达式（如 '2加3' → '2+3'）；
2. 必须调用 calculator 函数完成计算，禁止凭记忆心算给出数值结果；
3. 向用户展示「计算式 → 结果」，带单位的问题需注明单位；
4. 若表达式无法唯一确定（如缺少必要数值），先向用户追问澄清，不要猜测参数。
```

三行结构很清晰：

| 行 | 作用 |
|---|---|
| `【激活技能】xxx` + `命中意图词: ...` | 让 LLM 知道"你现在走的是这条流程"，也方便人 debug |
| `技能说明` + `推荐工具` | 告诉 LLM 这个技能管什么、建议用哪个工具 |
| `作业流程(SOP)` | **核心**，逐条约束 LLM 行为 |

`bound_tools` 为空时（如知识库问答），"推荐工具"那行会渲染成：

```
推荐工具: （无专用工具，依据知识库与通用能力处理）
```

---

## 七、最容易误解的一点：`bound_tools` 是"推荐"，不是"限制"

很多人以为写了 `bound_tools = ["calculator"]` 就等于"别的工具都被屏蔽了"。**不是。**

### 证据 1：注册表自己的注释

```13:13:modules/skills/skill_registry.py
- 技能只做「软引导」：注入 SOP 与推荐工具，不硬屏蔽其他工具，避免误匹配导致无法作答。
```

### 证据 2：`reason_step` 拿的是全量工具，没做任何过滤

```189:194:modules/agent/react_agent.py
        if self.tool_registry:
            tools_schema = self.tool_registry.get_openai_tools_schema()
            tools_desc = "\n".join(
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in tools_schema
            )
```

### 具体例子

用户说 `帮我算一下库存还有多少`——这句里既有"算一下"（数学技能触发词 0.6），又有"库存"（数据查询技能触发词 0.4）：

```
数学计算: 算一下(0.6) = 0.60
企业数据查询: 库存(0.4) = 0.40    → 同分？不，数学更高 → 数学技能胜出
```

结果：数学技能被激活，prompt 里写着"推荐工具: calculator"。**但因为工具清单是全量的，`data_query` 依然在列表里**，LLM 读到"库存"两个字，完全可以直接改调 `data_query`，照样把问题答对。

### 为什么故意这么设计？

硬屏蔽有真实风险：一旦技能误匹配（把"算一下库存"判成数学计算），再把 `data_query` 屏蔽掉，就**彻底答不了**了。

> **软引导最坏的结果是"引导错了"，硬屏蔽最坏的结果是"堵死了"。**
> 前者 LLM 还能自救，后者只能报错。这是有意为之，不是 bug。

---

## 八、既然有 Tool 了，为什么还要 Skill？

既然 LLM 能自己看工具清单挑，为什么还要多一层 Skill？它解决三件 Tool 解决不了的事：

| 价值 | 具体说明 | 例子 |
|---|---|---|
| **1. 固化流程** | 把"老员工怎么干"写成文本，改流程只改文字、不改代码 | 想加"必须先追问再计算"，只改 `instructions` 字符串，不用动 `calculator.py` |
| **2. 省钱又提速** | 意图路由用 Python 关键词打分，**0 次 LLM 调用** | 每次提问先花一次 LLM 调用做意图分类，成本翻倍 |
| **3. 可预测、可审计** | 命中结果记进 `state.active_skill`，能回溯"这轮走的哪个流程" | 关键词匹配结果稳定可复现，出问题能定位 |

最典型的是"**约束模型行为**"，看数学技能的 SOP 第 2 条：

```34:34:modules/skills/builtin_skills.py
        "2. 必须调用 calculator 函数完成计算，禁止凭记忆心算给出数值结果；\n"
```

这句话完美说明了互补关系：

> **Skill 负责"不许你心算"，Tool 负责"精确算出 21"。**
> **Skill 是纪律，Tool 是能力。**

没有 Skill，LLM 很可能张嘴就说"3乘7等于21"——这类简单题它答对，但换成 `365*24*1.13` 就危险了。Skill 的价值是把"你必须动手算"变成硬性规矩。

---

## 九、别忘了一件事：Tool 是真代码，有真实风险

Skill 只是文本，永远不会带来执行风险。Tool 会真的跑代码，所以必须防注入。

`calculator` 没有用 `eval()`，原因看注释：

```6:9:modules/tools/calculator.py
【为什么不用 eval()？】eval 能直接算 "3*7"，但也能跑 "os.system('rm -rf /')"。
    用 Python 自带的 ast 模块把表达式解析成"语法树"，再逐节点白名单求值——
    遇到 _SAFE_OPS / _SAFE_FUNCS 之外的节点（如调用 open/import os）直接抛错，
    类似前端用 DOMPurify 过滤 XSS，先把能干的事圈死。
```

实际效果对比：

| 输入表达式 | 结果 |
|---|---|
| `"3*7"` | `ToolResult(success=True, result=21)` |
| `"sqrt(16)"` | `ToolResult(success=True, result=4)` |
| `"2**10"` | `ToolResult(success=True, result=1024)` |
| `"__import__('os')"` | `ToolResult(success=False, error="计算失败: 不支持的函数: __import__")` |
| `"open('/etc/passwd')"` | `ToolResult(success=False, error="计算失败: 不支持的函数: open")` |
| `"1/0"` | `ToolResult(success=False, error="计算失败: division by zero")` |

机制是两步：

```105:106:modules/tools/calculator.py
            tree = ast.parse(expr, mode="eval")
            result = self._eval_node(tree)
```

1. `ast.parse` 把字符串变成**语法树**（不是直接执行）；
2. `_eval_node` 递归遍历，只认白名单里的运算符（`+ - * / ** % //`）和函数（`sin` / `sqrt` / `log` 等），碰到别的立刻 `ValueError`。

> 这是 Skill 与 Tool "本质不同"最实在的体现：**一个要防代码注入，一个不用。**
> 也解释了为什么 Tool 有 `ToolResult.success` 字段——真代码就会失败，失败信息还得回喂给 LLM。

---

## 十、速查表

| 问题 | 答案 |
|---|---|
| Skill 会执行代码吗 | 不会，只注入一段文本 |
| Tool 会执行代码吗 | 会，`execute()` 是抽象方法，子类必须实现 |
| 谁决定用哪个 Skill | **Python 代码**（关键词打分，阈值 0.35） |
| 谁决定用哪个 Tool | **LLM**（Function Calling） |
| Skill 的确定性 | 确定（同输入同结果，可复现） |
| Tool 的确定性 | 概率（模型可能忘调、可能调错参数） |
| `bound_tools` 会屏蔽其它工具吗 | **不会**，只是 prompt 里的一行"推荐工具" |
| Skill 能没有 Tool 吗 | 能（`knowledge_qa` 的 `bound_tools = []`，走 RAG） |
| Tool 能没有 Skill 吗 | 能（未命中任何技能时，LLM 照样自由调用 5 个工具） |
| 一个 Skill 能绑几个 Tool | 0~N（`web_search` 技能绑了 `web_search` + `get_current_time` 两个） |
| Skill 在哪生效 | `prepare` 节点，**一次性**注入 `skill_context` |
| Tool 在哪生效 | `reason` 决策 + `act` 执行，**可循环多次** |
| 匹配失败会怎样 | Skill：静默走通用流程；Tool：`error` 文本回喂 LLM |
| 两者唯一必填标识 | 都是 `name`（注册表的键） |
| 当前各有多少个 | Skill 4 个；Tool 5 个（2 本地 + 3 MCP） |

---

## 十一、一句话总结

> **Skill 是写给 LLM 看的作业指导书（纯文本，Python 打分选中，只注入一次），
> Tool 是 LLM 能按的按钮（真代码，LLM 通过 Function Calling 决定按不按，结果会反喂形成循环）。
> Skill 管"按什么流程做"（纪律），Tool 管"把活干出来"（能力）；
> `bound_tools` 只是推荐不是封锁，两者都可以脱离对方单独存在。**
