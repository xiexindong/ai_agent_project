用一个具体问句走一遍完整链路。取项目里的真实例子：

> **用户问：`"我的报销单到哪一步了？"`**

走的是 ReAct 链路（`react.reason` + `strategy="react"`），对应 `react_agent.py` 的真实配置。

---

## Step 0 · 构造期（整个会话只做一次）

```77:85:modules/agent/react_agent.py
role = get_registry().render("react.role", max_iter=self._max_iter)
self._builder = PromptBuilder(
    template_name="react.reason",
    strategy="react",
    role=role,
    max_iter=self._max_iter,
    max_context_tokens=s.max_prompt_context_tokens,
)
self.system_prompt = self._builder.system_prompt()
```

产出 `role` = `react.role` 模板渲染结果（`max_iter=10` 被填进 "最多循环10次"）。

| 工程点 | 说明 |
|---|---|
| 提示词**按名取**，不是硬编码 | `"react.role"` 一个名字换掉整段文案 |
| `max_iter` 是**变量** | 配置里改一处，全局所有提示词同步生效，不用翻字符串 |
| 预算来自**配置** | `MAX_PROMPT_CONTEXT_TOKENS`，不是散落的魔法数 |
| system **只算一次** | 多轮时 system 不变，服务端可命中前缀缓存（省 token 省钱） |

---

## Step 1 · 取模板，自动扫出变量表

```118:120:modules/prompt/builder.py
registry = self.registry or get_registry()
tpl = registry.get(self.template_name, self.template_version)
needed = tpl.variables
```

`needed` 自动算出 8 个槽位：

```
{query} {skill_context} {permission_context} {rag_context}
{memory_context} {tool_history} {observation} {tools_desc}
```

**工程点**：变量表是**代码扫出来的**（`templates.py:83-90`），不是人维护的清单——模板加一个槽位，校验立刻跟上，不会漏。模板没注册时报错还会**列出已注册的有哪些**（183 行），直接告诉你拼错成了什么。

---

## Step 2 · 判断"策略段放哪"（防重复烧钱）

```122:131:modules/prompt/builder.py
【策略段到底进 user 还是进 system】取决于模板有没有 {strategy_block} 插位：
   有插位（qa.grounded）→ 策略段作为变量渲染进 user，system 里不再拼；
   没插位（react.reason）→ 策略段整段进 system，user 只放上下文。
   两条路只走一条，否则同一段指令会出现两次，白烧 token。

【工具清单同理】strategy=react 的策略块自带「## 可用工具」，
   而 react.reason 模板也有 {tools_desc} 插位，两处都填就是同一份清单两遍。
```

`react.reason` 没有 `{strategy_block}` → 策略进 **system**，且 tools_desc 也要让给模板（149 行传空）。

**工程点**：这是**为成本写的判断**。同一份工具清单填两遍 = token 翻倍，代码里专门防了这件事，还写了为什么。

---

## Step 3 · 上下文装包：最核心的工程环节

```228:262:modules/prompt/builder.py
ordered = [k for k in _CONTEXT_PRIORITY if k in needed and k in _CONTEXT_KEYS]
filled = {k: str(context.get(k) or _EMPTY_PLACEHOLDER) for k in ordered}
budget = self.max_context_tokens     # 3000
for key in ordered:
    cost = count_tokens(text)
    if used + cost <= budget:      # 塞得下 → 整块留
        kept[key] = text; used += cost
    elif remain >= self._MIN_KEEP_TOKENS:   # 塞不下 → 截断
        kept[key] = self._clip(text, remain)
        trimmed.append(f"{key}: 截断 {cost} → {count_tokens(clipped)} token")
    else:                           # 剩太少 → 丢弃，但仍填占位符
        kept[key] = _EMPTY_PLACEHOLDER
        trimmed.append(f"{key}: 丢弃 {cost} token（预算已满）")
```

本轮实际发生什么（用 demo 第四幕的实测数字，预算 2000 的情形）：

| 块 | 原始 | 处理后 |
|---|---|---|
| `permission_context` | 60 | 60 ✅ 全留（优先级最高） |
| `tools_desc` | 120 | 120 ✅ |
| `observation` | 40 | 40 ✅ |
| `skill_context` | 300 | 300 ✅ |
| `rag_context` | **2040** | **截断 → 1987** |
| `memory_context` | 80 | **丢弃**（预算已满） |
| `tool_history` | 50 | **丢弃**（填"未提供"） |

**这一步集合了 6 个工程决策**：

| 决策 | 代码 | 为什么 |
|---|---|---|
| **优先级排序** | `_CONTEXT_PRIORITY:35-43` | 权限丢了会**越权**（安全问题）；记忆丢了只是不贴人 |
| **没传就填占位** | 235 行 | 保证模板一定能渲染出来 |
| **残羹冷炙阈值 50** | `_MIN_KEEP_TOKENS:88` | 半截上下文比没有**更误导**（看着像有依据，其实被砍断） |
| **丢弃也填占位符** | 260 行 | 不填 → `render()` 抛错，**一次裁剪把整轮对话搞挂** |
| **query 永不截断** | 224 行注释 | 砍 query 等于换了个问题 |
| **全程记账** | `trimmed` | 谁被砍了可查，否则只能猜"为什么这次答得差" |

手艺阶段对应的是什么？`REASON_PROMPT.format(rag_context=整段)`——**塞进去就完事**，超长就爆窗口，爆了也不知道哪块爆的。

---

## Step 4 · 渲染 user（先校验再 format）

```145:145:modules/prompt/builder.py
user = tpl.render(**ctx)
```

`render` 内部（`templates.py:110-122`）：

```python
missing = sorted(need - set(kwargs))
if missing:
    raise PromptError(f"模板 {self.name}({self.version}) 缺少变量", ...)
extra = sorted(set(kwargs) - need)
if extra:
    _logger.warning(f"模板 {self.name} 收到多余变量 {extra}，已忽略")
```

**工程点**在 95-97 行的注释里说透了：

> 为什么不能静默填空字符串：prompt 少一段上下文，模型照样会回答，只是答案质量悄悄变差且不报错 —— 这是最难排查的一类故障（**静默降质**）。

**缺 → 抛错，多 → 告警**，两个方向策略不同，是权衡过的（95-100 行）。

渲染出的 `user` 成品：

```
## 用户提问
我的报销单到哪一步了？

## 激活技能（SOP作业流程）
报销进度查询：先查报销单状态，再说明当前审批节点...

## 权限边界（硬性约束，越界即违规）
可访问：本人报销单；禁止：他人单据、薪资数据

## 知识库参考
《差旅报销管理办法》第三条：...（已截断）

## 历史记忆
（未提供）

## 历史工具调用
（未提供）

## 工具观察结果
（初始状态，暂无工具观察）

## 可用工具（Function Calling）
query_reimbursement / data_query / calculator

请分析当前情况，决定下一步行动：
...
```

注意那两个 **`（未提供）`**——不是空白，是**明确告诉模型"这块没有"**，比留空更不容易瞎编。

---

## Step 5 · 拼 system（三层叠加）

```190:212:modules/prompt/builder.py
parts = [role or self.role]                      # ① 角色层
if include_strategy:
    parts.append(build_strategy_block(...))      # ② 策略层
fmt = self._format_instruction()                 # ③ 输出层
if fmt: parts.append(fmt)
return "\n\n".join(p for p in parts if p)
```

策略层由 `build_strategy_block("react", max_iter=10, tools_desc=...)` 生成（`strategies.py:217-218`）：

```
## 工作流程（ReAct）
1. Reason：分析当前情况，判断是否需要调用工具
2. Action：如需工具，选择合适的工具并确认参数
3. Observation：观察工具返回，决定继续调用还是直接作答
- 最多循环10次，超过后直接给出当前最佳答案
- 同一个工具用相同参数重复调用没有意义，看到结果就该作答

## 可用工具
query_reimbursement / data_query / calculator
```

**最后那一行是工程的最好证据**——`builtin.py:58-61` 写明了它的来历：

> 补了「同一个工具用相同参数重复调用没有意义」这条，针对的就是**实测到的现象：同一句"3*7"曾被连续调 9 次计算器**。

**不是"我觉得这句写得更清楚"，是"我观测到了问题 → 改提示词 → 记录原因"。**

---

## Step 6 · 装成 messages 并计数

```153:162:modules/prompt/builder.py
messages = []
if system:
    messages.append({"role": "system", "content": system})
messages.append({"role": "user", "content": user})
prompt_tokens = count_tokens(system) + count_tokens(user)
```

**工程点**：system / user 分离。多轮对话时 system 字节不变 → 命中 **prefix caching**，省 token 也省钱；`react_agent.py:85` 把 system 缓存起来正是为了这个。

---

## Step 7 · 返回的不只是 prompt，还有一份"体检报告"

```163:172:modules/prompt/builder.py
return {
    "messages": messages,
    "system": system, "user": user,
    "template": "react.reason(v1)",     # 用的哪个模板哪个版本
    "strategy": "react",
    "prompt_tokens": 2847,              # 这次烧了多少
    "context_tokens": 2104,             # 上下文占了多少
    "trimmed": ["rag_context: 截断 2040 → 1987 token",
                "memory_context: 丢弃 80 token（预算已满）",
                "tool_history: 丢弃 50 token（预算已满）"],
}
```

`react_agent.py:381` 就拿它做了观测：`if rec["trimmed"]: ...打日志`。

**工程点**：**能度量才能优化**。你不知道哪块被砍了，就永远只能猜"为什么这次答得差"。手艺阶段这一步是完全不存在的。

---

## Step 8（可选）· 输出约束与解析形成闭环

如果这轮要求 JSON 输出（`output_format="json"`）：

```
拼装时 → builder._format_instruction() 追加：
    "## 输出格式\n只输出一个 JSON 对象，不要用 ``` 包裹，不要写任何解释文字。"
                                                    ↓
模型返回 '{"intent": "报销咨询", "confidence": 0.9,}'   ← 尾逗号，json.loads 会炸
                                                    ↓
收尾时 → parse_json_output(text)（builder.py:310-313）
        内部用 repair_json 修尾逗号/缺括号 → {"intent": "报销咨询", "confidence": 0.9}
        实在不是 JSON → 返回 None（不假装解析成功）
```

**工程点**：**发出去的约束和收回来的解析是配套设计的**。只约束不解析 = 一半工程；解析失败返回 `None` 而不是硬造一个 dict = 不制造假数据。

---

## 全流程一句话串起来

```
问句 "我的报销单到哪一步了？"
  │
  ├─ ① 取模板 react.reason(v1)          → 自动扫出 8 个变量   【可管理】
  ├─ ② 判断策略段归属                    → 防止填两遍烧钱      【成本意识】
  ├─ ③ 上下文按优先级装包                 → 截断/丢弃/占位/记账 【可降级+可观测】
  ├─ ④ render 先校验再 format            → 缺了抛错不静默降质  【可校验】
  ├─ ⑤ 三层叠加 system                   → 角色/策略/输出互不耦合【可切换】
  ├─ ⑥ system+user 分离                  → 命中前缀缓存        【可优化】
  ├─ ⑦ 返回 prompt + 体检报告            → token/裁剪可查      【可度量】
  └─ ⑧ JSON 约束 + 容错解析              → 闭环，不造假数据    【可约束】
```

## 最能说明"工程"的一件事

**把 `strategy="react"` 改成 `"cot"`，只有 Step 5 里那一段文案会变**，其余七步一模一样：

```python
PromptBuilder(
    template_name="react.reason",
    strategy="cot",        # ← 只改这一个字符串
    ...
)
```

模板不动、上下文不动、裁剪逻辑不动、预算不动。

而在"手艺"阶段，这等价于**去那一大段字符串里把 ReAct 流程那几行删掉、换成 CoT 步骤**——改完你无法确认有没有连带改坏别的，也无法回到上一版，也无法对比两种问法的成本。

**"改一个参数就能换整套问法，且能随时换回来、还能并排放着对比"** —— 这就是"工程"两个字在你的项目里最具体的样子。

要不要我把这份「一个问句的完整组装过程」也写成文档？我可以把它做成 16 号，和之前那份「工程在哪」配成一对：一份讲**为什么**，一份讲**怎么跑**。