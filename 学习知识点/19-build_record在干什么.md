# build_record 这个函数到底在干什么？

> 对应代码：[builder.py#L105-L172](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/prompt/builder.py#L105-L172)
> 重点难懂段：[L132-L145](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/prompt/builder.py#L132-L145)

---

## 一句话目的

**把零散的原料（用户问题 + 一堆上下文）加工成最终发给大模型的 messages，同时附上一张"账单"（token 数 + 裁剪记录）。**

```
原料                                        成品
query="报销标准"            ──build_record──►  messages = [
rag_context="报销需要..."                      {"role":"system", "content":"你是企业助手..."},
memory_context="用户偏好..."                   {"role":"user",   "content":"填好的完整提示词"},
tools_desc="可用工具列表..."                  ]
                                            + 账单：prompt_tokens / context_tokens / trimmed
```

对外的 `build()` 只是它的简写版：`build()` 只要 messages，`build_record()` 还把 token 和裁剪明细带出来，方便排查"这轮提示词里到底塞了什么、哪块被砍了"。

---

## 核心比喻：一张带空格的"填空题"模板

模板就是一段写好的文本，里面挖了空格，用 `{空格名}` 标记：

```
请基于以下资料回答问题。

【知识库参考】
{rag_context}

【历史记忆】
{memory_context}

问题：{query}
```

模板对象有两个关键能力（[templates.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/prompt/templates.py#L84-L122)）：

| 能力 | 作用 |
|---|---|
| `tpl.variables` | 自动扫描模板，列出所有空格名 → `{'rag_context', 'memory_context', 'query'}` |
| `tpl.render(**ctx)` | 把空格一个个填上，输出完整文本；**有空格没填会直接报错** |

所以整个函数的主线就是：**先问模板要哪些空格 → 逐个准备内容 → 填空**。

---

## L132-145 逐行看懂（最难的部分）

```python
tools_desc = str(context.get("tools_desc", "") or "")
ctx, trimmed = self._fit_context(needed, context)
if "strategy_block" in needed:
    ctx["strategy_block"] = build_strategy_block(...)
if "query" in needed:
    ctx["query"] = query
user = tpl.render(**ctx)
```

**第 1 步：取工具清单**
`context.get("tools_desc", "")` —— 调用方传了就取，没传给空串；外面套 `str(... or "")` 保证最终一定是字符串，不会拿到 None。

**第 2 步：上下文按预算裁剪打包**
`_fit_context(needed, context)` 把 rag/memory 这些上下文块按优先级往 token 预算里塞，返回：
- `ctx` = 填好的上下文（塞不下的被截断或填占位符）
- `trimmed` = 裁剪账单（哪些块被动过）

**第 3、4 步：模板要哪个空格，才补哪个**
这就是 `if "xxx" in needed` 的含义——**不同模板挖的空格不一样**：

| 模板 | 有的空格 | 走的分支 |
|---|---|---|
| qa.grounded（知识库问答） | `{strategy_block}`、`{query}`、`{rag_context}`... | 两个 if 都进，策略和问题填进 user |
| react.reason（ReAct 推理） | 没有 `{strategy_block}` | 跳过第一个 if，策略段稍后进 system |

为什么要用 if 包起来？`render` 对"模板没有、你却硬塞"的变量会忽略并警告。先判断 `needed` 再准备，既不做无用功，也让"这段内容去哪了"完全由模板决定。

**第 5 步：填空出成品**
`tpl.render(**ctx)` 把 ctx 里的内容填进空格，得到 user 消息正文。

---

## 那 system 消息呢？（L147-151，跟上面是反着配合的）

```python
system = self.system_prompt(
    tools_desc="" if "tools_desc" in needed else tools_desc,
    include_strategy="strategy_block" not in needed,
)
```

策略段和工具清单**只能出现一次**，所以 user 和 system 是"跷跷板"：

```
                     qa.grounded 模板        react.reason 模板
策略段去了哪？         填进 user（有空格）      进 system（无空格）
工具清单去了哪？       模板自己有               system 补带
                     ↓
判断依据全是 needed：模板有这个空格 → 内容给 user；没有 → system 补上
```

注意这里条件都是 `in needed` / `not in needed`，与上面 L134-144 正好相反，保证同一段指令不会在 system 和 user 里各写一遍（否则白烧 token）。

---

## 最后：组装 + 算账（L153-172）

```python
messages = []
if system:  messages.append({"role": "system", "content": system})  # system 为空就不加
            messages.append({"role": "user",   "content": user})
prompt_tokens = count_tokens(system) + count_tokens(user)           # 总 token
return { "messages": ..., "prompt_tokens": ..., "trimmed": trimmed, ... }
```

---

## JS 类比

就像 JS 模板字符串 + 一个"看模板需要哪些变量"的函数：

```js
// 模板：带空格的字符串
const tpl = '资料：{rag_context}\n问题：{query}'

// variables = 正则扫出空格名；render = 替换空格，缺了就报错
const needed = variables(tpl)                       // ['rag_context', 'query']

function buildRecord(query, context) {
  const ctx = fitContext(needed, context)           // 上下文按预算裁剪
  if (needed.includes('query')) ctx.query = query   // 模板要才给
  const user = render(tpl, ctx)                     // 填空
  const system = systemPrompt({ includeStrategy: !needed.includes('strategy_block') })
  return { messages: [{ role: 'system', content: system },
                      { role: 'user',   content: user }],
           promptTokens: countTokens(system) + countTokens(user) }
}
```

---

## 30 秒记住

1. build_record = **"备料 → 填空 → 组装 → 算账"**，产出 messages + token/裁剪账单。
2. 模板是带 `{空格}` 的填空题；`variables` 列出空格名，`render` 填空、缺了报错。
3. `if "xxx" in needed` 的意思：**模板有这个空格才准备它**，不同模板走不同分支。
4. system 和 user 是跷跷板：同一段策略/工具清单，模板有空格就进 user、没有就进 system，只出现一次。
