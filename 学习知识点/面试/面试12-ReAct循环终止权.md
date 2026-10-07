# 面试12：ReAct 循环什么时候停？终止权在谁手里？

> 配套代码：`modules/agent/graph.py:355-381`（_should_continue）、`modules/agent/react_agent.py:654、736-739`（max_iterations 强制收尾）、`react_agent.py:671-721`（loop_guard 三道刹车）、`graph.py:414`（recursion_limit=30）、`modules/tools/tool_registry.py:90、355-356`（去重与限次）
> 一句话结论：**终止权名义上在模型手里（它不给 tool_calls 就走 answer），实际上框架握着多道强制闸——单工具 5 次上限、同参去重、loop_guard 三道刹车、max_iterations=10 强制收尾，外加图层 recursion_limit=30 兜底；模型只有"提前停"的权利，没有"不停"的权利。**

---

## 〇、名词先扫盲（大白话版）

| 术语 | 大白话解释 |
|---|---|
| **ReAct 循环** | "思考 → 调工具 → 看结果 → 再思考"的转圈过程，直到模型说"不用调了，我直接答"。像做菜：尝一口（观察）→ 缺盐加盐（调工具）→ 再尝 → 好了出锅 |
| **tool_calls** | 模型输出里的"我要调某工具"请求。输出里没有它 = 模型要直接作答 = 循环该停了 |
| **条件边 / _should_continue** | 图里的"红绿灯"：每轮思考完看一眼状态里的旗子，决定下一站是调工具（act）还是收尾（answer） |
| **finished / need_tool_call** | 状态里的两面旗：`finished=True` = 模型说"完工"；`need_tool_call=True` = 模型说"还要干活" |
| **max_iterations=10** | 轮次硬顶：转到第 10 轮还想调工具，框架强制收尾——防止模型在"调工具-看结果-再调"里无限打转 |
| **recursion_limit=30** | LangGraph 图层限步：一次运行整张图最多跳 30 步，业务闸全部失灵时的最后保险丝 |
| **去重 / 结果缓存** | 同一个工具用同样的参数调第二次，不再真执行，直接把上次的结果给它——省一次工具调用，还给模型一个"别再调了"的信号 |

---

## 一、先说结论

| 面试问 | 一句话答 |
|---|---|
| 谁决定停？ | 模型提议（不给工具调用即收尾），框架批准（多道闸随时可否决） |
| 模型怎么"提议停"？ | reason 节点输出里没有 tool_calls → `need_tool_call=False` → 条件边走 answer |
| 框架的强制闸？ | 单工具 5 次上限 + 同参去重回灌 + loop_guard 三道刹车 + max_iterations=10 + 图层 recursion_limit=30 |
| 防烧 token 靠什么？ | 预算闸（次数/token/墙钟）+ 结果缓存回传（重复调用直接给上次答案，省一轮 LLM） |

---

## 二、大白话拆解：一个循环的控制点全景

```
reason（LLM 推理，react_agent.py:461 reason_step）
   ├─ :494  iteration += 1（轮次从 1 起算）
   ├─ :498+ 预算闸：墙钟超时 / token 打满 → 不问模型直接收尾
   ├─ :654  tool_calls 有 且 iteration < max_iterations？
   │     ├─ :671 闸A 同名同参去重 → 命中：回灌上次结果，need_tool_call=False
   │     ├─ :692 闸B 交替环（A→B→A→B 周期 2~3）→ 命中：解释回灌，强制收敛
   │     ├─ :711 闸C 无进展（连续 3 次结果一字不差）→ 命中：强制收敛
   │     └─ 全没命中 → need_tool_call=True，去 act
   └─ :736  iteration >= max_iterations 且还想调工具 → 强制 need_tool_call=False
   ↓ 输出带 tool_calls？
   ↓
graph.py:355  _should_continue（条件边，"红绿灯"）
   ├── finished=True                    → answer   （模型主动收尾）
   ├── need_tool_call 且 无 error       → act      （继续循环）
   └── 其余（含出错）                    → answer   （出错也走 answer，防止卡死在循环里）
   ↓
act（执行工具，graph.py:208）→ 回到 reason …… 循环
   ↓ 工具层还有两道闸（tool_registry.py）：
   闸D  单工具累计 ≤ 5 次        tool_registry.py:90  _max_calls_per_tool
   闸E  同名同参去重              tool_registry.py:355-356  _call_records + _result_cache
   ↓ 全图兜底：
   闸F  recursion_limit=30（graph.py:414）——前面所有闸全失灵时的最后保险
```

核心代码片段（modules/agent/graph.py:377-381）：

```python
# graph.py:377-381 —— 整张图的"红绿灯"，返回值决定下一跳
if state.get("finished"):
    return "answer"
if state.get("need_tool_call") and not state.get("error"):
    return "act"
return "answer"
```

核心代码片段（modules/agent/react_agent.py:736-739，轮次硬顶强制收尾）：

```python
# react_agent.py:736-739 —— 到顶了还想调工具：强制作答，不再循环
if state.iteration >= state.max_iterations and state.need_tool_call:
    _logger.warning(f"达到最大循环次数{state.max_iterations}，强制终止ReAct循环")
    state.need_tool_call = False
    state.final_answer = result["content"] or "已达到最大推理次数，基于当前信息给出最佳答案。"
```

| 部分 | 大白话解释 |
|---|---|
| `_should_continue` | 红绿灯：看 state 里的旗子决定下一跳是 act 还是 answer |
| `finished` | 模型举的"完工"旗——它给出最终答案时置 True |
| `need_tool_call` | 模型举的"还要干活"旗——输出里有 tool_calls 时置 True |
| `max_iterations=10` | 框架的硬顶：到 10 轮还想调工具？强制收尾，不再循环（config.py:103） |
| `recursion_limit=30` | LangGraph 图层限步：单次 invoke 内图最多跳 30 步，防前几道闸之外的路径失控 |

---

## 三、输入 → 输出对照

```python
# graph.py:377-381（条件边判定）
# 输入 state ①（模型收尾）: {"finished": True, "need_tool_call": False}
# 输出: "answer"      ← 循环结束

# 输入 state ②（继续干活）: {"finished": False, "need_tool_call": True, "error": None}
# 输出: "act"        → act 执行完回 reason，下一圈

# 输入 state ③（工具报错）: {"finished": False, "need_tool_call": True, "error": "timeout"}
# 输出: "answer"     ← 出错也走 answer：防止带着 error 状态在循环里打转

# react_agent.py:736-739（第 10 轮还想调工具）
# 输入: state.iteration=10, state.max_iterations=10, state.need_tool_call=True
# 执行: need_tool_call=False（强制降旗）, final_answer="已达到最大推理次数…"
# 日志: 达到最大循环次数10，强制终止ReAct循环

# react_agent.py:671-682（同名同参去重，reason 侧刹车）
# 输入: 本次动作 calculator('3*7')，历史里已有同名同参且成功的一条（result='21'）
# 输出: need_tool_call=False，observation="工具 calculator 已用相同参数执行过，
#        结果如下（不要重复调用，直接据此作答）：…21"

# 去重缓存（tool_registry.py:355-356，工具层刹车）——防烧 token 的关键一环
# 第 2 次同名同参调用 calculator('3*7') 时:
# 输出: success=True, result=21, duration_ms=0   ← 直接回传上次答案
#   不回传而是报"拒绝重复" → 模型以为工具坏了换个说法再调 → 实测 ReAct 跑满 10 轮、
#   同一工具调 9 次，单次对话拖到 13 秒（约 10 倍 token）
```

---

## 四、关键设计对照表（六道闸的分工）

| 闸 | 位置 | 抓什么 | 触发后动作 |
|---|---|---|---|
| ① 同名同参去重 | react_agent.py:671（reason 内） | 原地打转：同一工具同一参数再调一次 | 回灌上次结果，直接收尾 |
| ② 交替环检测 | react_agent.py:692（loop_guard） | 换着花样打转：A→B→A→B（周期 2~3） | 解释性回灌，强制收敛 |
| ③ 无进展检测 | react_agent.py:711（loop_guard） | 动作不同但连续 3 次结果一字不差 | 回灌最新结果，强制收敛 |
| ④ 单工具 5 次上限 | tool_registry.py:90 | 对一个工具死磕 | 判为死循环，拦截 |
| ⑤ max_iterations=10 | react_agent.py:736 | 轮次到顶还想调工具 | 强制 need_tool_call=False |
| ⑥ recursion_limit=30 | graph.py:414 | 以上全失灵（如图结构被改出意外环路） | LangGraph 抛异常终止 |

设计要点：**①②③ 在"还没执行"时拦截（省一次注定空调的工具调用），④ 在执行层拦截，⑤ 在轮次级兜底，⑥ 在框架级兜底**——由细到粗、层层独立。

---

## 五、为什么要这么写

| 做法 | 原因（不这么写会怎样） |
|---|---|
| 出错也路由到 answer | error 状态下继续循环：模型看到报错继续调 → 再报错 → 死循环在错误里打转 |
| 多道闸分级（工具级 → 轮级 → 图级） | 只有模型自觉：幻觉循环（模型坚信"再调一次就好"）会无限烧 token；多层独立，任何一层失灵还有下一层 |
| 去重命中回传**上次答案**而非报错 | 报错给模型的信号是"工具坏了"→ 它换说法重试 → 灾难循环；给答案它才有依据收尾 |
| 只缓存成功结果 | 失败结果被回传，模型会以为"答案就是这样"，失去改参数重试的机会 |
| 刹车时回灌结果/解释进 observation | 只降旗不给料：模型没有作答依据，只能硬编——observation 里写明"不要重复调用，直接据此作答"，模型有指令可循 |
| `max_iterations=10` 而不是 3 | 复杂任务（查库→计算→再查）合理路径就要 4-6 轮；顶太低任务做不完 |
| recursion_limit=30 独立于业务计数 | 业务计数可能被绕过（如 plan 分支的循环），图层兜底不依赖业务逻辑 |

---

## 六、可能的追问与回答要点

**Q1：模型"提议停"和框架"强制停"冲突时听谁的？**
答：框架。准确说没有"冲突"场景——模型想停（无 tool_calls）时框架不会拦；模型不想停时框架的闸随时可以把它摁住（react_agent.py:736 直接把 need_tool_call 改成 False）。所以分工是：**模型只有"提前停"的权利，没有"不停"的权利**。

**Q2：为什么出错也要走 answer 而不是重试？**
答：① 循环内重试没有新信息输入，同样的错误大概率再犯——原地重试就是打转；② 走 answer 节点至少能基于已有信息给用户一个交代（"工具暂时不可用，基于已有信息…"），体验好于卡死；③ 真要重试属于"下一轮对话"或"上游网关重试"的职责，不是循环内的事。

**Q3：为什么 max_iterations 用 >= 而不是 >？**
答：iteration 从 1 起算（react_agent.py:494 进 reason 先 +1），等于 10 时这一轮已经是第 10 轮，不能再开第 11 轮。用 > 会实际放到 11 轮。这类 off-by-one 是循环上限实现里最常见的 bug，面试答出来是加分项。

**Q4：10 轮还不够怎么办（任务真的复杂）？**
答：① 配置项可调（config.py:103 `max_react_iterations`）；② 但正确方向不是放大上限，而是上 Plan-and-Execute：复杂任务先拆成有序子任务（graph.py 的 plan 节点），每个子任务自己的循环收敛得快——把"一个大循环跑 20 轮"变成"5 步 × 每步 3 轮"；③ 上限本质是"模型已经迷路"的信号，放大上限只是让它迷路更久。

**Q5：去重键怎么算的？参数顺序不同算同一次吗？**
答：去重键 = `工具名:参数指纹`，指纹是参数项排序后的列表（tool_registry.py:68 注释）——所以 `{'a':1,'b':2}` 和 `{'b':2,'a':1}` 算同一次，语义相同仅顺序不同不会逃过去重。reason 侧的同名同参比对（react_agent.py:758-759）用的是 `arguments` 字典直接相等，Python dict 相等本身也不看顺序，两侧口径一致。

**Q6：这些闸会不会误杀正常任务？**
答：有可能，设计上有三处"防误杀"：① 同名同参只比对"成功过"的记录（react_agent.py:759 要求 success=True），失败重试不会被拦；② 无进展闸默认要连续 3 次结果全等才触发（_stall_rounds=3），偶尔重复不算；③ loop_guard 有总开关 `agent_loop_guard_enabled`（react_agent.py:685），线上误杀可一键关掉只留基础去重。

---

## 七、简单总结

1. 终止权分工：**模型有"提前停"的权利（finished 旗），框架有"必须停"的权力（六道闸）**——`_should_continue` 是裁判（graph.py:355）
2. 关键数字：单工具 **5 次**（tool_registry.py:90）、去重键 **工具名:参数指纹**、ReAct 上限 **10 轮**（config.py:103）、图层 **recursion_limit=30**（graph.py:414）
3. 防烧 token 的隐藏功臣是**结果缓存回传**：重复调用直接给上次答案（省一轮 LLM），报错式拒绝会把模型逼进"换个说法重试"的灾难循环
