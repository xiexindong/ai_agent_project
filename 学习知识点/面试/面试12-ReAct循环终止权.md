# 面试12：ReAct 循环什么时候停？终止权在谁手里？

> 配套代码：`modules/agent/graph.py:342-348`（_should_continue）、`modules/agent/react_agent.py:576、647-650`（max_iterations 强制收尾）、`graph.py:401`（recursion_limit=30）、`modules/tools/tool_registry.py`（去重与限次）
> 一句话结论：**终止权名义上在模型手里（它不给 tool_calls 就走 answer），实际上框架握着三道强制闸——单工具 5 次上限、同参去重、max_iterations=10 强制收尾，外加图层 recursion_limit=30 兜底；模型只有"提前停"的权利，没有"不停"的权利。**

---

## 一、先说结论

| 面试问 | 一句话答 |
|---|---|
| 谁决定停？ | 模型提议（不给工具调用即收尾），框架批准（三道闸随时可否决） |
| 模型怎么"提议停"？ | reason 节点输出里没有 tool_calls → `need_tool_call=False` → 条件边走 answer |
| 框架的强制闸？ | 单工具 5 次上限 + 同参去重 + max_iterations=10 + 图层 recursion_limit=30 |
| 防烧 token 靠什么？ | 预算闸（次数/token/墙钟）+ 结果缓存回传（重复调用直接给上次答案，省一轮 LLM） |

---

## 二、大白话拆解：一个循环的控制点全景

```
reason（LLM 推理）
   ↓ 输出带 tool_calls？
   ↓
graph.py:342  _should_continue（条件边，"红绿灯"）
   ├── finished=True                    → answer   （模型主动收尾）
   ├── need_tool_call 且 无 error       → act      （继续循环）
   └── 其余（含出错）                    → answer   （出错也走 answer，防止卡死在循环里）
   ↓
act（执行工具）→ 回到 reason …… 循环
   ↓ 每圈过三道闸：
   闸1  单工具累计 ≤ 5 次        tool_registry._max_calls_per_tool
   闸2  同名同参去重              _call_records + _result_cache
   闸3  iteration < max_iterations(10)   react_agent.py:647
   ↓ 撞闸强制收尾：
   闸3 触发时: need_tool_call=False, final_answer="已达到最大推理次数…"
   闸4  图层兜底: recursion_limit=30（graph.py:401）——前三道全失灵时的最后保险
```

| 部分 | 大白话解释 |
|---|---|
| `_should_continue` | 红绿灯：看 state 里的旗子决定下一跳是 act 还是 answer |
| `finished` | 模型举的"完工"旗——它给出最终答案时置 True |
| `need_tool_call` | 模型举的"还要干活"旗——输出里有 tool_calls 时置 True |
| `max_iterations=10` | 框架的硬顶：到 10 轮还想调工具？强制收尾，不再循环 |
| `recursion_limit=30` | LangGraph 图层限步：单次 invoke 内图最多跳 30 步，防前三道闸之外的路径失控 |

---

## 三、输入 → 输出对照

```python
# graph.py:342-348（条件边判定）
# 输入 state ①（模型收尾）: {"finished": True, "need_tool_call": False}
# 输出: "answer"      ← 循环结束

# 输入 state ②（继续干活）: {"finished": False, "need_tool_call": True, "error": None}
# 输出: "act"        → act 执行完回 reason，下一圈

# 输入 state ③（工具报错）: {"finished": False, "need_tool_call": True, "error": "timeout"}
# 输出: "answer"     ← 出错也走 answer：防止带着 error 状态在循环里打转

# react_agent.py:647-650（第 10 轮还想调工具）
# 输入: state.iteration=10, state.max_iterations=10, state.need_tool_call=True
# 执行: need_tool_call=False（强制降旗）, final_answer="已达到最大推理次数…"
# 日志: 达到最大循环次数10，强制终止ReAct循环

# 去重缓存（tool_registry）——防烧 token 的关键一环
# 第 2 次同名同参调用 calculator('3*7') 时:
# 输出: success=True, result=21, duration_ms=0   ← 直接回传上次答案
#   不回传而是报"拒绝重复" → 模型以为工具坏了换个说法再调 → 实测 ReAct 跑满 10 轮、
#   同一工具调 9 次，单次对话拖到 13 秒（约 10 倍 token）
```

---

## 四、为什么要这么写

| 做法 | 原因（不这么写会怎样） |
|---|---|
| 出错也路由到 answer | error 状态下继续循环：模型看到报错继续调 → 再报错 → 死循环在错误里打转 |
| 三道闸分级（工具级 → 轮级 → 图级） | 只有模型自觉：幻觉循环（模型坚信"再调一次就好"）会无限烧 token；三层独立，任何一层失灵还有下一层 |
| 去重命中回传**上次答案**而非报错 | 报错给模型的信号是"工具坏了"→ 它换说法重试 → 灾难循环；给答案它才有依据收尾 |
| 只缓存成功结果 | 失败结果被回传，模型会以为"答案就是这样"，失去改参数重试的机会 |
| `max_iterations=10` 而不是 3 | 复杂任务（查库→计算→再查）合理路径就要 4-6 轮；顶太低任务做不完 |
| recursion_limit=30 独立于业务计数 | 业务计数可能被绕过（如 plan 分支的循环），图层兜底不依赖业务逻辑 |

---

## 五、简单总结

1. 终止权分工：**模型有"提前停"的权利（finished 旗），框架有"必须停"的权力（三道闸）**——`_should_continue` 是裁判（graph.py:342）
2. 四个数字：单工具 **5 次**、去重键 **工具名:参数指纹**、ReAct 上限 **10 轮**、图层 **recursion_limit=30**
3. 防烧 token 的隐藏功臣是**结果缓存回传**：重复调用直接给上次答案（省一轮 LLM），报错式拒绝会把模型逼进"换个说法重试"的灾难循环
