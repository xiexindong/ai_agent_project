# 面试题：Agent 一直在调同一个工具，怎么检测并防止？

> 对应代码：`modules/agent/react_agent.py`
> 本文把"防循环"这件事讲成面试可直接复述的答案，并给出本项目的**五层防御**实现与实测数据。
> 相关：`docs/plan_step-逐步执行机制.md`

---

## 一、一句话答案（30 秒版）

> **多层防御，从便宜到昂贵依次拦：**
> ① 给每次调用做"工具名 + 规范化参数"的指纹，与历史比对，**同名同参且已成功过**就不再执行；
> ② 再查**周期模式**（A→B→A→B），抓"换着花样打转"；
> ③ 再做**进展检测**，最近 N 次结果一模一样就是停摆；
> ④ 叠一层**配额**（单工具最多几次、总次数、token、墙钟）；
> ⑤ 最后是**轮次硬上限**兜底。
> 拦下时**不硬拒**——把上次结果回灌进上下文，让模型"据此直接作答"，否则它会换个说法继续绕。

---

## 二、为什么必须做（本项目实测）

代码 570-574 行的注释记着真实事故：

```
实测：同一句"3*7"，模型连续调了 9 次计算器，
     单次对话被拖到 14 秒、token 约为正常 10 倍。
```

三类成本：**钱（token ×10）、时间（14 秒）、体验（用户看着它原地打转）**。

---

## 三、五层防御总览

| # | 层 | 判定 | 本项目实现 | 状态 |
|---|---|---|---|---|
| 1 | 精确去重 | 同名 + 同参 + 已成功 → 不再执行 | `_last_same_success_call`（649） | 原有 |
| 2 | **周期检测** | 尾部动作由某个周期重复构成 | `_detect_cycle`（682） | **本次新增** |
| 3 | **进展检测** | 最近 N 次成功调用结果完全一致 | `_no_progress`（706） | **本次新增** |
| 4 | 配额预算 | 单工具次数 / 总次数 / token / 墙钟 | `BudgetSpec`（budget.py:34） | 原有 |
| 5 | 轮次上限 | `max_iterations=10` | reason_step（639） | 原有 |

顺序很关键：**越便宜越靠前**。配额和轮次是"迟早会撞上的墙"，前三层是"提前认出你在绕圈"。

---

## 四、三种"打转"形态（面试画图很好用）

| 形态 | 长什么样 | 谁能抓到 |
|---|---|---|
| 原地打转 | `A A A A`（同名同参） | 第 1 层 ✅ |
| 交替环 | `A B A B A B` | 第 2 层 ✅（第 1 层抓不到：每步都和上次不同） |
| 隐形卡死 | `A B C`，但三次结果一字不差 | 第 3 层 ✅（前两层都抓不到：动作全都不一样） |

---

## 五、第 1 层：精确去重（649-664）

```659:664:modules/agent/react_agent.py
        name = call.get("name", "")
        args = call.get("arguments", {})
        for log in reversed(getattr(state, "tool_calls_history", []) or []):
            if log.get("success") and log.get("name") == name and log.get("arguments") == args:
                return log
        return None
```

三个条件**缺一不可**：

| 条件 | 不写会怎样 |
|---|---|
| 工具名相同 | — |
| **参数相同** | 误伤合法调用：`data_query(Q3)` 与 `data_query(Q4)` 是两件事 |
| **且已成功过** | 把"失败重试"也拦了，一次网络抖动就永久卡死 |

---

## 六、第 2 层：交替环检测（本次新增，682-704）

### 算法

把"历史动作 + 本次待执行动作"压成键序列，对每个候选周期 `p`（2 或 3）取尾部 `2p` 个键，检查 `key[i] == key[i-p]`；全等 = 这段重复了两遍 = 出现环。

```python
keys = [cls._action_key(h) for h in (history or [])]
keys.append(cls._action_key(pending_call or {}))     # 把"即将执行的这次"也算进去
for p in range(min_period, max_period + 1):
    if len(keys) < 2 * p:
        continue
    tail = keys[-(2 * p):]
    if all(tail[i] == tail[i - p] for i in range(p, len(tail))):
        return p, [k.split("|", 1)[0] for k in tail[:p]]
```

### 两个设计点

1. **为什么要 `2p` 而不是 `p`**：只看 p 个无法区分"重复"和"恰好相同的一步"。**必须看到同一片段出现两次**才算环，这也是避免误伤的关键。
2. **为什么把 pending 算进去**：否则要等这次执行完、下一轮才能发现，白白多跑一次工具 + 一次 LLM。

### 动作键怎么压（669-679）

```python
args_text = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
return f"{name}|{args_text}"
```

`sort_keys=True` 解决"键顺序不同被当成两个动作"；`default=str` 保证参数里有不可序列化对象时不崩（宁可粗，也不能让主流程挂掉）。

### 实测

```
A,B,A + 待执行 B       → (2, ['data_query', 'report_write'])     ✅ 抓到
A,B,C,A,B + 待执行 C   → (3, ['data_query','report_write','calculator'])  ✅ 抓到
A,B,C + 待执行 D       → None                                     ✅ 不误伤
空历史 + 待执行 A      → None
A,B + 待执行 A（不足 2p）→ None                                   ✅ 数据不够不乱判
键顺序不同算同一动作    → True
```

复杂度：O(周期数 × 2p)，p≤3，可忽略。

---

## 七、第 3 层：进展检测（本次新增，706-719）

```python
logs = [h for h in (history or []) if h.get("success")]
if len(logs) < rounds:
    return False
tail = logs[-rounds:]
results = [str(t.get("result", "") or "").strip() for t in tail]
return bool(results[0]) and len(set(results)) == 1
```

判定：**最近 N（默认 3）次成功调用的结果完全一致** → 再调下去也不会有新信息。

两个防误伤条件：

| 条件 | 为什么 |
|---|---|
| 只看 `success` 的 | 失败记录的结果都是错误串，本来就相同，会把"一直失败"误判成"无进展"；一直失败另有单工具次数上限与重规划处理 |
| 结果非空 | 空结果可能只是"工具没返回内容"，不足以判定停摆 |

### 实测

```
3 次结果相同          → True
3 次结果都不同        → False
3 次全失败（不该算）  → False      ✅
历史不足 3 条         → False      ✅
结果为空（不该算）    → False      ✅
```

---

## 八、拦下之后做什么（比"检测"更关键）

三层命中后处理一致（586-592 / 613-629）：

```python
state.need_tool_call = False     # 不再进 act 节点
state.observation = "…已用相同参数执行过，结果如下（不要重复调用，直接据此作答）：\n{上次结果}"
state.final_answer = ""          # 清空残留答案，让 answer 节点基于 observation 重新生成
return state                     # 提前返回，跳过 act
```

**核心是"刹车 + 给台阶"**：
- 只说"不许调" → 模型会换个工具/换个说法继续绕，问题没解决；
- 把上次结果塞回 `observation` 并明确"据此作答" → 它才有东西可答，循环自然终止。

---

## 九、配置（不配也能用，默认全开）

```python
self._loop_guard_enabled = bool(getattr(s, "agent_loop_guard_enabled", True))   # 总开关
self._cycle_min_period = max(2, int(getattr(s, "agent_cycle_min_period", 2)))   # 最短环长
self._cycle_max_period = max(..., int(getattr(s, "agent_cycle_max_period", 3))) # 最长环长
self._stall_rounds     = max(2, int(getattr(s, "agent_stall_rounds", 3)))       # 连续几次同结果算停摆
```

用 `getattr(settings, 名字, 默认)` 而不是新增必填字段：**加能力不强制改配置类**（与项目既有写法一致，如 `tool_sandbox_image`）。

---

## 十、面试追问 & 标准答法

**Q：为什么不用"同一个工具调用超过 3 次就拦"这种计数法？**
A：太粗。`data_query(Q3)`、`data_query(Q4)`、`data_query(Q2)` 是三次合法调用，计数法会误杀。要用**工具名 + 参数**组成精确键。

**Q：参数里有时间戳 / 随机 id，每次都不一样怎么办？**
A：先做**参数归一化**——进指纹前剔除易变字段（时间、trace_id、随机数），或只取参与业务判定的字段。本项目用 `sort_keys` 解决键顺序问题，业务上真遇到易变参数应在工具层归一化。

**Q：失败重试会不会被误拦？**
A：不会。去重要求"已成功过"（`log.get("success")`），进展检测也只看成功记录。失败重试由**单工具次数上限**和**计划重排**去管，两拨机制不打架。

**Q：有了 max_iterations 还要这些干嘛？**
A：上限是**最后一道墙**，撞上之前已经白跑了 9 轮 LLM（实测 14 秒、token ×10）。前三层是**提前认出绕圈**，把同样的问题压到 1 轮内解决。

**Q：A→B→A→B 这种呢？**
A：单步去重抓不到（每步都和上次不同），要靠**周期检测**：尾部 `2p` 个动作键满足 `key[i] == key[i-p]` 即判定成环，p 取 2、3 覆盖绝大多数情况。

**Q：会不会误伤正常流程？**
A：三重保险——要求片段完整重复**两遍**（`2p`）、只看**成功**记录、结果必须**非空**。正常多步任务（查 Q4 → 查 Q3 → 成文）不会命中。

**Q：分布式/多轮怎么存历史？**
A：本项目 `tool_calls_history` 在 `AgentState` 里，随 state 走 checkpoint（PG / MemorySaver），断点续跑后历史仍在，检测跨轮次有效。

**Q：性能开销？**
A：周期检测 O(2p)、进展检测 O(N)，都是纯内存字符串比较，相对一次 LLM 调用（百毫秒到秒级）可忽略。

---

## 十一、可背的三个数字

| 数字 | 含义 |
|---|---|
| **9 次** | 实测同一句"3*7"被连续调用的次数 |
| **14 秒 → 1 轮** | 加去重后同一问题的耗时变化 |
| **5 层** | 去重 / 周期 / 进展 / 配额 / 轮次上限 |

---

## 十二、一句收尾

**"检测循环不难，难的是拦下之后让模型有台阶下。检测是工程，回灌是产品。"**
