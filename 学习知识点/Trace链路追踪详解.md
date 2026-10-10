# Trace 链路追踪详解

## 1. Trace 是什么？

Trace 是一个 **pydantic 数据模型类**，位于 `modules/orchestration/tracer.py`（L105），用于记录**一次多Agent协作运行的完整观测链路**。

### 通俗理解

想象一个**项目进度看板**：

| 概念 | 生活比喻 |
|:---|:---|
| `Trace` | 一次完整项目的总看板 |
| `Span` | 看板上的每个任务卡片 |
| `trace_id` | 项目编号（唯一标识） |
| `parent_id` | 父子任务关系（大任务拆小任务） |

### 为什么要自建这个 trace 树？

代码顶部注释说得很清楚（L7-L13）：

1. **离线/内网也能追溯** — 不依赖 LangSmith 等外部平台
2. **多Agent的"谁在等谁"** — 自定义结构，外部平台不认识这些节点
3. **编排器要按 span 结果做决策** — 例如某 span 超时就降级

---

## 2. Trace 是怎么记录数据的？

### 存储结构：平铺表 + 树指针

```python
class Trace(BaseModel):
    trace_id: str      # 链路唯一ID，自动生成
    name: str          # 链路名，如 "run.multiagent"
    run_id: str        # 关联的运行ID
    started_ms: int    # 开始时间
    ended_ms: int      # 结束时间（0=未结束）
    root_id: str       # 根span的id
    spans: dict[str, Span]  # 平铺表：{span_id: Span对象}
```

**为什么用平铺 dict 而不是嵌套树？**（L103 注释）
> span 是并发写的，平铺 dict 好加锁、好按 id 直查

### 记录流程（三步）

```
new_trace() -> start_span() 多次 -> end_span() 多次 -> finish()
```

| 步骤 | 方法 | 做什么 |
|:---|:---|:---|
| **建链路** | `new_trace()` | 创建 Trace，登记到 `self.traces` 字典 |
| **开span** | `start_span()` | 创建 Span 加入 `spans` 平铺表，同时挂到父节点的 `children` |
| **关span** | `end_span()` | 结算耗时、定状态（OK/ERROR/TIMEOUT）、补输出 |
| **收口** | `finish()` | 给未关闭的 span 强制标记 TIMEOUT |

### 树形关系怎么还原？

靠 `root_id` + 每个 Span 的 `children` 列表：

```
spans = {
    "span-001": Span(name="run", children=["span-002", "span-005"]),
    "span-002": Span(name="plan", children=["span-003", "span-004"]),
    "span-003": Span(name="agent.researcher"),
    ...
}
root_id = "span-001"
```

`to_tree()` 方法（L130）递归遍历，把平铺表拼成嵌套 JSON 树。

### 实际使用方式

```python
# 方式1：手动开/关
trace = tracer.new_trace("multiagent.run")
span = tracer.start_span(trace, "plan", kind="plan")
# ... 干活 ...
tracer.end_span(span, SpanStatus.OK)

# 方式2：上下文管理器（推荐，自动收尾）
with tracer.span(trace, "plan", kind="plan") as sp:
    sp.set_io(inputs={"query": query})
    # 正常退出自动记 OK，抛异常自动记 ERROR
```

---

## 3. Trace 具体长什么样？

文件顶部注释（L15-L24）给出了一条典型 trace 的示例：

```
run.multiagent (root)
|-- plan              主管拆活
|-- dispatch.batch_1
|   |-- agent.researcher_1   耗时 480ms  status=ok
|   |-- agent.analyst_1      耗时 502ms  status=ok
|-- consensus         相似度 0.21 -> 冲突
|-- arbitration
|-- synthesize
```

线上出问题，按 `trace_id` 把整棵树拉出来，**哪一环慢、哪一环错，一眼可见**。

---

## 4. 关键代码位置速查

| 功能 | 文件 | 行号 | 说明 |
|:---|:---|:---|:---|
| Trace 类定义 | tracer.py | L105 | pydantic 模型，存整条链路 |
| Span 类定义 | tracer.py | L60 | 单个观测单元（一个环节） |
| Tracer 类 | tracer.py | L215 | span 树的写入器 |
| new_trace() | tracer.py | L231 | 开一条新链路 |
| start_span() | tracer.py | L244 | 开一个 span |
| end_span() | tracer.py | L259 | 结束一个 span |
| finish() | tracer.py | L299 | 运行结束收口 |
| to_tree() | tracer.py | L130 | 导出嵌套 JSON 树 |
| to_dict() | tracer.py | L141 | 导出摘要字典 |
| summary() | tracer.py | L181 | 树形文本摘要 |
| 在 harness 中使用 | harness.py | L400-L401 | `new_trace` + `bind_trace` |
| 在 server 中调用 | server.py | L752 | `runner.run()` 触发完整链路 |

---

## 5. 面试/答辩要点

**Q: 为什么要自建 trace 而不是直接用 LangSmith？**

A: 三个刚需：
1. 离线/内网环境要能用（CI 断言）
2. 多Agent的"谁在等谁"是自定义结构，外部平台不认识
3. 编排器要按 span 结果做实时决策（如超时降级）

**Q: 为什么用平铺 dict 存 span 而不是嵌套树？**

A: 并发写友好。多个 Agent 同时开 span，平铺 dict 好加锁、好按 id 直查；嵌套树并发写容易丢 children。

**Q: 如果运行被打断，span 没关怎么办？**

A: `finish()` 会遍历所有 span，把还在 RUNNING 的强制收口为 TIMEOUT，区分"卡住了"和"忘了关"。
