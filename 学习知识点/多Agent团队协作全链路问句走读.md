# 多Agent团队协作全链路问句走读

> 本文按一次代码走读的真实顺序，收录本会话的**全部问句**（Q1-Q23）。
> 从 API 入口一行代码出发，一路追到：run 函数 → trace/span 观测 → RunContext → 委派台账 → 五道闸 → 防环
> → 主管规划调模型 → plan 后流水线 → _emit 事件留痕 → HITL 人工确认前后端全链路。

---

## 走读地图（先有全局）

```
POST /agent/team/run  (server.py L712)
└─ server.py L752：runner.run(req.query, mm=mm, session_id=session_id)
   └─ Harness.run()                     harness.py L371   （native 默认后端）
      ├─ tracer.new_trace()             tracer.py L231 → Trace  L105
      ├─ orchestrator.bind_trace()      orchestrator.py L106
      ├─ tracer.start_span()            tracer.py L244 → Span  L60
      ├─ RunContext(...)                agents.py L94
      ├─ _new_delegation()              harness.py L233
      │  └─ DelegationLedger            delegation.py L89
      │     └─ can() 五道闸             delegation.py L115
      │        └─ chain 防环            delegation.py L134-L140
      ├─ bus.clear()                    coordination.py L151
      │
      ├─ 第1步 主管规划                  harness.py L429 → agents.py L491 plan()
      │  └─ llm.invoke_structured()     agents.py L517 → llm.py L425（一次调用返回 JSON 工单）
      │     PlanResult.model_validate   agents.py L518（本地验票，失败重试→降级）
      ├─ _emit("plan") 事件留痕         harness.py L435 → _emit L218（trace 账本 + on_event 喇叭）
      ├─ 第2步 is_direct 直答收工       harness.py L438
      ├─ 第2.5步 HITL 计划确认          harness.py L449 gate.ask()
      │  ├─ 下行：Notifier → SSE /hitl/stream 推 interrupt 卡片
      │  └─ 上行：POST /hitl/reply → store.submit() → Event.set() 唤醒
      ├─ 第3步 topo_batches 分批        harness.py L475 → L339（按 depends_on 拓扑分批）
      │  └─ _run_batch 并行执行         harness.py L500 → L771 → agent.run() agents.py L303
      ├─ 第3.5步 失败重规划（可选）      harness.py L529
      ├─ 第4步 一致性/仲裁 → synthesize harness.py L541 / L612
      └─ 第5步 review 终审 + 终稿确认    harness.py L633 / L649
```

## 快速跳转目录（点文件名直达代码）

| 文件 | 职责 |
|---|---|
| [modules/api/server.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py) | HTTP 接口层：/agent/team/run 入口 |
| [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py) | 团队编排总入口 Harness.run |
| [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py) | 三个 Agent 类与 BaseAgent 执行逻辑 |
| [modules/multiagent/delegation.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py) | 委派台账与五道闸 |
| [modules/multiagent/models.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py) | SubTask / PlanResult 等数据模型 |
| [modules/multiagent/coordination.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/coordination.py) | 消息总线 MessageBus |
| [modules/multiagent/crewai_backend.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/crewai_backend.py) | 可选 CrewAI 后端 |
| [modules/orchestration/tracer.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py) | Trace / Span 观测层 |
| [modules/orchestration/orchestrator.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/orchestrator.py) | 编排器（消息治理） |
| [modules/hitl/gate.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/gate.py) | 人工审批闸门 |
| [modules/hitl/store.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py) | 审批请求存储与 Event 唤醒 |
| [modules/hitl/notify.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/notify.py) | 审批通知三通道扇出 |
| [modules/api/hitl_routes.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py) | SSE 卡片下行 / 回帖上行接口 |
| [modules/agent/llm.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/agent/llm.py) | LLM 调用（invoke_structured 等） |

> 注意：项目里有**多个 harness.py**——本文提到的 `harness.py` 一律指 [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py)（团队编排）；
> 你 IDE 里可能看到 [modules/harness/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/harness/harness.py)（工具治理）或 [modules/eval/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/eval/harness.py)（评估），都是别的模块，别走错门。

---

# 一、入口：team_run 接口

## Q1：[server.py L752](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py#L752) 这句话什么意思？run 是拆分子任务吗？

```python
return runner.run(req.query, mm=mm, session_id=session_id)
```

这是 **POST /agent/team/run 多Agent协作接口**的核心调用，启动一次多Agent团队协作任务。

| 参数 | 含义 |
|:---|:---|
| `req.query` | 用户的任务指令 |
| `mm` | 多模态输入（图片）；纯文本任务为 None |
| `session_id` | 团队会话ID；没传则自动生成 `team-` + 8位十六进制 |

**是的，run() 会自动拆子任务**，内部流水线：

```
用户任务 → 主管Agent拆任务 → 并行执行(Worker) → 汇总 → 终审 → RunResult
```

与 `/chat` 的区别：`/chat` 是一个人干活（单Agent ReAct循环）；`/agent/team/run` 是一个团队分工干活。

## Q2：run 这个函数在哪里？

`runner` 根据后端二选一，入参和返回值完全一致，可无缝切换：

| 后端 | 文件 | 行号 |
|:---|:---|:---|
| **native（默认）** | [harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py) | [L371](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L371) |
| **crewai**（可选框架，懒加载） | [crewai_backend.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/crewai_backend.py) | [L222](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/crewai_backend.py#L222) |

```python
runner = ctx.harness                    # native
if backend == "crewai":
    runner = ctx.crew_backend           # crewai
```

---

# 二、观测层：trace 和 span

## Q3：[harness.py L400-401](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L400-L401) 这个函数在哪里？

```python
trace = self.tracer.new_trace("multiagent.run", run_id=result.run_id)
self.last_trace = trace
```

这两行不是函数定义，是 `run()` 方法内部的局部代码。
`new_trace()` 定义在 **[tracer.py L231](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L231)**（`Tracer` 类）。

## Q4：trace 是干什么的？

Trace 是一个 **pydantic 数据模型类**（[tracer.py L105](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L105)），记录**一次多Agent协作运行的完整观测链路**。

通俗理解——**项目进度看板**：

| 概念 | 比喻 |
|:---|:---|
| Trace | 一次完整项目的总看板 |
| Span | 看板上的任务卡片 |
| trace_id | 项目编号 |
| parent_id | 父子任务关系 |

为什么自建而不用 LangSmith：① 离线/内网也能追溯；② 多Agent"谁在等谁"是自定义结构；③ 编排器要按 span 结果做决策（超时就降级）。

一条 trace 长这样：

```
run.multiagent (root)
├─ plan              主管拆活
├─ dispatch.batch_1
│  ├─ agent.researcher_1   480ms  ok
│  └─ agent.analyst_1      502ms  ok
├─ consensus         相似度 0.21 → 冲突
├─ arbitration
└─ synthesize
```

## Q5：trace 这个"函数"在哪里？怎么记录的？

- Trace **类**：[tracer.py L105](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L105)（不是函数，是数据容器）
- Span **类**：[tracer.py L60](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L60)
- Tracer **写入器类**：[tracer.py L215](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L215)

存储结构——**平铺表 + 树指针**：

```python
class Trace(BaseModel):
    trace_id: str               # 自动生成
    root_id: str                # 根span id
    spans: dict[str, Span]      # 平铺表 {span_id: Span}
```

为什么平铺不嵌套：span 是并发写的，平铺 dict 好加锁、好按 id 直查。

记录流程：

```
new_trace() → start_span() 多次 → end_span() 多次 → finish()
```

树形关系靠 `root_id` + 每个 Span 的 `children` 列表还原；`to_tree()`（[L130](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L130)）递归拼成嵌套 JSON。

## Q6：[harness.py L404-405](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L404-L405) 函数在哪里？作用是什么？

```python
self.orchestrator.bind_trace(trace)                                    # L404
root = self.tracer.start_span(trace, "run", kind="run", query=...)    # L405
```

| 调用 | 定义位置 |
|:---|:---|
| `bind_trace()` | [orchestrator.py L106](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/orchestrator.py#L106) |
| `start_span()` | [tracer.py L244](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L244) |

- **L404**：把 trace 挂给编排器，让消息 span 知道往哪写。
- **L405**：开**根 span**，是整棵树的起点，后续所有 span 都挂它下面。

这一步是 trace 树的"落地生根"。

## Q7：[tracer.py L248-249](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L248-L249) 这个 span 在干什么？

```python
span = Span(trace_id=trace.trace_id, name=name, kind=kind,
            parent_id=parent_id, attributes=attributes)
```

这是 `start_span()` 的第一步：**创建一个 Span 对象**。

关键点：**Span 一创建，计时就开始了**——`start_ms` 默认 `now_ms()`，状态出生即 RUNNING。此时还没登记进树（入册在 [L251](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L251)）。

## Q8：这个 span 在哪里？

同一个文件——**[tracer.py L60-L96](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L60-L96)**，`Span(BaseModel)`：

```
span_id(自动) / parent_id / trace_id / name / kind
start_ms(自动取现在) / end_ms(0) / duration_ms / status(RUNNING)
inputs / outputs / error / attributes / children
done(属性) / set_io(链式补记)
```

[tracer.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py) 一共三个核心类：`SpanStatus`（[L48](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L48)）、`Span`（[L60](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L60)）、`Trace`（[L105](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L105)） + 写入器 `Tracer`（[L215](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L215)）。

---

# 三、共享上下文：RunContext

## Q9：[harness.py L411-412](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L411-L412) RunContext 在哪里？有什么作用？

```python
ctx = RunContext(query=query, mm=mm, session_id=session_id,
                 bus=self.obus, gate=self.gate)
```

定义：**[agents.py L94-L120](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L94-L120)**（`@dataclass`，只读共享上下文）。

作用：一次运行中所有 Agent 共享的**"资料袋 + 公共设施"**。

| 类别 | 字段 | 作用 |
|:---|:---|:---|
| 任务资料 | query / mm / upstream / session_id / principal | 原始问题、图片、**上游产出**、会话隔离、RBAC身份 |
| 公共设施 | bus / gate | 消息总线（横向对话）、人工闸门（敏感动作拍板） |
| 委派三件套 | delegation / depth / chain / delegated_by | 委派执行器、层数、防环链、受谁委派 |

通俗比喻：剧组发给每个演员的资料包——剧本 + 对讲机 + 喊停按钮 + 转包合同。

---

# 四、委派初始化

## Q10：[harness.py L414-415](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L414-L415) 函数在哪里？做什么？

```python
self._new_delegation(ctx, result)  # 建本轮委派台账与执行器
self.bus.clear()                   # 上一轮消息不跨运行
```

| 调用 | 定义位置 |
|:---|:---|
| `_new_delegation()` | [harness.py L233](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L233) |
| `bus.clear()` | [coordination.py L151](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/coordination.py#L151)（`MessageBus.clear`） |

- **L414**：新建本轮委派设施（台账 + 事件回调 + 执行回调），挂到 `ctx.delegation`。委派次数是"这一轮"的预算，不能跨轮累计。
- **L415**：清空总线全部历史消息——上一轮的异议留到下一轮是纯噪声。

核心思想：**本轮的状态本轮建、本轮清**。

## Q11：DelegationLedger 是干什么的？为什么 Ctrl+点击跳不过去？

DelegationLedger **是类不是函数**，定义在 **[delegation.py L89](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L89)**，在 [harness.py L42-L46](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L42-L46) 正常导入。

它是一次运行的委派**"记账本"**：① 记账（成功/拒绝）；② 计数（总数/单人）；③ 防环（委派链）。

**Ctrl+点击跳不动**的排查（导入写法本身没问题）：
1. 等 Pylance 完成索引
2. `Python: Select Interpreter` 选解释器
3. 点"信任此作者"
4. **确认 VS Code 打开的是 `ai_agent_project` 这一层**（绝对导入 `modules.xxx` 依赖根目录正确）
5. `Developer: Reload Window`

## Q12：ledger 到底是什么东西？（通俗版）

**工地保安亭里的那本【转包登记本】**：

```
一次 run = 开工一个工地
工人要把活转给别的工人 → 必须到保安亭登记
保安翻开本子对照规矩，放行或拒绝
不管放行还是拒绝，都在本子上记一笔
```

`self._delegation_ledger = ledger`（[L243](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L243)）= 把登记本挂在工地墙上，收工（finally）时整本归档进 RunResult。

每次 run 换新本子：旧本子是上个工地的，不换的话新工地的转包配额会被上个工地用光。

---

# 五、五道闸与防环

## Q13：[delegation.py L263-264](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L263-L264) 过哪五道？

**任一闸不过 = 不委派，主流程照常往下走。**

| 闸 | 检查 | 被拒原因 |
|:---:|:---|:---|
| ① | **开关** | "委派未启用" |
| ② | **资格**：can_delegate；白名单 | "未被授权" / "不在白名单" |
| ③ | **存在性**：在花名册、启用中 | "不在花名册" / "已停用" |
| ④ | **边界**：深度/总次数/单人次数 | 三条配额（默认 1 / 6 / 2） |
| ⑤ | **防环**：不能给自己、不能给链上已有的人 | "不能委派给自己" / "委派成环" |

代码实际顺序（便宜的纯内存判断在前）：
`开关 → 自环 → 成环 → 深度 → 总次数 → 单人次数 → 资格 → 存在性`

## Q14：[L128-129](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L128-L129) 怎么防死循环（虚循环）？

死循环：`analyst_1 → researcher_1 → analyst_1 → …` 互相甩锅停不下来。

机制：**chain 经手人名单，只增不减**。每次转派把自己追加进去（[L275](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L275)：`chain + [from_agent]`）。

```
① 主管派 analyst_1              chain = []
② analyst_1 转 researcher_1     researcher_1 在[]里？不在 → 放行
                                 chain = ["analyst_1"]
③ researcher_1 转回 analyst_1   "analyst_1" 在链上 → 拒绝"委派成环"
                                 返回 None，只能自己干
```

两道检查（[L134-L140](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L134-L140)）：
```python
if delegator == delegatee:  deny("不能委派给自己")   # A→A
if delegatee in chain:      deny("委派成环")         # A→B→A 任意形状
```

一定防得住：想第二次接同一个活，名字必在链上，`in` 必失败。再加深度闸（默认1层）双保险。

**一句话：查一下"经手人名单"，受托方在名单里就拒绝。**

## Q15：被拒绝为什么也要留痕？为什么返回 None 不抛异常？

留痕（[L128-L129](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L128-L129)）：分清"根本没想转"和"想转但被拦"——排查"为什么没人接"时有据可查，所以 can() 返回中文原因而不只是 bool。

返回 None：委派是"锦上添花"，转不出去原任务照样要做；抛异常会为一个可选能力赔上整个子任务。

---

# 六、消息信封与委派执行细节

## Q16：[orchestrator.py L159-168](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/orchestrator.py#L159-L168) 这一段的总目的？

```python
env = Envelope(
    sender=..., receiver=..., kind=..., content=...,
    round=self.round_no, hop=(parent.hop + 1) if parent else 0,
    parent_id=..., trace_id=..., run_id=run_id,
)
```

**总目的：把 `send()` 入口收到的一堆散参数，打包成一个标准化的"信封"对象。**

为什么打包：后面六道关卡（校验→限流→配额→防环→投递→上报）都要用这些信息。不打包的话每关都要收一堆独立参数，改字段所有关卡都要改签名。打包后**所有关卡只跟信封说话**。

信封字段按作用分：

| 分类 | 字段 | 作用 |
|:---|:---|:---|
| 基本信息 | sender / receiver / kind / content | 谁发、给谁、什么类型、正文 |
| 治理用 | sender_role / ref_task_id / meta | 权限判定、任务关联、附加数据 |
| 轮次/跳数 | round / hop | 每轮配额、防死循环跳数闸 |
| 链路追踪 | parent_id / trace_id / run_id | 拼对话链、挂 span、对上 RunResult |

两个关键计算：
- **hop**：回复别人的消息算一跳，主动发起是第 0 跳
- **parent_id**：挂父消息 id，才能拼出"谁回复谁"的对话链

## Q17：[delegation.py L272-276](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L272-L276) DelegationRecord 各字段作用？

五道闸放行后，正式创建一条委派流水账：

| 字段 | 含义 |
|:---|:---|
| `from_agent` | 委派方：谁把活转出去 |
| `to_agent` | 受托方：活转给谁 |
| `instruction` | 转出去那块活的具体指令 |
| `reason` | 为什么转（给受托方/主管看） |
| `parent_task_id` | 从哪个任务转出来（trace 可追溯） |
| `depth` | 当前委派深度（0=主管直派，再转+1） |
| `chain` | **经手人名单**：把自己追加进去，传给受托方防环 |

这条 `rec` 接着 `ledger.open()` 记账，执行完 `ledger.close()` 回填结果，收工时整本归档。

## Q18：[delegation.py L298](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L298) `self._run_fn` 调用的谁？

调用的是**受托方 Agent 的 `run()` 方法**。

`_run_fn` 是从 harness 注入的回调（[harness.py L266-L299](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L266-L299)），核心是：

```python
agent = self.agents.get(agent_name)   # 按 to_agent 取出受托方实例
out = agent.run(task, sub_ctx)        # 真正跑受托方任务
return out
```

**为什么用回调**：harness 依赖 delegation（建台账），delegation 不能反向依赖 harness（循环导入）。所以 delegation 只留"插口"，具体怎么取 agent、造上下文、包 span 都由 harness 注入。

## Q19：[harness.py L297](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L297) `agent.run` 是哪个函数？

是 **`Agent.run()`**，定义在 **[agents.py L303](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L303)**：

```python
def run(self, task: Optional[SubTask], ctx: RunContext) -> AgentOutput:
```

注意它和最早问的 `runner.run` 不是同一个：

| 方法 | 位置 | 干什么 |
|:---|:---|:---|
| `Harness.run()` | [harness.py L371](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L371) | 一次**完整团队协作**总入口（拆活、派活、汇总） |
| `agent.run()` | [agents.py L303](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L303) | **单个工人**执行自己那一块任务 |

调用关系：
```
Harness.run()           ← server.py 调的（团队总入口）
   └─ 主管拆任务 → 派给多个 worker
        └─ agent.run(task, ctx)   ← harness.py L297 调的（单个工人干活）
```

---

# 七、主管规划：大模型怎么拆任务

## Q20：[agents.py L517-518](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L517-L518) 这一步是不是调用大模型进行分步骤？

```python
data = self.llm.invoke_structured(prompt, system_prompt=self.system_prompt)  # L517
return PlanResult.model_validate(data)                                       # L518
```

**是的，调大模型拆步骤的核心动作就在 L517，但两行职责不同：**

| 行 | 干什么 | 是否调模型 |
|:---|:---|:---:|
| L517 | 把填好的规划提示词发给大模型，**要求返回一个 JSON**（拆几个子任务、每个派给谁、依赖谁） | ✅ 真正发请求，全程就这一次 |
| L518 | 拿到的只是普通字典，本地用 Pydantic 按 `PlanResult` 结构**当场验票**：字段齐不齐、assignee 有没有填 | ❌ 纯本地校验 |

`invoke_structured` 内部（**[llm.py L425-448](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/agent/llm.py#L425-L448)**）：拼上 system_prompt、追加"请严格以 JSON 格式输出"，并用 `response_format={"type": "json_object"}` 强制服务端返回合法 JSON；温度特意压到 **0.2** 保证输出稳定可解析。

模型返回长这样（相当于在固定格子里填空）：

```json
{
  "thought": "先查后算，两步可并行",
  "subtasks": [
    {"id": "t1", "assignee": "researcher_1", "instruction": "查 Q3/Q4 营收"},
    {"id": "t2", "assignee": "analyst_1", "instruction": "算环比", "depends_on": ["t1"]}
  ],
  "direct_answer": "",
  "needs_review": false
}
```

**关键区分："分步骤"的决策在一次调用里整体返回，模型不是边执行边想下一步**；真正的逐步执行是后面调度器照这张工单派工。

通俗比喻——**主管填派工单**：L510-515 备料（问题 + 工人名单 roster + 派工规则 + 工单格式 schema）→ L517 打电话问大脑 → L518 验工单。

两道保险（[agents.py L519-539](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L519-L539)）：校验失败把错误原文**回喂重试一次**（最常见错是漏字段）；第二次还失败就**降级为单 Agent 直答**，团队不停摆。

## Q21：plan() 执行完毕，下一步在哪里？

`plan()` 自己不决定下一步——它是被 **`Harness.run()`** 调用的，调用点在 **[harness.py L429](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L429)**。返回 `PlanResult` 后控制权回到 `run()`，沿流水线往下：

```
plan() 返回 PlanResult
  │
  ├─ ① 记账留痕      L433-435  plan 写进 result + emit("plan") 推送
  ├─ ② 直接答收工    L437-443  is_direct 为真（寒暄/单步）→ 用 direct_answer 返回
  ├─ ②.5 人工确认拆法 L445-472  HITL 可选：人可批准/驳回/终止
  ├─ ③ DAG 分批      L474-478  topo_batches()（L339）按 depends_on 切批次
  ├─ ③ 逐批并行执行  L481-525  _run_batch()（L771）线程池并行调 agent.run()
  │                            产出写两处：result.outputs / ctx.upstream
  ├─ ③.5 失败重规划  L527-530  可选，默认关（省 token）
  ├─ ④ 一致性/仲裁   L541-602  产出打架时比对相似度、仲裁员/人工裁决
  ├─ ④ 主管汇总      L604-615  synthesize() 汇成草稿
  ├─ ⑤ 终审          L632-642  可选：ReviewerAgent.review()
  └─ ⑤.5 终稿人工确认 L649+     最后一道人工放行 → 组装 RunResult
```

真正开始干活的两个入口：

- **topo_batches()**（[harness.py L339](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L339)）：带子依赖的任务按拓扑序切成"批次"，同批可并行，有环的依赖被剔除；
- **_run_batch()**（[harness.py L771](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L771)）：一批之内并行调每个成员的 `agent.run()`（[agents.py L303](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L303)）。

比喻：plan 是主管画完**施工横道图** → 图纸贴公告栏 → 小事主管自己答 → 真动工先排分期工程（分批）→ 每期多工种同时进场（并行）→ 前期成果放共享区（upstream）→ 主管验收汇总、质检员终审。

旁路：挂了 CrewAI 后端时调用点在 [crewai_backend.py L244](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/crewai_backend.py#L244)，规划仍用自研 plan()，执行交给 CrewAI 框架。

---

# 八、事件留痕：_emit

## Q22：[harness.py L435-436](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L435-L436) emit 是干什么？

`_emit` 是 Harness 内部的**事件播报器**——流水线每到关键节点就喊一嗓子"现在发生了什么"，外面看得见进度、事后能复盘。定义只有 9 行（**[harness.py L216-226](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L216-L226)**），做两件事：

```python
event = TraceEvent(kind=kind, message=message, agent=agent, payload=payload)
result.trace.append(event)   # ① 内部账本：追加到本次运行事件流水，随 RunResult 返回
if self.on_event:            # ② 外部喇叭：回调构造 Harness 时传入的钩子（SSE/审计）
    self.on_event(event)     #    钩子异常 try/except 吞掉只记 debug，绝不炸主流程
```

| 组成 | 含义 | L435 那行的实际值 |
|:---|:---|:---|
| `kind` | 事件类型，取值被 TraceKind 写死（[models.py L240-249](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py#L240-L249)），写错字符串类型检查阶段就报错 | `"plan"` |
| `message` | 给人看的一句话 | 主管思考摘要，没有就写"拆出 N 个子任务" |
| `agent` | 这事谁干的 | supervisor.name |
| `payload` | 机器读的结构化附载 | 完整子任务清单 |
| `ts` | 毫秒时间戳，建事件时自动盖 | 自动生成 |

`TraceEvent` 定义在 [models.py L254-261](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py#L254-L261)；kind 全集：`plan / dispatch / agent_start / agent_end / synthesize / review / done / error / consensus / arbitration / hitl / replan / delegate`。

通俗比喻——工地门口：**`result.trace` 是施工日志本**（按时间登记、收工存档），**`on_event` 是广播喇叭**（同步喊给前端进度条/审计系统），`_emit` 就是"一边登记一边按喇叭"。

易混点：委派模块也有个 `on_event`（[delegation.py L228](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L228)），但它的归宿仍是 `_emit`——委派放行/被拒后在 [harness.py L247-255](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L247-L255) 的回调内部调 `self._emit(result, "delegate", ...)`。**全团队所有环节的事件最终走 _emit 一个出口**，trace 格式统一。

---

# 九、人工确认（HITL）：后端等人与前端互通

## Q23：[harness.py L450-458](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L450-L458) gate.ask 阻塞等人工，怎么通知到前端？前端又怎么把决策传回来？

靠三样东西配合：**一个阻塞的 `threading.Event`（后端等人）、一条 SSE 长连接（服务器→前端推卡片）、一个 POST 接口（前端→服务器回决策）**。

### 完整时序

```
Agent 工作线程                     后端 FastAPI                    前端审批页
─────────────                     ────────────                    ──────────
gate.ask()（gate.py L135）
  ├─ store.save(req) 登记请求 + 预热 threading.Event
  ├─ on_event(req) ──► InterruptNotifier.notify()（notify.py L174）
  │                      三通道扇出：① stderr 终端
  │                      ② 塞进每个 SSE 订阅者的队列
  │                      ③ 钉钉/飞书 webhook（可选）
  │                                       GET /hitl/stream ◄── 提前挂着的 SSE 长连接
  │                                       队列拿到 req ─────► event: interrupt 帧
  │                                                           渲染卡片（标题/子任务/倒计时）
  │                                       POST /hitl/reply ◄── 人点"批准"
  │                                       {request_id, action}
  │                      gate.submit() → store.submit()（store.py L118）
  │                        存回复、WAITING→RESOLVED
  │                        ★ event.set() 唤醒
  ◄─ event.wait() 立刻醒来拿到 HumanReply
  ▼
harness L459 按 reply.action 决定继续派活 / 驳回 / 终止
```

### 下行（服务器 → 页面）：请求怎么变成卡片

1. **登记并挂起**：gate.ask()（[gate.py L155-178](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/gate.py#L155-L178)）先 `store.save(req)` 存字典并配一个 Event，再调 `on_event(req)` 钩子；
2. **通知器扇出**：钩子是 InterruptNotifier.notify()，SSE 这一路把请求塞进每个订阅者的专属队列（[notify.py L300](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/notify.py#L300) subscribe）；
3. **SSE 端点推帧**：前端挂着的 `GET /hitl/stream`（[hitl_routes.py L103-132](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py#L103-L132)）消费队列，推标准帧：

```
event: interrupt
data: {"id":"hitl_xxx","kind":"before_plan","title":"确认任务分解（2 个子任务）",
       "context":{...},"options":[...],"wait_seconds":60}
```

补救渠道：连上 SSE 的瞬间**先补发积压待办**（[L112/116](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py#L112)）；也可以直接轮询 `GET /hitl/pending`（[L89-95](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py#L89-L95)）。

### 上行（页面 → 服务器）：点按钮怎么唤醒线程

人点按钮 → 普通 POST（[hitl_routes.py L140-163](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py#L140-L163)）：

```
POST /hitl/reply   {"request_id":"hitl_xxx","action":"approve","modified":"","note":""}
```

服务端 `gate.submit()` → **store.submit()**（[store.py L118-139](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py#L118-L139)）做三件事：
① 校验请求仍是 WAITING（重复点击第二次返回 **409**，保住第一次决策）；
② 存回复、状态改 RESOLVED、回填归档；
③ **`event.set()`——这一行就是"唤醒"**。

Agent 线程阻塞在 store.wait() 的 `event.wait(timeout)`（[store.py L150-166](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py#L150-L166)），set() 是内核级信号，**立刻**醒来取回复返回，ask() 拿到 HumanReply，[harness.py L459](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L459) 按动作分支继续。

### 为什么线程阻塞不卡服务

团队执行跑在**线程池**的工作线程里（_run_batch 并行派活），阻塞的是工作线程；POST /hitl/reply 由 asyncio 主线程接收，submit() 只是写字典 + 拨 Event，毫秒级完成。两个世界靠 threading.Event 碰面。不用轮询等待的原因（[store.py L13-17](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py#L13-L17)）：白占线程 + 最坏一个轮询周期才生效，体感"点了没反应"。

### 通俗比喻：手术室等家属签字

医生开签字单（InterruptRequest，带单号）贴护士站（store）→ 坐休息室等叫号（Event.wait）→ 大喇叭三渠道喊人（终端/SSE/ webhook）→ 家属 App 看单子点"同意"（POST 回单号）→ 护士盖章归档按呼叫铃（event.set）→ 医生立刻听见回去手术，不用自己跑去问。

### 三道兜底

| 情况 | 行为 | 位置 |
|:---|:---|:---|
| 人一直不点 | 等到超时（配置默认 60s）按 default_action 兜底（工具默认拒绝、终稿默认放行） | [gate.py L182-192](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/gate.py#L182-L192) |
| 网络重试点两次 | 第二次返回 409 Conflict，第一次决策不可覆盖 | [store.py L124-126](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py#L124-L126) |
| 流程提前结束 | cancel_waiting() 置 CANCELLED 并唤醒，避免线程挂到超时占并行度 | [gate.py L283-295](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/gate.py#L283-L295) |

### 两个部署注意

1. **存储是进程内的**（[store.py L19-22](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py#L19-L22)）：多副本部署时抛中断与收回复可能不在同一台机器，会"点了批准却永远等不到"，须换共享存储（预留 PGInterruptStore）；
2. **浏览器原生 EventSource 发不了自定义请求头**，本项目鉴权走 X-API-Key（[hitl_routes.py L100-102](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py#L100-L102)），网页端要用 fetch + ReadableStream 接 SSE 或网关注入 key，否则 401/403。

---

# 附录 A：关键代码位置总表

| 功能 | 文件 | 行号 |
|:---|:---|:---|
| team_run 接口 | [modules/api/server.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py) | [L712](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py#L712) / 调用在 [L752](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py#L752) |
| run（native） | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py) | [L371](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L371) |
| run（crewai） | [modules/multiagent/crewai_backend.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/crewai_backend.py) | [L222](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/crewai_backend.py#L222) |
| Trace 类 | [modules/orchestration/tracer.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py) | [L105](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L105) |
| Span 类 | [modules/orchestration/tracer.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py) | [L60](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L60) |
| Tracer / new_trace | [modules/orchestration/tracer.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py) | [L215](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L215) / [L231](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L231) |
| start_span / end_span | [modules/orchestration/tracer.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py) | [L244](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L244) / [L259](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/tracer.py#L259) |
| bind_trace | [modules/orchestration/orchestrator.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/orchestrator.py) | [L106](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/orchestration/orchestrator.py#L106) |
| RunContext | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py) | [L94](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L94) |
| _new_delegation | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py) | [L233](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L233) |
| MessageBus.clear | [modules/multiagent/coordination.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/coordination.py) | [L151](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/coordination.py#L151) |
| DelegationPolicy | [modules/multiagent/delegation.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py) | [L67](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L67) |
| DelegationLedger / can() | [modules/multiagent/delegation.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py) | [L89](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L89) / [L115](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L115) |
| 防环检查 | [modules/multiagent/delegation.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py) | [L134-L140](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L134-L140) |
| DelegationExecutor / delegate() | [modules/multiagent/delegation.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py) | [L210](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L210) / [L253](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L253) |
| SupervisorAgent.plan | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py) | [L491](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L491)（调模型 [L517](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L517) / 校验 [L518](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L518) / 重试降级 [L519-539](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L519-L539)） |
| invoke_structured | [modules/agent/llm.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/agent/llm.py) | [L425](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/agent/llm.py#L425)（json_object 强制 JSON，温度 0.2） |
| plan 调用点 / 规划 span | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py) | [L429](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L429) / [L426](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L426) |
| _emit 事件播报器 | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py) | [L218](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L218)（trace 记账 + on_event 回调） |
| TraceEvent / TraceKind | [modules/multiagent/models.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py) | [L254](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py#L254) / [L240](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py#L240) |
| topo_batches 拓扑分批 | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py) | [L339](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L339)（调用在 [L475](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L475)） |
| _run_batch 并行派活 | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py) | [L771](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L771)（调用在 [L500](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L500)） |
| synthesize 汇总 / review 终审 | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py) | [L612](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L612) / [L640](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L640) |
| HumanGate.ask | [modules/hitl/gate.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/gate.py) | [L135](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/gate.py#L135)（ask_tool [L203](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/gate.py#L203) / ask_final [L241](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/gate.py#L241) / cancel_waiting [L283](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/gate.py#L283)） |
| InMemoryInterruptStore | [modules/hitl/store.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py) | save [L78](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py#L78) / submit [L118](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py#L118) / wait [L150](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py#L150)（Event 唤醒在 [L134](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py#L134)） |
| InterruptNotifier | [modules/hitl/notify.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/notify.py) | notify [L174](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/notify.py#L174) / subscribe [L300](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/notify.py#L300) |
| HITL 三个 HTTP 接口 | [modules/api/hitl_routes.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py) | GET /hitl/stream [L103](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py#L103)、POST /hitl/reply [L140](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py#L140)、GET /hitl/pending [L89](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py#L89) |

---

# 附录 B：面试/答辩要点

1. **runner.run 是什么？** 团队协作主入口，主管拆任务→并行→汇总→终审；native/crewai 两后端接口一致可切换。
2. **Trace/Span 关系？** Trace 是总看板，Span 是任务卡片；平铺表存储、加锁并发写、靠 root_id+children 还原树。
3. **RunContext 作用？** 全队共享的资料袋（问题/上游产出）+ 公共设施（bus/gate）+ 委派三件套，避免参数层层透传。
4. **怎么防委派死循环？** chain 经手人名单只增不减，受托方在名单上即拒绝；深度上限做第二道保险。
5. **委派被拒为什么留痕、为什么返回 None？** "没想转"和"想转被拦"要可区分；委派是可选能力，不能拖垮主流程。
6. **规划为什么用结构化输出？** 自由文本靠正则抠 assignee/depends_on 极脆弱；喂 JSON Schema 让模型在格子里填空，model_validate 当场校验，失败带错误原文重试一次，再失败降级单 Agent 直答。
7. **"分步骤"发生在哪？** 分步决策在 plan 的一次 LLM 调用里整体返回；模型不边执行边想，逐步执行由 topo_batches 按依赖分批、_run_batch 并行派活完成。
8. **_emit 的两个去向？** 内部 result.trace 事件流水（随 RunResult 返回可复盘）+ 外部 on_event 钩子（SSE 推送/审计落库）；钩子异常隔离不炸主流程，全团队事件统一走这一个出口。
9. **HITL 阻塞等待怎么和前端互通？** 后端工作线程靠 threading.Event 内核级等待；下行 Notifier 扇出经 SSE /hitl/stream 推 interrupt 卡片，上行 POST /hitl/reply 经 store.submit() 置 RESOLVED 并 event.set() 立刻唤醒；超时按默认动作兜底，重复提交返回 409。
10. **阻塞为什么不拖垮服务？** 阻塞的是线程池工作线程，HTTP 回帖由 asyncio 主线程毫秒级处理；进程内 Event 只适用单实例，多副本须换共享存储。
