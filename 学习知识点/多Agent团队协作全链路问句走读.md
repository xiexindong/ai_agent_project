# 多Agent团队协作全链路问句走读

> 本文按一次代码走读的真实顺序，收录本会话的**全部问句**。
> 从 API 入口一行代码出发，一路追到：run 函数 → trace/span 观测 → RunContext → 委派台账 → 五道闸 → 防环。

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
      └─ bus.clear()                    coordination.py L151
```

---

# 一、入口：team_run 接口

## Q1：`server.py L752` 这句话什么意思？run 是拆分子任务吗？

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
| **native（默认）** | harness.py | L371 |
| **crewai**（可选框架，懒加载） | crewai_backend.py | L222 |

```python
runner = ctx.harness                    # native
if backend == "crewai":
    runner = ctx.crew_backend           # crewai
```

---

# 二、观测层：trace 和 span

## Q3：`harness.py L400-401` 这个函数在哪里？

```python
trace = self.tracer.new_trace("multiagent.run", run_id=result.run_id)
self.last_trace = trace
```

这两行不是函数定义，是 `run()` 方法内部的局部代码。
`new_trace()` 定义在 **tracer.py L231**（`Tracer` 类）。

## Q4：trace 是干什么的？

Trace 是一个 **pydantic 数据模型类**（tracer.py L105），记录**一次多Agent协作运行的完整观测链路**。

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

- Trace **类**：tracer.py L105（不是函数，是数据容器）
- Span **类**：tracer.py L60
- Tracer **写入器类**：tracer.py L215

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

树形关系靠 `root_id` + 每个 Span 的 `children` 列表还原；`to_tree()`（L130）递归拼成嵌套 JSON。

## Q6：`harness.py L404-405` 函数在哪里？作用是什么？

```python
self.orchestrator.bind_trace(trace)                                    # L404
root = self.tracer.start_span(trace, "run", kind="run", query=...)    # L405
```

| 调用 | 定义位置 |
|:---|:---|
| `bind_trace()` | orchestrator.py L106 |
| `start_span()` | tracer.py L244 |

- **L404**：把 trace 挂给编排器，让消息 span 知道往哪写。
- **L405**：开**根 span**，是整棵树的起点，后续所有 span 都挂它下面。

这一步是 trace 树的"落地生根"。

## Q7：`tracer.py L248-249` 这个 span 在干什么？

```python
span = Span(trace_id=trace.trace_id, name=name, kind=kind,
            parent_id=parent_id, attributes=attributes)
```

这是 `start_span()` 的第一步：**创建一个 Span 对象**。

关键点：**Span 一创建，计时就开始了**——`start_ms` 默认 `now_ms()`，状态出生即 RUNNING。此时还没登记进树（入册在 L251）。

## Q8：这个 span 在哪里？

同一个文件——**tracer.py L60-L96**，`Span(BaseModel)`：

```
span_id(自动) / parent_id / trace_id / name / kind
start_ms(自动取现在) / end_ms(0) / duration_ms / status(RUNNING)
inputs / outputs / error / attributes / children
done(属性) / set_io(链式补记)
```

tracer.py 一共三个核心类：`SpanStatus`(L48)、`Span`(L60)、`Trace`(L105) + 写入器 `Tracer`(L215)。

---

# 三、共享上下文：RunContext

## Q9：`harness.py L411-412` RunContext 在哪里？有什么作用？

```python
ctx = RunContext(query=query, mm=mm, session_id=session_id,
                 bus=self.obus, gate=self.gate)
```

定义：**agents.py L94-L120**（`@dataclass`，只读共享上下文）。

作用：一次运行中所有 Agent 共享的**"资料袋 + 公共设施"**。

| 类别 | 字段 | 作用 |
|:---|:---|:---|
| 任务资料 | query / mm / upstream / session_id / principal | 原始问题、图片、**上游产出**、会话隔离、RBAC身份 |
| 公共设施 | bus / gate | 消息总线（横向对话）、人工闸门（敏感动作拍板） |
| 委派三件套 | delegation / depth / chain / delegated_by | 委派执行器、层数、防环链、受谁委派 |

通俗比喻：剧组发给每个演员的资料包——剧本 + 对讲机 + 喊停按钮 + 转包合同。

---

# 四、委派初始化

## Q10：`harness.py L414-415` 函数在哪里？做什么？

```python
self._new_delegation(ctx, result)  # 建本轮委派台账与执行器
self.bus.clear()                   # 上一轮消息不跨运行
```

| 调用 | 定义位置 |
|:---|:---|
| `_new_delegation()` | harness.py L233 |
| `bus.clear()` | coordination.py L151（`MessageBus.clear`） |

- **L414**：新建本轮委派设施（台账 + 事件回调 + 执行回调），挂到 `ctx.delegation`。委派次数是"这一轮"的预算，不能跨轮累计。
- **L415**：清空总线全部历史消息——上一轮的异议留到下一轮是纯噪声。

核心思想：**本轮的状态本轮建、本轮清**。

## Q11：DelegationLedger 是干什么的？为什么 Ctrl+点击跳不过去？

DelegationLedger **是类不是函数**，定义在 **delegation.py L89**，在 harness.py L42-L46 正常导入。

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

`self._delegation_ledger = ledger`（L243）= 把登记本挂在工地墙上，收工（finally）时整本归档进 RunResult。

每次 run 换新本子：旧本子是上个工地的，不换的话新工地的转包配额会被上个工地用光。

---

# 五、五道闸与防环

## Q13：`delegation.py L263-264` 过哪五道？

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

## Q14：`L128-129` 怎么防死循环（虚循环）？

死循环：`analyst_1 → researcher_1 → analyst_1 → …` 互相甩锅停不下来。

机制：**chain 经手人名单，只增不减**。每次转派把自己追加进去（L275：`chain + [from_agent]`）。

```
① 主管派 analyst_1              chain = []
② analyst_1 转 researcher_1     researcher_1 在[]里？不在 → 放行
                                 chain = ["analyst_1"]
③ researcher_1 转回 analyst_1   "analyst_1" 在链上 → 拒绝"委派成环"
                                 返回 None，只能自己干
```

两道检查（L134-L140）：
```python
if delegator == delegatee:  deny("不能委派给自己")   # A→A
if delegatee in chain:      deny("委派成环")         # A→B→A 任意形状
```

一定防得住：想第二次接同一个活，名字必在链上，`in` 必失败。再加深度闸（默认1层）双保险。

**一句话：查一下"经手人名单"，受托方在名单里就拒绝。**

## Q15：被拒绝为什么也要留痕？为什么返回 None 不抛异常？

留痕（L128-L129）：分清"根本没想转"和"想转但被拦"——排查"为什么没人接"时有据可查，所以 can() 返回中文原因而不只是 bool。

返回 None：委派是"锦上添花"，转不出去原任务照样要做；抛异常会为一个可选能力赔上整个子任务。

---

# 六、消息信封与委派执行细节

## Q16：`orchestrator.py L159-168` 这一段的总目的？

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

## Q17：`delegation.py L272-276` DelegationRecord 各字段作用？

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

## Q18：`delegation.py L298` `self._run_fn` 调用的谁？

调用的是**受托方 Agent 的 `run()` 方法**。

`_run_fn` 是从 harness 注入的回调（harness.py L266-L299），核心是：

```python
agent = self.agents.get(agent_name)   # 按 to_agent 取出受托方实例
out = agent.run(task, sub_ctx)        # 真正跑受托方任务
return out
```

**为什么用回调**：harness 依赖 delegation（建台账），delegation 不能反向依赖 harness（循环导入）。所以 delegation 只留"插口"，具体怎么取 agent、造上下文、包 span 都由 harness 注入。

## Q19：`harness.py L297` `agent.run` 是哪个函数？

是 **`Agent.run()`**，定义在 **agents.py L303**：

```python
def run(self, task: Optional[SubTask], ctx: RunContext) -> AgentOutput:
```

注意它和最早问的 `runner.run` 不是同一个：

| 方法 | 位置 | 干什么 |
|:---|:---|:---|
| `Harness.run()` | harness.py L371 | 一次**完整团队协作**总入口（拆活、派活、汇总） |
| `agent.run()` | agents.py L303 | **单个工人**执行自己那一块任务 |

调用关系：
```
Harness.run()           ← server.py 调的（团队总入口）
   └─ 主管拆任务 → 派给多个 worker
        └─ agent.run(task, ctx)   ← harness.py L297 调的（单个工人干活）
```

---

# 附录 A：关键代码位置总表

| 功能 | 文件 | 行号 |
|:---|:---|:---|
| team_run 接口 | modules/api/server.py | L712 / 调用在 L752 |
| run（native） | modules/multiagent/harness.py | L371 |
| run（crewai） | modules/multiagent/crewai_backend.py | L222 |
| Trace 类 | modules/orchestration/tracer.py | L105 |
| Span 类 | modules/orchestration/tracer.py | L60 |
| Tracer / new_trace | modules/orchestration/tracer.py | L215 / L231 |
| start_span / end_span | modules/orchestration/tracer.py | L244 / L259 |
| bind_trace | modules/orchestration/orchestrator.py | L106 |
| RunContext | modules/multiagent/agents.py | L94 |
| _new_delegation | modules/multiagent/harness.py | L233 |
| MessageBus.clear | modules/multiagent/coordination.py | L151 |
| DelegationPolicy | modules/multiagent/delegation.py | L67 |
| DelegationLedger / can() | modules/multiagent/delegation.py | L89 / L115 |
| 防环检查 | modules/multiagent/delegation.py | L134-L140 |
| DelegationExecutor / delegate() | modules/multiagent/delegation.py | L210 / L253 |

---

# 附录 B：面试/答辩要点

1. **runner.run 是什么？** 团队协作主入口，主管拆任务→并行→汇总→终审；native/crewai 两后端接口一致可切换。
2. **Trace/Span 关系？** Trace 是总看板，Span 是任务卡片；平铺表存储、加锁并发写、靠 root_id+children 还原树。
3. **RunContext 作用？** 全队共享的资料袋（问题/上游产出）+ 公共设施（bus/gate）+ 委派三件套，避免参数层层透传。
4. **怎么防委派死循环？** chain 经手人名单只增不减，受托方在名单上即拒绝；深度上限做第二道保险。
5. **委派被拒为什么留痕、为什么返回 None？** "没想转"和"想转被拦"要可区分；委派是可选能力，不能拖垮主流程。
