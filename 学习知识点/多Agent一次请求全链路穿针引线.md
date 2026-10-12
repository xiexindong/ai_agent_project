# 多 Agent 一次请求全链路穿针引线

> 从 `POST /agent/team/run` 进来的一个请求，到 `RunResult` 返回，中间经过**每一个函数**的调用顺序、位置、作用。
> 主线只讲"一次成功执行"的正路；每个阶段旁标注可选分支（直答 / HITL / 冲突 / 失败）。
> 配套阅读：《多Agent团队协作全链路问句走读.md》（Q1-Q23 问句式走读，本文是它的"正路总纲"版）。

---

## 一、全景链路图（先看这张）

```
客户端 POST /agent/team/run {"query": "查Q3/Q4营收，算环比，写报告"}
  │
  ▼ ① HTTP 层
team_run(req, user)                          server.py L713
  │  校验 harness / 选后端 / 验图片 / 兜底 session_id
  ▼
Harness.run(query, mm, session_id)           harness.py L371
  │  ② 起观测、组上下文、装配委派
  │
  ▼ ③ 规划（第 1 次 LLM 调用）
SupervisorAgent.plan(ctx, roster)            agents.py L491
  │  PLAN_PROMPT 填空 → invoke_structured → model_validate（失败重试→降级）
  │
  ▼ ④ 直答闸门
plan.is_direct? ──是──► final_answer = direct_answer ──► return（收工）
  │ 否
  ▼ ⑤ 计划人工确认（可选 HITL）
gate.ask(BEFORE_PLAN)                        harness.py L450
  │  approve 继续 / abort·reject 直接失败返回
  ▼ ⑥ 分批（纯算法，无 LLM）
topo_batches(plan.subtasks)                  harness.py L339
  │  Kahn 拓扑分层 → [[t1],[t2,t3],[t4]] + 环剔除
  ▼ ⑦ 逐批并行（批间串行、批内并行）
for batch in batches:                        harness.py L482
  │  超时/步数预算两道检查 → dispatch.batch_N span
  └─ _run_batch(batch, ...)                  harness.py L771
       │  ThreadPoolExecutor(max_workers=min(配置, 批大小))
       ├─ futures = {pool.submit(_one, t): t}     L838  ★并行在这
       └─ 逐个 future.result(timeout) 收账        L839
            │
            └─ _one(task)                    harness.py L786（内嵌函数）
                 ├─ agents.get(assignee) 按名取人
                 ├─ 任务前 HITL（可选）
                 ├─ emit agent_start / agent span
                 ├─ agent.run(task, ctx)     agents.py L303  ★真正干活
                 └─ emit agent_end
  │  每个产出：result.outputs + steps_used + ctx.upstream + 总线共享(FINDING)
  ▼ ⑧ 失败重规划（可选，默认关）
replan.need_replan → _try_replan             harness.py L530
  ▼ ⑨ 一致性 → 仲裁 → 人工裁决（可选）
consensus.detect → 少数派异议消息             harness.py L546
  │  arbitrator.arbitrate → 裁决进汇总输入     L567
  │  gate.ask(ON_DISPUTE) → human_final       L578
  ▼ ⑩ 汇总（第 2 次主链 LLM 调用）
SupervisorAgent.synthesize(ctx, outputs)     agents.py L545
  │  human_final 非空则跳过；SYNTH_PROMPT → llm.invoke；失败回退拼接
  ▼ ⑪ 终审（可选，第 3 次 LLM 调用）
ReviewerAgent.review(ctx, draft)             agents.py L580
  │  REVIEW_PROMPT → invoke_structured → ReviewVerdict；解析失败默认放行
  ▼ ⑫ 终稿人工确认（可选 HITL）
gate.ask_final(final_answer)                 harness.py L650
  ▼ ⑬ 定状态收口
status=SUCCESS/PARTIAL；finally：唤醒等待审批 / 委派收口 / 关 span
  │
  ▼
return RunResult（HTTP 响应给客户端）
```

---

## 快速跳转目录（点文件名直达代码）

| 文件 | 职责 |
|---|---|
| [modules/api/server.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py) | HTTP 接口层：/agent/team/run 入口 |
| [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py) | 团队编排总入口 Harness.run |
| [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py) | 三个 Agent 类与 BaseAgent 执行逻辑 |
| [modules/multiagent/delegation.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py) | 委派台账与五道闸 |
| [modules/multiagent/models.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py) | SubTask / PlanResult 等数据模型 |
| [modules/hitl/gate.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/gate.py) | 人工审批闸门（阻塞询问） |
| [modules/hitl/store.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py) | 审批请求存储与 Event 唤醒 |
| [modules/hitl/notify.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/notify.py) | 审批通知三通道扇出 |
| [modules/api/hitl_routes.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py) | SSE 卡片下行 / 回帖上行接口 |

> 注意：项目里有**两个 harness.py**——本文提到的 `harness.py` 一律指 `modules/multiagent/harness.py`（团队编排）；
> 你 IDE 里打开的 `modules/harness/harness.py` 是工具治理 Harness，是另一个模块，别走错门。

---

## 二、逐阶段拆解：每步函数的输入、动作、产出

### 阶段① HTTP 入口：`team_run`

**位置**：[server.py L712-752](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py#L712-L752)

| 项 | 内容 |
|---|---|
| 输入 | `TeamRunRequest`（query、images、backend、session_id）+ JWT 用户 |
| 前置检查 | `ctx.harness` 不存在 → 503（多Agent模块未启用） |
| 后端选择 | `req.backend > 全局配置 > native`；选 crewai 时懒加载 CrewAIBackend |
| 图片安全 | 只放行 `data:` / `http(s)` 前缀，本地路径拒绝（防路径穿越读服务器文件） |
| 输出 | 调 `runner.run(req.query, mm=..., session_id=...)`，返回 `RunResult` |

**作用一句话**：把 HTTP 请求"翻译"成 Harness 能听懂的参数，安全检查全在这层做完。

### 阶段② run() 开场：起观测、组上下文

**位置**：[harness.py L390-417](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L390-L417)

| 行 | 调用 | 作用 |
|---|---|---|
| L390-391 | `time.time()` / `RunResult(...)` | 记开始时刻；初始化结果对象 |
| L400 | `tracer.new_trace("multiagent.run")` | 起观测链路（Trace），名字定死便于检索 |
| L403-404 | `orchestrator.reset()` / `bind_trace(trace)` | 清上一轮消息计数；把 trace 挂给编排器 |
| L405 | `tracer.start_span("run")` | 开根 span：整次运行在链路树上的"树干" |
| L411-412 | `RunContext(query, mm, session_id, bus=self.obus, gate=self.gate)` | 组装全队共享上下文；bus 用编排门面（Agent 只能读+受治理地发） |
| L414 | `_new_delegation(ctx, result)` | 装配本轮委派台账 DelegationLedger + 执行器（必须先于派活） |
| L415 | `self.bus.clear()` | 清空消息总线：旧异议不跨运行 |
| L416-417 | `deadline` / `steps_used=0` | 算总超时截止时刻；步数预算清零 |

**作用一句话**：开工前的"三通一平"——通水电（上下文）、通道路（trace/事件）、清场地（bus）。

### 阶段③ 主管规划：`SupervisorAgent.plan`

**位置**：[agents.py L491-541](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L491-L541)，调用点 [harness.py L430](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L430)（外面包着 `plan` span）

| 步骤 | 代码 | 说明 |
|---|---|---|
| 备料 | L501-515 | roster 拼成员名单（name/角色/描述/工具）→ 填进 `PLAN_PROMPT` 四个占位符（query/modality/roster/schema） |
| 调模型 | **L517** `self.llm.invoke_structured(...)` | ★第 1 次主链 LLM 调用；json_object 强制 JSON、温度 0.2 |
| 验票 | L518 `PlanResult.model_validate(data)` | Pydantic 校验：字段齐不齐、assignee 填没填 |
| 重试 | L519-533 | 校验失败把错误原文回喂再试一次（模型最常见错是漏字段） |
| 降级 | L534-541 | 二次仍失败 → 单任务直答 `t1 派给 roster[0]`，团队不停摆 |

**产出**：`PlanResult`（thought / subtasks / direct_answer / needs_review）。

### 阶段④ 直答闸门

**位置**：[harness.py L438-444](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L438-L444)

`plan.is_direct` 为真（寒暄/单步问题不值得开团队）→ 直接用 `plan.direct_answer` 当最终答案，记 `done` 事件返回。**后面的阶段全部跳过。**

### 阶段⑤ 计划人工确认（可选 HITL）

**位置**：[harness.py L446-473](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L446-L473)

- 条件：`gate` 存在且 policy 开了 `BEFORE_PLAN`；
- `gate.ask(...)` **阻塞**等工作线程里的审批结果（前端经 SSE 收卡片、POST /hitl/reply 回帖唤醒，见走读文档 Q23）；
- `approve` → 继续；`abort` / `reject` / 超时未批准 → 组装失败结果提前 return。

### 阶段⑥ DAG 分批：`topo_batches`

**位置**：[harness.py L339-364](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L339-L364)，调用点 L476

| 项 | 内容 |
|---|---|
| 算法 | Kahn 分层：每轮找出"依赖全部就绪"的任务凑成一批 |
| 输入 | `[t1, t2(←t1), t3(←t1), t4(←t2,t3)]` |
| 输出 | `([[t1], [t2,t3], [t4]], [])`——批次列表 + 环中任务 id |
| 防挂死 | 一轮下来凑不出任何就绪任务 = 有环/引用不存在的 id → 剔除返回（L357-359），否则 while 排不完整次运行卡死 |
| 同批排序 | 按 priority 降序，重要的先跑（L360） |
| 成本 | 纯 Python 循环，**无 LLM 调用** |

### 阶段⑦ 逐批并行：`for batch` + `_run_batch` + `_one` + `agent.run`

**外层循环**：[harness.py L482-527](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L482-L527)——批与批之间串行（下游依赖上游产出）。

每批开始前两道预算检查：

| 检查 | 触发后果 |
|---|---|
| `time.time() > deadline`（L483） | 本批剩余任务全部记"总超时，未执行"，status=TIMEOUT，break |
| `steps_used >= max_total_steps`（L491） | 本批剩余任务全部记"预算耗尽"，break |

**`_run_batch`**：[harness.py L771-851](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L771-L851)

```
L837  ThreadPoolExecutor(max_workers=min(配置上限, 批大小))   ← 不多开空转线程
L838  futures = {pool.submit(_one, t): t}                    ← submit 瞬间全部在后台线程开跑 ★
L839  for future, task in futures.items():                   ← 按提交顺序收账（非完成顺序）
L842      future.result(timeout=deadline 剩余)               ← 带超时等待
L843-849  超时/异常 → AgentOutput.failed（单任务炸不连累同批）
```

**`_one(task)`**：[harness.py L786-834](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L786-L834)（内嵌函数，每个线程拿一份"单任务全套动作说明书"）

| 步骤 | 位置 | 作用 |
|---|---|---|
| 按名取人 | L787 | `self.agents.get(task.assignee)`；取不到 → 失败产出"未注册的 Agent" |
| 任务前 HITL | L797-813（可选） | `gate.ask(BEFORE_TASK)`；reject→失败产出；modify→用改后指令造 task 副本 |
| 起止事件 | L815 / L831 | `emit("agent_start")` / `emit("agent_end")`，前端实时可见 |
| Agent span | L821-828 | 包一层 `agent.{name}` span，记录输入指令与成败/步数/耗时 |
| 真正干活 | **L825** `out = agent.run(task, ctx)` | ★进入单个 Agent 的世界（见下方支线） |

**批后记账**（外层循环 L511-527）：每个产出写四处——
`result.outputs`（对外账单）、`steps_used += out.steps`（预算）、`ctx.upstream[task_id]`（下游可见）、成功产出经 `orchestrator.send(..., kind=FINDING)` 截 400 字上总线共享（防重复查同一件事）。

### 阶段⑧ 失败重规划（可选，默认关）

**位置**：[harness.py L529-531](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L529-L531)。`replan_enabled` 且有失败产出 → `_try_replan` 补派。默认关闭：要多调一次 LLM，是真金白银。

### 阶段⑨ 一致性 → 仲裁 → 人工裁决（可选）

**位置**：[harness.py L533-602](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L533-L602)

| 层 | 触发条件 | 动作 |
|---|---|---|
| 一致性检测 | consensus_enabled 且产出≥2 | `consensus.detect(outputs)` 比相似度；冲突时给少数派发 `OBJECTION` 异议消息 |
| 机器仲裁 | arbitration_enabled 且 needs_arbitration | `arbitrator.arbitrate(...)`（裁判=reviewer 或主管）；裁决结论 append 进 `synth_inputs` |
| 人工裁决 | gate 开了 ON_DISPUTE | `gate.ask(ON_DISPUTE)` 展示各方观点；modify→`human_final` 直接当终稿；abort→失败返回 |

设计要点：主管汇总时看到的是"矛盾产出 + 官方裁决"，不是自己在矛盾里硬选。

### 阶段⑩ 汇总：`SupervisorAgent.synthesize`

**位置**：[agents.py L545-557](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L545-L557)，调用点 [harness.py L612](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L612)

- **优先级**：`human_final` 非空 → 直接采纳跳过汇总（L605-607）；
- 有成功产出 → 拼 `SYNTH_PROMPT`（成功取 content，失败如实标"该部分未能完成"）→ **`self.llm.invoke`（第 2 次主链 LLM 调用）**；
- 汇总 LLM 挂了 → 回退为成功产出的原文拼接（L556-557）；
- 零成功产出 → 超时话术 / FAILED 提前 return。

### 阶段⑪ 终审：`ReviewerAgent.review`（可选）

**位置**：[agents.py L580-603](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L580-L603)，调用点 [harness.py L640](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L640)

- 条件：`review_enabled` 且 `plan.needs_review`（主管规划时就决定了要不要审）；
- `_first_by_role(REVIEWER)` 找终审员 → REVIEW_PROMPT（草稿截 4000 字）→ **`invoke_structured`（第 3 次 LLM 调用）** → `ReviewVerdict(passed/score/issues/suggestion)`；
- 解析失败不阻断：返回"默认放行"的中性结论。

### 阶段⑫ 终稿人工确认（可选 HITL）

**位置**：[harness.py L644-658](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L644-L658)。`gate.ask_final(...)` 最后一道关：modify 用人改的文本、reject 则不发送（给"未通过人工审核"话术）。**默认兜底是放行**——终稿已过汇总+终审，卡住比放行更糟。

### 阶段⑬ 定状态收口

**位置**：[harness.py L660-706](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L660-L706)

| 环节 | 逻辑 |
|---|---|
| 定状态 | 有失败成员 → PARTIAL，否则 SUCCESS；TIMEOUT 不覆盖（超时要扩容/减并行，失败要查执行者，运维信号不同） |
| except 兜底 | 主流程任何异常 → status=FAILED 返回，**不向外抛** |
| finally 三收口 | ① `gate.cancel_waiting()` 作废还在等的审批，释放挂着的线程；② 委派台账复制进 `result.delegations`；③ 关根 span、封 trace、span 快照写进结果——**任何 return 路径都留完整链路** |

---

## 三、支线：单个 Agent.run() 内部（WorkerAgent / 继承自 BaseAgent）

**位置**：[agents.py L303-381](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L303-L381)，被 `_one` 的 L825 调用。

```
run(task, ctx)
 ├─ 备料四路文本
 │   ├─ instruction        工单指令（无工单则用户原话）
 │   ├─ _upstream_text     上游产出（每条截 1200 字）      L398
 │   ├─ describe_images    图片→文字描述（多模态）
 │   └─ bus.inbox_text     同伴消息（最近 20 条，每条截 400 字）L312
 ├─ 委派决策（可选，四道前置：委派器/开关/资格/有任务）      L323-342
 │   ├─ decide_delegation(llm, ...)   ★一次 LLM 调用：要不要转、转给谁
 │   └─ dex.delegate(...)             走五道闸（开关/资格/存在性/深度/防环）
 ├─ _build_prompt(...)     任务/图片/上游/委派产出/同伴/要求 依序拼接  L410
 ├─ 执行（二选一）
 │   ├─ 有工具 → _run_with_tools(prompt, steps_budget)    L355
 │   │    while 步数预算内:
 │   │      llm.invoke_with_tools(messages, schemas)      ★LLM 调用
 │   │      无 tool_calls → 返回文本
 │   │      有 → gate.ask_tool（可选 HITL）→ tools.execute_tool → observe 回喂
 │   └─ 无工具 → self.llm.invoke(prompt)                  L357  ★LLM 调用
 └─ 包装 AgentOutput（agent/role/task_id/content/steps/tool_calls/耗时/委派账）
     异常 → AgentOutput.failed，不崩整轮
```

**上下文隔离的最后边界**（L347-351 注释）：子 Agent 能看到的全部信息已压缩进 prompt 一个字符串；结束后**只有 AgentOutput.content 回流**，内部 messages 与工具日志不进主管上下文。

---

## 四、LLM 调用点盘点（一次"标准"运行）

| # | 调用者 | 方法 | 目的 | 备注 |
|---|---|---|---|---|
| 1 | SupervisorAgent.plan | invoke_structured | 拆任务出 JSON 工单 | 主链必经 |
| 2 | 每个成员 agent.run | invoke / invoke_with_tools | 干活（有工具则每步一调） | 并行发生，次数=Σ各成员步数 |
| 2.5 | decide_delegation（可选） | invoke | 要不要委派的决策 | 委派关闭=零开销 |
| 3 | SupervisorAgent.synthesize | invoke | 汇总成终稿 | 主链必经（直答/人工裁决除外） |
| 4 | ReviewerAgent.review（可选） | invoke_structured | 质量打分 | review_enabled+needs_review |
| — | arbitrator（可选） | invoke | 冲突仲裁 | arbitration_enabled |

---

## 五、观测与治理贯穿点

- **事件（_emit）**：plan / dispatch / agent_start / agent_end / hitl / replan / consensus / arbitration / synthesize / review / error / done——全部经 `_emit`（[harness.py L218](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L218)）双写：result.trace 账本 + on_event 外部喇叭（SSE/审计）。
- **Span 树**：根 `run` → `plan` → `dispatch.batch_N` → `agent.{name}`，消息类另有消息 span；任何 return 路径在 finally 里封树。
- **HITL 四个卡口**：BEFORE_PLAN（计划）→ BEFORE_TASK（任务前，_one 内）→ ON_DISPUTE（冲突裁决）→ BEFORE_FINAL（终稿）+ 工具调用卡口 ask_tool（Agent 内）。
- **预算三道**：总超时 deadline（批循环开头）、步数预算 max_total_steps（批循环开头）、每 Agent 的 max_steps（_run_with_tools 循环）。

---

## 六、通俗比喻：一次装修工程的全流程

1. **客户下单**（HTTP 请求）→ 前台登记核对（team_run：验身份、验图纸格式）；
2. **项目经理接单**（Harness.run）：搭工地看板（trace）、备公共设施（bus/gate）、清空上一单的杂物（bus.clear）；
3. **项目经理画横道图**（plan）：打电话问大脑怎么拆活，画出"水电→泥瓦→油漆"分期表（subtasks+depends_on）；
4. **客户确认图纸**（HITL 可选）；
5. **排分期**（topo_batches）：谁不挡谁的道就同一期进场；
6. **逐期施工**（for batch）：每期包工头把工单全撒出去（submit），多个工种同时干（线程池），干完按单验收（future.result）；
7. 每个工人干活先看**共享白板**（upstream+同伴消息），缺料可以先**转包**给同事（委派），工具使用要过**保安亭**（gate/Harness 治理）；
8. **验料冲突时**（一致性）：质检员比色卡，裁不动请项目经理仲裁，再不行客户拍板（人工裁决）；
9. **项目经理写竣工报告**（synthesize），质检员签字（review），客户验收（终稿 HITL）；
10. **撤场**（finally）：唤醒所有还在等的审批、委派账归档、看板封存（trace 快照）——无论顺利还是返工，档案都要齐。

---

## 七、附录：函数位置速查表

| 函数 | 文件（点击直达） | 行号 | 一句话职责 |
|---|---|---|---|
| team_run | [modules/api/server.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py#L712-L752) | L713 | HTTP 入口：翻译请求+安全检查 |
| Harness.run | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L371-L417) | L371 | 团队总入口：流水线编排 |
| Harness._new_delegation | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L233) | L233 | 装配本轮委派台账与执行器 |
| Harness._emit | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L218-L226) | L218 | 事件双写：trace 账本+外部喇叭 |
| Harness.topo_batches | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L339-L364) | L339 | Kahn 拓扑分层（静态方法） |
| Harness._run_batch | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L771-L851) | L771 | 线程池并行跑一批 |
| _one（内嵌） | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L786-L834) | L786 | 单任务全套：取人/HITL/span/执行 |
| Harness._first_by_role | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L855-L859) | L855 | 按角色找终审员/仲裁者 |
| Harness.evaluate | [modules/multiagent/harness.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/harness.py#L866) | L866 | 回归评测（关键词门禁） |
| SupervisorAgent.plan | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L491-L541) | L491 | 拆任务（LLM#1，重试+降级） |
| SupervisorAgent.synthesize | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L545-L557) | L545 | 汇总（LLM#2，失败回退拼接） |
| WorkerAgent（空壳继承） | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L562-L563) | L562 | 完全复用 BaseAgent.run |
| ReviewerAgent.review | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L580-L603) | L580 | 终审打分（LLM#3，失败放行） |
| BaseAgent.run | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L303-L381) | L303 | 单任务执行：备料/委派/拼prompt/二选一执行 |
| BaseAgent._run_with_tools | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L235-L299) | ~L235 | 工具循环：LLM↔工具直到无 tool_calls |
| BaseAgent._build_prompt | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L410-L427) | L410 | 六段式提示词拼接 |
| BaseAgent._upstream_text | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L398-L406) | L398 | 上游产出文本（截 1200） |
| BaseAgent.llm（property） | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L162-L168) | L163 | 惰性 LLM：注入优先，否则现建 |
| build_default_team | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L608-L639) | L608 | 默认 4 人花名册（AgentSpec 档案） |
| build_agent | [modules/multiagent/agents.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/agents.py#L644-L650) | L644 | 角色→类映射工厂 |
| decide_delegation | [modules/multiagent/delegation.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py) | — | 委派决策（LLM） |
| DelegationLedger.can / delegate | [modules/multiagent/delegation.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/delegation.py#L89) | L89/L253 | 委派五道闸 / 执行转派 |
| HumanGate.ask / ask_final / ask_tool | [modules/hitl/gate.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/gate.py#L135) | L135/L241/L203 | 四类 HITL 阻塞询问 |
| InMemoryInterruptStore.submit | [modules/hitl/store.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/store.py#L118-L139) | L118 | 回帖落账 + Event.set 唤醒 |
| InterruptNotifier.notify | [modules/hitl/notify.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/hitl/notify.py#L174-L181) | L174 | 三通道扇出（终端/SSE/webhook） |
| GET /hitl/stream、POST /hitl/reply | [modules/api/hitl_routes.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/hitl_routes.py#L103) | L103/L140 | 审批卡片下行 / 决策回传上行 |
