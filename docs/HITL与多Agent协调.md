# Human-in-the-Loop 与 多 Agent 协调

> 两件事：让 Agent 在关键节点**停下来等人拍板**，以及让多个 Agent **在有分歧时能收敛**。

---

## 一、为什么需要这两块

只有 Supervisor-Worker 派活时，有三件事没人管：

| 缺口 | 后果 |
|---|---|
| Agent 全程自己做主 | 删库、外发、查他人数据这类不可逆动作也照做 |
| 产出互相矛盾 | 主管只能"挑一个"，没有明确的收敛规则 |
| 一批人失败了 | 标注"部分失败"就收工，没人补位 |

HITL 补第一项，协调能力补后两项。

---

## 二、HITL：六种中断点、四种动作

### 六个中断点（`InterruptKind`）

| 中断点 | 什么时候停 | 默认兜底 |
|---|---|---|
| `before_plan` | 主管拆完任务、派活前 | approve |
| `before_task` | 单个子任务执行前 | approve |
| `before_tool` | 调用敏感工具前 | **reject** |
| `before_final` | 最终答案发出前 | approve |
| `on_review_fail` | 终审不通过 | approve |
| `on_dispute` | 多 Agent 结论冲突 | approve |
| `on_budget` | 步数/时间超阈值 | approve |

**只有 `before_tool` 默认拒绝**：调工具是唯一"做了就不可逆"的环节，宁可不做也不能做错；其余环节卡住的代价比放行的代价更大。

### 四种人工动作（`HumanAction`）

| 动作 | 含义 | Agent 怎么用 |
|---|---|---|
| `approve` | 照原样放行 | 继续 |
| `modify` | 改一下再放行 | 用 `edited` 替换（工具参数 / 指令 / 最终答案） |
| `reject` | 这一步不做 | 跳过，把"未获批准"作为观察结果回喂模型 |
| `abort` | 整次终止 | 运行结束，状态置 FAILED |

### 超时怎么办

每个请求自带 `timeout_seconds` 和 `default_action`。人一直不来：

```
BEFORE_TOOL  超时 → reject（不执行这个工具）
BEFORE_FINAL 超时 → approve（答案已过汇总与终审）
```

**不会挂死**。这是最关键的一条：没有人值守时系统仍然出结果，只是按保守策略走。

---

## 三、和已有的 `review_queue` 有什么不同

| | `review_queue`（PG） | `hitl`（本模块） |
|---|---|---|
| 时机 | 事后离线复核 | 事中阻塞式拍板 |
| 影响 | 不改变已给出的答案 | 改变接下来的执行 |
| 谁来 | 有空再看 | 不回复就等（或超时兜底） |
| 比喻 | **事后看回放** | **直播时喊停** |

---

## 四、代码怎么用

```python
from modules.hitl import build_gate, auto_resolver, HumanAction

# 生产：真的等人（HTTP 审批页点按钮）
gate = build_gate(enabled=True, sensitive_tools=["data_query", "mysql_*"])

# 测试 / CI / 无人值守：注入自动答复器，不阻塞
gate = build_gate(enabled=True, sensitive_tools=["data_query"],
                  resolver=auto_resolver(HumanAction.REJECT))

# 链路里就一行
go, args = gate.ask_tool("data_query", {"table": "orders"})
if not go:
    ...  # 跳过这一步
```

完整端到端（harness 自动接入）：

```python
from modules.multiagent import build_harness
h = build_harness(llm=llm, gate=gate)      # 不传 gate 时按 HITL_ENABLED 配置决定
r = h.run("公司今年营收是多少？")
print(r.interrupt_count, r.final_answer)
```

### HTTP 审批端点

| 端点 | 用途 |
|---|---|
| `GET /hitl/pending?limit=50` | 当前等着人拍板的请求（审批页轮询它） |
| `GET /hitl/stream` | SSE 长连接，一有中断就推一帧（审批页免轮询） |
| `POST /hitl/reply` | 提交决策：`{request_id, action, modified, note}` |
| `GET /hitl/records?limit=50` | 历史留痕（审计：谁、何时、最终按什么动作执行） |

重复提交返回 `409`，非法 action 返回 `422`，未启用返回 `503`。

**闸门必须用进程级共享的那个**（`get_global_gate()`）：链路和审批端点若各持一个 gate，人点了批准链路永远收不到。多副本部署时把 store 换成共享存储即可，接口一行不用改。

> 浏览器原生 `EventSource` 发不出自定义请求头，而本项目鉴权走 `X-API-Key`。
> 网页端请用 `fetch` + `ReadableStream`，或让网关把 key 注入请求头。

### 通知到人：`modules/hitl/notify.py`

`gate.ask()` 只负责"登记 → 等 → 超时兜底"。登记完请求只是躺在内存里，
**没通知就等于"系统自己按兜底动作处理了，人全程不知情"**。通知器挂在 `gate.on_event` 上补这一环，三条通道：

| 通道 | 开关 | 说明 |
|---|---|---|
| console | `HITL_NOTIFY_CONSOLE`（默认 `true`） | 打终端 `stderr`，本地 / CLI / CI 立刻看得见 |
| webhook | `HITL_NOTIFY_WEBHOOK`（默认空） | 钉钉 / 飞书 / 企业微信 / 自定义 URL；后台线程发，不吃等待时间 |
| SSE | 无（连上就有） | `/hitl/stream` 消费订阅队列，审批页不用轮询 |

```python
from modules.hitl import build_gate, InterruptNotifier

# 推钉钉（kind=auto 按域名自动识别平台）
notifier = InterruptNotifier(webhook="https://oapi.dingtalk.com/robot/send?access_token=xxx")
gate = build_gate(enabled=True, sensitive_tools=["send_notice"], on_event=notifier.notify)
```

终端看到的就是这段（`format_interrupt_text` 渲染，三个渠道共用同一份文本）：

```
[待人工拍板] before_tool hitl_3f9a2c11
标题：调用工具：send_notice
原因：工具 send_notice 在人工批准白名单内
上下文：{"tool": "send_notice", "arguments": {"to": "U007"}}
可选：approve / modify / reject / abort
60s 内无人处理 → 按 reject 兜底
处理：POST /hitl/reply  {"request_id": "hitl_3f9a2c11", "action": "approve"}
```

**通知失败不影响审批**：webhook 超时 / 报错 / 没配，都只记 warning，Agent 照常等人、超时、兜底。

---

## 五、多 Agent 协调：四件事

### 1. 消息总线（横向对话）

```python
bus.post("analyst_1", "你的口径是含税还是不含税？",
         receiver="researcher_1", kind=MessageKind.CLARIFY)
```

| 消息类型 | 用途 |
|---|---|
| `finding` | 共享发现 |
| `clarify` | 请求澄清 |
| `delegate` | 转派（只"说一句这块你来做"；真要让活被干掉并拿回结果，用委派机制） |
| `objection` | 异议（只作为仲裁输入） |

成员能在 prompt 的「同伴消息」里看到前人的发现，避免重复查同一件事。三条约束保证协作不退化成群聊：消息只在同一批次内有效、总线有容量上限、异议不触发新的执行轮次。

> **转派消息 ≠ 委派**：消息不执行、不回传、不受治理；
> 真正的"把活转出去、拿回产出、并入结论"见 [`多Agent委派.md`](./多Agent委派.md)。
>
> **脉络位置**：一次运行的完整顺序（规划→派活→委派→一致性→仲裁→重规划→终审→汇总）
> 见 [`多Agent协作全景.md`](./多Agent协作全景.md)，本文档讲的是其中的人工介入与成员对话。

### 2. 一致性判定（确定性，不调 LLM）

用字符 2-gram Jaccard 算产出相似度，低于阈值（默认 `0.35`）判冲突：

```
"营收 1200 万，来自订单系统"  vs  "营收 800 万，同比 -5%"
→ 相似度 0.18 < 0.35 → 判为冲突，需仲裁
```

**为什么不用 LLM 判**：每批都调一次太贵，且模型自己会漂移。确定性算法今天判冲突、明天同样输入还判冲突，适合进 trace 做追溯。LLM 只在**确认有冲突后**才介入仲裁。

### 3. 仲裁

冲突（或无多数派）时，由 reviewer 出裁决结论，作为一份额外产出交给主管汇总——主管看到的是「三份矛盾产出 + 一份裁决」，而不是自己在矛盾里硬选。

### 4. 重规划

有人失败时，主管只针对未完成的部分重新拆活：

```
第 1 次重规划：新增 1 个任务，待补 1 项失败
```

默认只重规划 `1` 次：失败往往是任务本身不可解（缺数据、工具不可用），重试多次结果一样，多试就是烧钱。重规划后 `ctx.query` 必须还原——不还原的话下一批 Agent 会拿到"上一轮谁失败了"这类无关信息。

---

## 六、配置项（`config.py` / `.env`）

```bash
# 多 Agent 协调
MULTIAGENT_CONSENSUS_ENABLED=true       # 一致性检测（默认开，只多记一条 trace）
MULTIAGENT_CONSENSUS_THRESHOLD=0.35     # 相似度低于此值判冲突
MULTIAGENT_ARBITRATION_ENABLED=true     # 冲突时让审核者仲裁
MULTIAGENT_SHARE_FINDINGS=true          # 成员产出共享到总线
MULTIAGENT_REPLAN_ENABLED=false         # 失败重规划（默认关：要多调一次 LLM）
MULTIAGENT_MAX_REPLANS=1

# Human-in-the-Loop
HITL_ENABLED=false                      # 总开关（默认关：开启后会真的停下来等人）
HITL_KINDS=                             # 启用哪些中断点，逗号分隔；留空=全部
HITL_SENSITIVE_TOOLS=data_query,mysql_* # 需批准的工具名，支持 mysql_* 前缀
HITL_TIMEOUT_SECONDS=60                 # 单次等待上限

# 通知到人（中断抛出的那一刻怎么让人知道）
HITL_NOTIFY_CONSOLE=true                # 打终端 stderr
HITL_NOTIFY_WEBHOOK=                    # 机器人地址；留空=不推送
HITL_NOTIFY_WEBHOOK_KIND=auto           # auto 按域名识别 dingtalk/feishu/wecom，认不出走 generic
HITL_NOTIFY_WEBHOOK_TIMEOUT=3.0         # 推送超时（后台线程发，不算进等待时间）
```

**两个默认值的取舍**：一致性检测默认开（成本只是多记一条 trace）；重规划和 HITL 默认关（前者要多调 LLM，后者会真的阻塞）。这样装上去默认行为基本不变，想用再打开。

---

## 七、验证

```bash
python -m pytest tests/ -q                        # 135 passed（新增 63 个）
python examples/hitl_coordination_demo.py         # 五幕演示，全程离线
```

演示输出（真实运行）：

```
状态=partial  耗时=24ms  重规划=1 次  人工介入=6 次
一致性: conflict=True similarity=0.1818
[replan]     第 1 次重规划：新增 1 个任务，待补 1 项失败
[arbitration] 裁决：采用 1200 万这一口径（有订单系统数据支撑），800 万缺少来源说明。
[hitl]       冲突裁决: modify
最终答案: 人工裁决：以 1200 万为准，800 万缺少来源，已标注待核实。
```

---

## 八、文件清单

| 文件 | 职责 |
|---|---|
| `modules/hitl/models.py` | 中断请求 / 人工回复 / 归档记录的数据契约 |
| `modules/hitl/store.py` | 请求存取与 Event 等待唤醒（可换共享存储） |
| `modules/hitl/policy.py` | 什么时候打断、超时按什么动作兜底 |
| `modules/hitl/gate.py` | 人工闸门：ask / ask_tool / ask_final / 自动答复器 |
| `modules/multiagent/coordination.py` | 消息总线、一致性判定、仲裁、重规划 |
| `modules/multiagent/models.py` | 新增 `AgentMessage` / `ConsensusResult` 与 6 项配置 |
| `modules/multiagent/harness.py` | 主流程接入（计划/任务/工具/冲突/终稿五处中断 + 重规划） |
| `modules/api/hitl_routes.py` | 三个审批端点 |
| `examples/hitl_coordination_demo.py` | 离线演示（脚本模型顶替 LLM） |
| `tests/test_hitl.py` / `test_coordination.py` / `test_hitl_api.py` | 63 个新增用例 |
