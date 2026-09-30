# 文档索引

> 按「先跑通 → 再深入 → 再扩展」的顺序组织。每篇开头都有一句话结论。

---

## 一、主线（建议按这个顺序读）

| 顺序 | 文档 | 一句话 |
|---|---|---|
| 1 | [`Harness执行流程.md`](./Harness执行流程.md) | **工具侧**：模型说要调工具 → 结果回灌 prompt，六阶段 + 拦截码 + 配置映射 |
| 2 | [`Planning与执行.md`](./Planning与执行.md) | 先规划再执行这条链路怎么接，和 ReAct 有什么区别（速查） |
| 3 | [`Planning全流程-阅读文档.md`](./Planning全流程-阅读文档.md) | **从一句话到最终答案**：十阶段逐步走通，带实测计划进度与重规划输出 |
| 4 | [`多Agent协作全景.md`](./多Agent协作全景.md) | **多 Agent 脉络总览**：规划→派活→委派→一致性→仲裁→重规划→终审→汇总 |
| 5 | [`多AgentHarness执行流程骨架.md`](./多AgentHarness执行流程骨架.md) | **`run()` 的骨架**：构造期/运行期/收口三段、十步流水线、分支点、退出路径 |
| 6 | [`HITL与多Agent协调.md`](./HITL与多Agent协调.md) | 人怎么插手：工具审批、仲裁升级、终审修改；成员之间怎么对话 |
| 7 | [`多Agent委派.md`](./多Agent委派.md) | 执行者把活转给别人：8 步链路、五道闸、降级表、API（速查） |
| 8 | [`多Agent委派-阅读文档.md`](./多Agent委派-阅读文档.md) | **从零读懂委派**：为什么需要、跟着代码走一遍、实测输出、自测 10 问 |

---

## 二、能力专项

| 文档 | 内容 |
|---|---|
| [`工具Harness.md`](./工具Harness.md) | 工具侧治理：能力声明、调用前校验、人工审批、沙箱、预算、结果过滤 |
| [`评测与性能基线.md`](./评测与性能基线.md) | 评测集、跑法、基线数字与回归口径 |

---

## 三、三张对照表（速查）

### 任务怎么流转

| 机制 | 发起方 / 时机 | 是否真执行 | 落在哪 |
|---|---|---|---|
| 派活 dispatch | 主管 / 规划阶段 | 是 | `PlanResult.subtasks` |
| 委派 delegation | 执行者 / 执行过程中 | 是，产出回传 | `RunResult.delegations` |
| 转派消息 DELEGATE | 谁都能发（worker 发会被拒） | 否 | `MessageBus` |

### 观测三条线

| 线 | 粒度 | 回答什么 |
|---|---|---|
| `RunResult.trace` | 12 类事件流 | 发生了什么 |
| `RunResult.spans` | span 树 | 慢在哪一环 |
| 日志 | 每步一行 INFO | 启动参数与汇总数字 |

### 默认开关（装上去不改变现有行为）

| 能力 | 默认 | 为什么默认关 |
|---|---|---|
| 委派 | 关 | 要多一次"要不要转派"的 LLM 决策 |
| Planning | 关 | 复杂流程才划算，简单问题不该付这次 LLM 的钱 |
| 重规划 | 关 | 要多调一次 LLM，是真金白银的成本 |
| 人工介入 | 关 | 没人值守会一直等到超时 |
| 一致性检测 | 开 | 只多记一条 trace，不改答案路径 |
| 编排治理 | 开 | 限额给得很宽（20 QPS / 突发 40），正常协作撞不到 |

---

## 四、代码与演示对照

| 想看什么 | 代码 | 演示脚本 |
|---|---|---|
| 团队运行时 | `modules/multiagent/harness.py` | `examples/multiagent_demo.py` |
| 委派 | `modules/multiagent/delegation.py` | `examples/multiagent_delegation_demo.py` |
| 编排治理 | `modules/orchestration/` | `examples/orchestration_demo.py` |
| 工具治理 | `modules/harness/` | `examples/tool_harness_demo.py` |
| Planning | `modules/agent/planner.py` + `graph.py` 的 plan 节点 | `tests/test_agent_planning.py` |
| 人工介入 | `modules/hitl/` | `examples/hitl_coordination_demo.py` |
| 测试 | `tests/`（`pytest tests -q`） | — |

---

## 五、配置速查

`.env` 里与多 Agent 相关的开关（完整清单见 `.env.example`）：

```
MULTIAGENT_ENABLED=true            # 关闭时 /agent/* 接口返回 503
MULTIAGENT_MAX_WORKERS=4           # 并行线程数
MULTIAGENT_MAX_TOTAL_STEPS=30      # 整次运行总步数预算
MULTIAGENT_TIMEOUT_SECONDS=120     # 整次运行总超时
MULTIAGENT_REVIEW_ENABLED=true     # 终审

MULTIAGENT_DELEGATION_ENABLED=false       # 委派总开关
MULTIAGENT_DELEGATION_MAX_DEPTH=1         # 最多转派几层
MULTIAGENT_DELEGATION_MAX_PER_RUN=6       # 单次运行最多委派几次
MULTIAGENT_DELEGATION_MAX_PER_AGENT=2     # 单个 Agent 最多委派几次
```

优先级：**显式传参 > `.env` > 默认值**（`TeamConfig` 上的同名参数会覆盖 `.env`）。
