# 学习知识点

本项目阅读过程中沉淀的知识点，一篇一个主题，讲「原理 + 本项目现状 + 怎么补」。

## 索引

| 编号 | 主题 | 一句话 |
|---|---|---|
| 01 | [断点续跑：项目中如何实现](./01-断点续跑.md) | 存档链路 + `agent_state` 30 个字段解析；现状是只存不读 |
| 02 | [数据库写入那几行](./02-数据库写入那几行.md) | `execute(模板, 值)` 为啥要分家写；`commit()` 漏了会怎样 |
| 03 | [装饰器 handle_exception](./03-装饰器handle_exception.md) | `@handle_exception(...)` 在套什么壳；reraise 与 default 怎么选 |
| 04 | [fetchone 与游标](./04-fetchone与游标.md) | `execute` 只是下单，`fetch` 才是取餐；为什么取出的是 dict |
| 05 | [conn 与 cur 的区别](./05-conn与cur的区别.md) | 连接管事务、游标管执行；`commit()` 该挂谁身上 |
| 06 | [MCP 接入全流程](./06-MCP接入全流程.md) | 子进程 + JSON-RPC：拉起→握手→发现→适配→调用 6 阶段完整链路 |
| 07 | [依赖注入与 Agent 全链路](./07-依赖注入与Agent全链路.md) | 5 个组件怎么接线；prepare→reason↔act→answer→save 完整执行链路 |
| 08 | [Skill 与 Tool 的区别](./08-Skill与Tool的区别.md) | Skill=作业指导书(文本)，Tool=螺丝刀(代码)；`bound_tools` 只是软引导 |
| 09 | [一句话走完全流程函数调用清单](./09-一句话走完全流程函数调用清单.md) | 按执行顺序编号列出「敲一句话→拿到回答」经过的每一个函数 |
| 10 | [本地工具与 MCP 远程工具相结合](./10-本地工具与MCP远程工具相结合.md) | 同一个注册表装填两次：本地保底、MCP 扩展、适配器抹平；现状是 `user_level` 没透传 |
| 11 | [Prompt 拼接与循环控制权](./11-Prompt拼接与循环控制权.md) | 7 个槽位拼 prompt；Milvus 只是数据源；循环由 `add_edge("act","reason")` 控制，不是 ReAct |

## 计划中（想到再加）

- 02 短期记忆 / 长期记忆 / Checkpoint 三者怎么配合
- 03 向量检索：对称 vs 非对称，问句向量与摘要向量之争
- 04 pgvector 的 `<=>` 算子、ivfflat 索引、`COALESCE` 为什么会让索引失效
- 06 RAG 的两阶段：粗召回（向量）→ 精排（BGE rerank）
