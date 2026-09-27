# 主干 01 — `main.py`：装配与入口

> 本文是「主干五站」的第 1 站。目标：不读全部代码，也能知道一句话从哪进去、答案从哪出来。
>
> 主干五站（按这个顺序读，5 个文件打通全链路）：
>
> | 站 | 文件 | 干什么 |
> |---|---|---|
> | 1 | `main.py` | 装配零件 + 命令分发（本文） |
> | 2 | `graph.py` | `prepare → reason ↔ act → answer` 循环 |
> | 3 | `react_agent.py` | 单轮：拼 prompt、调 LLM、解析 tool_calls |
> | 4 | `tool_registry.py` | 工具注册与调度 |
> | 5 | `llm.py` | 真正发请求给大模型 |
>
> 其余目录（`harness/` `enterprise/` `hitl/` `multiagent/` `pipeline/` `skills/` `mcp/`）全是**旁支**，主干通了再读。

---

## 一句话定位

**`main.py` 不参与推理。** 整个文件只做两件事：**装配零件** + **分发命令**。

---

## ① 命令分发（主干只看 `chat`）

文件头 12-38 行列了 8 个子命令，由 `argparse` 分发：

```
chat        交互式对话（CLI）        ← 主干走这个
serve       启动 FastAPI 服务
ingest      知识库文档入库
stats       查看知识库统计
skills      查看技能清单 / 测试意图匹配
mcp-server  MCP Server 独立启动
web-demo    前端预览（全 Mock）
offboard    离职数据处置（跨库对账）
```

其余几个先全部跳过，等主干通了再看。

**一个小细节**：`import` 都写在函数内部（`main.py:67-75`），目的是「跑 `stats` 就不加载大模型那些重依赖」。

---

## ② 装配：6 大件，一行一个

```80:102:main.py
    llm = LLMClient()                        # LLM 客户端：封装模型调用与重试
    pg = PGManager()                         # PG：会话与记忆存储
    memory = MemoryManager(llm_client=llm)   # 记忆管理：摘要 + 向量召回
    ...
    tool_registry = ToolRegistry(harness=tool_harness)   # 工具注册表
    tool_registry.register_batch(tool_registry.get_default_tools())   # 注册内置工具  _tools
    ...
    skill_registry = SkillRegistry()         # 技能注册表（SOP 流程）
    skill_registry.register_batch(get_default_skills())               # 注册内置技能
```

| 零件 | 是什么 | 装配行号 |
|---|---|---|
| `llm` | 跟大模型说话的客户端 | 80 |
| `pg` / `memory` | 会话落库 + 长期记忆 | 81-82 |
| `tool_harness` | 工具安全治理（能力/审批/沙箱/预算） | 86-93 |
| `tool_registry` | 工具注册表 | 95-98 |
| `skill_registry` / `router` | 技能 + MCP 网关 + 工具召回 | 100-147 |
| `rag` | 知识库（Milvus） | 149-154 |

**第 96 行是最该记住的一行**：

```python
tool_registry.register_batch(tool_registry.get_default_tools())
```

`get_default_tools()` 里 `return [CalculatorTool(), DataQueryTool()]`，就是这一行把工具实例塞进了 `_tools` 字典 —— 后面 `get_tool("calculator")` 能取到东西，源头在这里。

---

## ③ 依赖注入：把零件塞进 Agent

```158:167:main.py
    react_agent = ReActAgent(
        llm_client=llm,
        tool_registry=tool_registry,
        rag_chain=rag,
        memory_manager=memory,
        skill_registry=skill_registry,
        router=router,
        tool_harness=tool_harness,
    )
    agent_graph = AgentGraph(react_agent)
```

**这 10 行不执行任何业务逻辑，只是「把螺丝拧上」。**

好处：任何零件传 `None` 就自动降级。这也是下面到处是 `try/except` 的原因。

---

## ④ 主干入口就这一行

```200:200:main.py
        result = agent_graph.run(query, session_id, user_id)
```

外面 `while True`（178 行）只是「你打字、退出」的壳。

**200 行是主干真正的入口**：一句话进去，一个 dict 出来，里面有：

| 字段 | 含义 |
|---|---|
| `final_answer` | 最终回答 |
| `iteration` | 推理了几轮 |
| `tool_call_logs` | 工具调用记录 |
| `active_skill` | 命中的技能 |
| `memory_compressed` | 历史记忆是否被压缩过 |

---

## ⑤ 贯穿全项目的写法：到处降级

这三处的 `except` 都是**把变量设成 None 然后继续跑**，而不是崩：

| 位置 | 失败后 | 退化成 |
|---|---|---|
| `main.py:92-93` | `tool_harness = None` | 无治理（接入 Harness 前的行为） |
| `main.py:145-147` | `router = None` | 全量工具模式（不按语义召回） |
| `main.py:153-154` | `rag = None` | 无知识库模式 |

> **这是这个项目的一条设计原则：新机制是增强，不是依赖。**

看懂这句话，后面一半的 `if xxx is None` 分支都不用再问。

---

## 一句话记住 `main.py`

> 造 6 个零件 → 塞进 `ReActAgent` → 包一层 `AgentGraph` → `agent_graph.run(问题)` 拿答案。命令分发和降级都是配菜。

**只要记 3 个行号：**

- `96` — 注册工具（工具从这里进 `_tools`）
- `158` — 注入依赖（零件装上 Agent）
- `200` — 主干入口（问题从这里进去）

---

## 下一站

`graph.py`：第 200 行 `agent_graph.run()` 进去之后，`prepare → reason ↔ act → answer` 这个循环是怎么转起来的。
