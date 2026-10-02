# Harness 六大能力模块全景（对照学习文档）

> 图片里列出的 Harness 六个通用能力模块：**QueryEngine / Tool System / Permission System / Context Management / Memory / Extensibility**，本项目**六个全部都有实现**，并且已经在 CLI（main.py）和 API（server.py）两条主链路里接线运行。本文逐一对照每个模块的「职责 → 代码位置 → 核心数据结构 → 流程图 → JS 类比 → 输入输出对照」，最后把它们串成一次完整请求。

---

## 0. 一页总览：六大模块 ↔ 项目代码映射

| # | 能力模块 | 一句话职责 | 主要代码位置 | 核心数据结构 |
|---|---|---|---|---|
| 1 | **QueryEngine** 核心引擎 | 跑「思考→调工具→看结果」的推理循环 | [modules/agent/](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/agent) | `AgentState`、LangGraph 状态图 |
| 2 | **Tool System** 工具系统 | 一切工具的注册、调度、限流、重试 | [modules/tools/](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/tools) | `ToolRegistry` 的 4 本账 |
| 3 | **Permission System** 权限系统 | 把 API Key 解析成身份，并在各环节卡权限 | [modules/enterprise/](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/enterprise)、[harness/middleware.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/harness/middleware.py) | `Principal`、权限码集合 |
| 4 | **Context Management** 上下文管理 | 请求级链路号 + Prompt 级 token 预算裁剪 | [common/context.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/common/context.py)、[modules/prompt/](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/prompt) | `ContextVar`、`PromptBuilder` |
| 5 | **Memory** 记忆系统 | 短期原文 + 长期向量召回的双层记忆 | [modules/memory/](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/memory) | PG pgvector、摘要向量 |
| 6 | **Extensibility** 扩展系统 | MCP 远程工具 + Skill 技能包的按需扩展 | [modules/mcp/](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/mcp)、[modules/skills/](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/skills) | MCP 网关目录、`BaseSkill` |

**六大模块协作全景（一次请求的流向）**：

```
                        ┌──────────────────────────────────────────────┐
   用户提问 + API Key ──► │  ③ Permission：API Key → Principal（身份快照） │
                        └──────────────────────────────────────────────┘
                                          │ principal 贯穿全程
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ ⑥ Extensibility：MCP 网关提供远程工具目录                            │
   │     AgentRouter 语义召回 → 本轮工具名单（⑥ 与 ② 在这里握手）          │
   └─────────────────────────────────────────────────────────────────────┘
                                          │ 本轮名单
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ ① QueryEngine：LangGraph 编排 ReAct 循环（prepare→reason→act…）       │
   │     reason：④ PromptBuilder 拼 prompt（⑤记忆 + RAG + 权限 + 工具）    │
   │     act   ：② ToolRegistry 调度 → Guard→Retry→沙箱执行                │
   └─────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                          ⑤ Memory 落库（原文/摘要/向量）
```

---

## 1. QueryEngine —— 核心引擎

### 1.1 职责
驱动 Agent 的推理循环：每轮让模型做一次决策——**直接回答**还是**调用工具**；调完工具把结果喂回去，继续决策，直到模型收尾。

### 1.2 代码组成

| 文件 | 角色 |
|---|---|
| [react_agent.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/agent/react_agent.py) | `ReActAgent`：ReAct 业务引擎，prepare/reason/act/answer 的业务逻辑全在这 |
| [graph.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/agent/graph.py) | `AgentGraph`：把 ReAct 装进 LangGraph，6 节点 8 条边 |
| [state.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/agent/state.py) | `AgentState`（dataclass，业务用）/ `AgentStateDict`（TypedDict，LangGraph 用） |
| [llm.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/agent/llm.py) | `LLMClient`：封装模型调用、Function Calling、JSON 修复 |
| [planner.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/agent/planner.py) | `TaskPlanner`：Plan-and-Execute，复杂问题先拆步骤（默认关闭） |

### 1.3 LangGraph 图结构（核心流程图）

```
            ┌─────────┐  条件边            ┌──────┐
   入口 ───► │ prepare │ ───plan─────────► │ plan │ （可选节点）
            └─────────┘                   └──────┘
                │ answer                    │
                │ (注入拦截)                 │ 固定边
                ▼                           ▼
            ┌─────────┐  条件边   ┌─────┐  固定边  ┌────────┐
            │ answer  │ ◄──────── │act  │ ◄─────── │ reason │ ◄─┐
            └─────────┘           └─────┘          └────────┘   │
                │                      ▲                          │
                ▼                      └──────── ReAct 循环 ───────┘
            ┌──────┐
            │ save │ ──► END
            └──────┘
```

- **prepare**：输入清洗、注入检测、召回记忆/技能/RAG（只跑一次，靠 `prepared` 标记）
- **reason**：LLM 决策，产出 `need_tool_call / action / final_answer`
- **act**：执行工具，结果写回 `observation`，然后固定边回 reason（形成循环）
- **answer**：生成完整答案 + 一句话摘要
- **save**：落 PG（原文+摘要+向量+工具日志）

### 1.4 循环终止权（面试重点）

ReAct 最大的风险是「停不下来」，项目里有**五层刹车**，从早到晚依次生效：

| 刹车 | 触发条件 | 位置 |
|---|---|---|
| 预算强制收尾 | 墙钟超时 120s 或 token 打满 6000 | reason_step 开头 |
| 同名同参强制收敛 | 本次动作与历史某次成功调用完全一致 | reason_step 内 |
| 交替环检测 | A→B→A→B / A→B→C→A→B→C | `_detect_cycle` |
| 无进展检测 | 连续 3 次成功结果一模一样 | `_no_progress` |
| 轮次上限 | iteration ≥ 10，强制作答 | reason_step 末尾 |

另外 LangGraph 自身还有 `recursion_limit=30` 兜底（图内最大跳转步数）。

### 1.5 JS 类比

- **ReAct 循环** ≈ 你手写一个 `while (true)`：问模型 → 执行 → 再问模型。
- **LangGraph 改造** ≈ 从 Express 手写 `if (path===...)` 升级到 NestJS 的装饰器路由：节点是"工序"、边是"跳转规则"，流程可视化，还自带状态持久化。
- `AgentState`（dataclass）↔ `AgentStateDict`（TypedDict）的互转，类比 JS 里 class 实例 ↔ plain object 的序列化。

### 1.6 输入输出对照（用户问"3*7 等于多少"）

```
输入: query="3*7 等于多少", session_id="abc", user_id="cli-user"
轨迹: prepare → reason(要调calculator) → act(算出21)
      → reason(直接答) → answer("3*7 等于 21") → save → END
输出 dict: {"final_answer": "3*7 等于 21", "iteration": 2, "finished": True, ...}
```

---

## 2. Tool System —— 工具系统

### 2.1 职责
Agent 调用一切工具的**唯一入口**：本地工具（calculator/data_query）和 MCP 远程工具注册进来后，走同一套「校验 → 去重 → 限次 → 执行 → 过滤」流程。

### 2.2 代码组成

| 文件 | 角色 |
|---|---|
| [base_tool.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/tools/base_tool.py) | `BaseTool`（工具统一接口 name/run/to_openai_format）、`ToolResult`（success/result/error/meta） |
| [tool_registry.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/tools/tool_registry.py) | `ToolRegistry`：注册与调度核心 |
| [calculator.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/tools/calculator.py) / [data_query.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/tools/data_query.py) | 两个内置本地工具 |
| [concurrency.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/tools/concurrency.py) | `ToolConcurrencyGuard`（信号量限流）、`ToolInstancePool`（有状态工具实例池，预留） |
| [retry.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/tools/retry.py) | `ToolRetryPolicy`（瞬时故障指数退避重试） |
| [tool_rag.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/tools/tool_rag.py) | `ToolRAGIndex`：工具语义索引，供路由召回 |

### 2.3 ToolRegistry 的四本账（核心数据结构）

```python
_tools         # 工具名 → 工具实例           （注册表本体）
_call_records   # 会话ID → {"工具名:参数指纹"}  （同参去重）
_call_counts    # 工具名 → 累计调用次数         （单工具限次，默认上限 5）
_result_cache   # 会话ID → {去重键: 成功结果}   （重复调用直接回传上次答案）
```

【为什么要结果缓存】同参重复调用时，只回"拒绝重复执行"会让模型以为工具坏了、继续换说法重试（实测曾跑满 10 轮、拖到 13 秒）。把上次的成功答案原样返回，模型才有依据立刻收尾。

### 2.4 执行期三层包裹（Guard → Retry → runner）

这是已经接好线的真实执行链（见 [harness.py 的 execute()](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/harness/harness.py#L334-L370)）：

```
ToolRegistry.execute_tool
   └─ ToolHarness.execute
       └─ Guard.execute(工具名)       ① 信号量：同名工具同时最多 N 个（默认 4），超出排队
           └─ Retry.run(工具名)        ② 失败且属瞬时故障（超时/连接断开）→ 指数退避重试
               └─ runner.run()         ③ 真正执行：进程内 / 子进程 / 容器 / 远程沙箱
```

重试包在闸门内：一趟在途的所有重试只占一个名额，不会因重试挤占排队者。

### 2.5 JS 类比

- `ToolRegistry` ≈ 前端的**统一 API 请求层**（一个 axios 实例 + 拦截器），业务代码不直接 fetch。
- `Guard` ≈ 请求**并发控制**（p-limit 这类库），`Retry` ≈ axios-retry 拦截器。
- 去重缓存 ≈ React Query 按 queryKey 缓存结果，同 key 重复请求直接返回旧数据。

### 2.6 输入输出对照

```
calculator(3*7) 首次          → success=True,  result="21"
calculator(3*7) 同会话再次调用 → success=True,  result="21"（命中缓存，duration_ms=0）
"calc"（模型幻觉的名字）        → success=False, error="工具不存在: calc"
calculator 第 6 次调用（上限5）→ success=False, error="工具 calculator 调用次数超限(5)，疑似死循环"
```

---

## 3. Permission System —— 权限系统

### 3.1 职责
在 HTTP 入口把**一条 API Key 解析成 Principal**（身份+角色+权限+数据域+知识库可见范围），之后全链路的权限判定都在内存里完成；并在参数、RAG、数据查询各环节做隔离。

### 3.2 代码组成

| 文件 | 角色 |
|---|---|
| [rbac_service.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/enterprise/rbac_service.py) | `RBACService`（全局单例）：Key 解析、工具过滤、RAG 过滤表达式 |
| [models.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/enterprise/models.py) | `Principal`：一次请求的权限快照，含 `can()/can_any()/scope_for()` |
| [masking.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/enterprise/masking.py) | `mask_record`：按 mask_level 对字段打码（手机号/薪资） |
| [audit.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/enterprise/audit.py) | `log_audit`：鉴权、工具调用、拦截全部留痕 |
| [harness/middleware.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/harness/middleware.py) | `PermissionMiddleware`、`ScopeMiddleware`：执行前再卡一道 |

### 3.3 API Key → Principal 的解析流程（4 次查询）

```
① sys_api_key → sys_user       身份、在职状态、Key 是否吊销/过期
② sys_user_role → sys_role     角色列表 + 数据域(data_scope) + 脱敏强度(mask_level)
③ 四表联查                      权限码集合（人→角色→角色权限→权限点）
④ sys_kb / sys_kb_grant         可见知识库（public ∪ 部门/角色/个人授权）
```

- **缓存**：结果按「Key 的 sha256 前 16 位」缓存 60s（`RBAC_CACHE_TTL`）。明文 Key 不进内存、不进日志。
- **fail-closed**：启用 RBAC 但 MySQL 挂了时默认**拒绝请求**（`RBAC_FAIL_CLOSED=True`），而不是降级放行 demo Key 造成权限裸奔。
- 多角色合并规则：数据域取**最宽**（self < dept < all），脱敏取**最强**。

### 3.4 权限码与通配（Principal.can 的判定口径）

```
tool:calculator     精确匹配
tool:mcp:*          前缀通配 → 命中 tool:mcp:web_search 等一切 MCP 工具
data:orders:all     订单数据可见全公司
kb:finance:read     财务知识库可读
空权限集             一律拒绝（默认拒绝，不默认放行）
```

### 3.5 权限落地的三道防线（同一件事在三个位置卡）

```
第一道（源头）  工具 schema 生成时过滤 → 模型看不到未授权工具，就调不到
第二道（执行前） PermissionMiddleware → 模型靠幻觉拼出工具名也拦
第三道（数据层） ScopeMiddleware + 注入 _dept_code/_scopes/_mask_level
                → 财务的人查不到 HR 部门数据，手机号自动打码
```

### 3.6 JS 类比

- `Principal` ≈ JWT 解码后挂在 `req.user` 上的用户信息对象，入口解析一次、后面全内存读。
- 三道防线 ≈ 前端按钮隐藏（源头）+ 路由守卫（执行前）+ 后端接口再次鉴权（数据层），三层都做才稳。
- 权限码前缀通配 ≈ 用 `perm.startsWith("tool:mcp:")` 做通配判断。

### 3.7 输入输出对照

```
合法 Key            → AuthResult(ok=True, principal=<张三 U001 normal>, reason="ok")
空 Key              → AuthResult(ok=False, reason="no_key")
库里查不到的 Key     → AuthResult(ok=False, reason="invalid",  message="API Key 无效或已失效")
吊销的 Key          → AuthResult(ok=False, reason="disabled")
过期的 Key          → AuthResult(ok=False, reason="expired")
MySQL 挂且fail-closed→ AuthResult(ok=False, reason="mysql_unavailable")
```

---

## 4. Context Management —— 上下文管理

项目里的"上下文管理"分**两个层面**，别混在一起：

### 4.1 层面一：请求级上下文（跨库串联号）

代码：[modules/common/context.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/common/context.py)

**背景痛点**：项目是双库架构——MySQL 存权限/审计，PG 存会话/记忆，两库之间没有外键、没有跨库事务。唯一能把两边记录串起来的是**业务流水号 task_id**。

```
三个 ContextVar 槽位：
  _task_id      业务流水号，如 T-20261002-3f9c1a2b（MySQL+PG 共用）
  _session_id   会话号，对应 PG 的 chat_history
  _user_code    工号，审计与用量归集
```

【为什么用 ContextVar 不用全局变量】Agent 在 `/chat` 里是 `asyncio.to_thread` 丢到线程池跑的：ContextVar 的值会随 context 复制进子线程，且并发请求互不串号；全局变量会被并发请求互相覆盖。

### 4.2 层面二：Prompt 级上下文（拼装 + token 预算裁剪）

代码：[modules/prompt/builder.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/prompt/builder.py)

`PromptBuilder` 把 prompt 分四层拼装，四层互不耦合：

```
角色层   role           模型"是谁"
策略层   strategy       "怎么问"（zero_shot / few_shot / cot / react）
上下文层 **context      "拿什么答"（RAG、记忆、权限、工具清单）
输出层   output_format  "答成什么样"（text/json/markdown）
```

**贪心装包裁剪**（防上下文膨胀的核心算法）：按优先级一块一块往 token 预算（默认 3000）里塞——塞得下整块保留；塞不下截断；剩余空间太少整块丢弃并填占位符。

```
优先级从高到低（预算不够时从最低的开始丢）：
  权限边界 > 工具清单 > 观察结果 > 技能SOP > 知识库 > 历史记忆 > 工具历史
  ↑ 权限是硬约束（丢了会越权）排最前；工具历史只是"可能有用"排最后
```

另外 system 段（角色+策略）每轮都一样，只渲染一次缓存起来，还能命中服务端**前缀缓存**省钱。

### 4.3 JS 类比

- `ContextVar` ≈ Node.js 的 **AsyncLocalStorage**：一次请求内全局可读，并发请求互不干扰，不用靠函数参数层层透传。
- `PromptBuilder` 四层 ≈ 前端组件：`role` 是布局壳、`strategy` 是行为、`context` 是 props 数据、`output` 是渲染格式。
- token 预算裁剪 ≈ 响应式数据太多时按优先级丢弃次要部分，类似虚拟列表只渲染可见区。

### 4.4 输入输出对照

```
set_request_context(task_id="T-20261002-3f9c1a2b", session_id="s-1", user_code="U001")
get_task_id()   → "T-20261002-3f9c1a2b"
snapshot()      → {"task_id": "T-20261002-3f9c1a2b", "session_id": "s-1", "user_code": "U001"}

build_record(query, **context) → {
  "messages": [{role:"system",...}, {role:"user",...}],
  "prompt_tokens": 812,
  "trimmed": ["rag_context: 截断 1200 → 486 token"]   # 空列表 = 没超预算
}
```

---

## 5. Memory —— 记忆系统

### 5.1 职责
让助手"记得住事"：短期靠最近几轮原文保证精准，长期靠摘要向量做跨会话语义召回。

代码：[modules/memory/](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/memory)

| 文件 | 角色 |
|---|---|
| [memory_manager.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/memory/memory_manager.py) | `MemoryManager`：双层记忆的对外入口（save_turn / build_context） |
| [pg_manager.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/memory/pg_manager.py) | PG + pgvector 读写：会话、向量检索、工具流水、标准答案 |
| [summary_generator.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/memory/summary_generator.py) | 摘要生成与多摘要合并（压缩用） |
| [embedding_service.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/memory/embedding_service.py) | 文本 → 1024 维向量（智谱 embedding-2） |
| [archiver.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/memory/archiver.py) | 冷热分离：90 天前的会话先导出 JSONL 再删 |

### 5.2 双层记忆架构图

```
                    ┌────────────────────────────────┐
   用户提问 ───────► │ MemoryManager.build_context    │
                    └────────────────────────────────┘
                       │                      │
              问题转向量 ▼                      ▼ 按会话取
        ┌────────────────────┐      ┌────────────────────┐
        │ 长期记忆（跨会话）   │      │ 短期记忆（本会话）   │
        │ pgvector 余弦召回   │      │ 最近 3 轮完整原文    │
        │ 返回历史摘要 top-5  │      │ 精准上下文           │
        └────────────────────┘      └────────────────────┘
                       └──────────┬───────────┘
                                  ▼
                    【历史记忆摘要】+【近期对话】拼进 prompt
```

### 5.3 写入治理（save_turn 流水线，三步）

```
① 入库筛选  寒暄（你好/谢谢）、过短（<4字）、纯标点 → 不入库，零 LLM 成本
② 落库      原文+摘要+摘要向量+问句向量+工具日志一次写入
③ 冲突处理  同用户相似度 ≥ 0.90 的旧记忆置为 superseded（"住址:上海"被"住址:北京"取代）
            顺序固定"先插新、再废旧"：插入失败也不会让旧记忆凭空消失
```

### 5.4 读取治理（召回流水线）

```
向量召回 top-5
  → 相似度阈值过滤（默认 ≥ 0.65，低于丢弃，防"问天气召回到股票"）
  → 关键词二次校验（字面完全不重合 → 标记低置信，触发人工审核）
  → 时效衰减重排：有效分 = 相似度 × 0.5^(天数/180)，半年前的记忆权重减半
  → 超 token 上限时多摘要合并压缩
```

### 5.5 冷热归档

PG 只留最近 90 天热会话；更老的先导出成 `archives/日期/会话号.jsonl`（不可变文件，接口形状对齐对象存储，将来可平滑换 OSS），再从 PG 删除——体积可控，审计证据也在。

### 5.6 JS 类比

- **短期记忆** ≈ 页面内存里的 `useState`：当前会话精准，但刷新就没。
- **长期记忆** ≈ 服务端数据库 + 向量检索：跨会话持久，按语义查询。
- 冲突取代 ≈ 用户资料表的 update：新资料覆盖旧资料，只是这里用向量相似度判定"是不是同一件事"。
- 冷热归档 ≈ 把冷数据从 MySQL 搬到 S3 Glacier。

### 5.7 输入输出对照

```
save_turn(query="你好", ...)            → None（寒暄被筛选，不落库）
save_turn(query="差旅报销标准是多少")    → 42（新行 id）
build_context(session, user, "3*7")     → {
  "long_term_memory":  [{summary:"3*5=15", similarity:0.91}],
  "short_term_memory": [{query:"3*5?", answer:"3*5=15"}],
  "context_text": "【历史记忆摘要】…\n【近期对话】…",
  "compressed": False
}
```

---

## 6. Extensibility —— 扩展系统

扩展性在本项目里由**两套互补机制**承担：MCP（接远程工具）+ Skill（封装能力包）。

### 6.1 MCP 技能网关

代码：[modules/mcp/gateway.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/mcp/gateway.py)

把多个 MCP Server 收敛成一个统一入口，只暴露三件事：

```
catalog()            有哪些远程工具（名字+描述+参数表），不连进程也能看目录
ensure_registered()  只有被选中要用的工具才连 server、注册进 ToolRegistry（懒加载）
call_tool()          调用方不需要知道工具在哪个子进程/远程机器上
```

**支持两种传输**（`MCPServerConfig`）：

| 传输 | 工具跑在哪 | 配置 |
|---|---|---|
| stdio | 本机子进程（进程隔离） | `command: ["python", "-m", "..."]`，也能接 Node 写的 server |
| http | 远程机器上的常驻服务（跨机器、跨语言，MCP 本意） | `endpoint` + Bearer `token` |

**核心设计：远程工具不直连 LLM**

```
改造前：启动就全量注册 → 每轮把 100 份工具 schema 全发给模型（技能膨胀，选错率上升）
改造后：
  MCP Server → 网关纳管建目录 → Tool-RAG 语义索引（只存语义，不进 prompt）
  用户提问 → 语义召回 top-k → 按需连 server+注册 → LLM 只看到这 k 个
远程工具数量增长，不再影响单轮 prompt 长度
```

### 6.2 Skill 技能体系

代码：[modules/skills/base_skill.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/skills/base_skill.py)、[skill_registry.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/skills/skill_registry.py)

**Skill 与 Tool 的分工**（关键概念）：

```
Tool = 原子能力：一个函数做一件事（calculator 算一个表达式），LLM 用 Function Calling 调
Skill = 能力包：  触发词 + SOP 作业流程 + 推荐工具集合
                 ReAct 开始前按意图匹配，把专属流程注入 prompt，
                 引导 LLM 按标准步骤组合调用多个 Tool
```

子类只声明类属性即可定义一个技能：`name / display_name / description / trigger_keywords / instructions(SOP) / bound_tools / priority`。

**三级路由**（技能怎么被选中）：

| 层级 | 判定方式 | 分数含义 |
|---|---|---|
| L1 关键词路由 | 触发词命中，按长度分档加权（短语0.6/双字词0.4/单字0.15） | 命中权重累加 |
| L2 向量语义路由 | query 与技能 routing_text 的余弦相似度 | 语义相似 |
| L3 LLM 兜底路由 | LLM 自评置信度 | confidence |

### 6.3 JS 类比

- **MCP 网关** ≈ 服务注册发现（npm 生态里的微服务网关）：服务可以本机起子进程，也可以远程调用。
- **按需注册** ≈ 动态 `import()`：用到了才加载这个模块，不用不占资源。
- **Skill** ≈ 一个 npm 包：自带触发条件、使用说明（SOP）、依赖清单（bound_tools）。
- Tool vs Skill ≈ 单个函数 vs 一个封装好流程的 SDK。

### 6.4 输入输出对照

```
MCP 网关（size=2）:
  catalog()                    → [web_search, get_current_time]（目录，不连进程）
  未被召回的工具                → 不起进程、不进 prompt
  ensure_registered(web_search)→ 拉起对应 server，注册进 ToolRegistry

Skill:
  query="报销流程怎么走" + 技能触发词含"报销" → SkillMatch(skill=报销咨询, score=0.6, route="keyword")
  format_context(match)        → "【激活技能】报销咨询\n作业流程(SOP):\n1.确认身份…"
```

---

## 7. 端到端：一次请求如何穿过六大模块

以用户问 **"帮我看看我上个月有哪些订单，再算一下总金额"**（带 API Key）为例：

```
【③ Permission】
 1. RBACService.resolve(apiKey)：查 4 次库 → Principal(李四, data_scope=self, mask_level=1)
    结果缓存 60s；审计记一条 auth/allow

【⑥ Extensibility ↔ ② Tool】
 2. AgentRouter.route(query)：
    Skill 三级路由命中"订单查询"技能（L1关键词，score=0.6）
    Tool-RAG 语义召回 → 名单 [data_query, calculator]（技能 bound_tools 强约束带上）
 3. 网关按需 ensure_registered：若用到远程工具此刻才连 server

【① QueryEngine：prepare 节点】
 4. 注入检测 → 输入清洗
 5. ⑤ 记忆召回：问题转向量 → 长期摘要召回 + 短期 3 轮原文
 6. RAG 召回：按 Principal.kb_codes 过滤知识库

【① QueryEngine：reason 节点（第 1 轮）】
 7. ④ PromptBuilder.build_record：8 个上下文槽位按 token 预算裁剪，拼 [system, user]
 8. LLM 决策 → need_tool_call=True, action={data_query, {data_type:"orders", month:"last"}}

【① QueryEngine：act 节点】
 9. ② ToolRegistry.execute_tool
    → ③ PermissionMiddleware/ScopeMiddleware 再校验
    → Guard(拿名额) → Retry → InProcessRunner
    → 注入服务端私有参数 _user_code/_scopes/_mask_level
    → data_query 返回该用户订单（敏感字段打码）
10. 结果回写 observation，工具历史记账

【reason 第 2 轮 → act】
11. LLM 决策调 calculator 算总金额 → 同样链路执行 → observation=总金额

【reason 第 3 轮 → answer → save】
12. LLM 判定信息齐全 → 生成最终答案 + 一句话摘要
13. ⑤ save_turn：原文+摘要+摘要向量+工具日志落 PG，agent_state 快照存盘（支持断点续跑）
14. 返回最终答案给前端
```

---

## 8. 推荐阅读路径

按「先主干、后细节」的顺序读代码，配合本目录已有的学习笔记：

```
第一步  主干流程
   main.py（装配入口）→ agent/graph.py（图全貌）
   配套笔记：主干01-main装配与入口.md、全流程执行链路.md

第二步  工具系统
   tools/tool_registry.py → harness/harness.py
   配套笔记：面试01-工具池并发控制.md、五道拦截器.md、harness.md

第三步  权限与上下文
   enterprise/rbac_service.py → harness/middleware.py → common/context.py
   配套笔记：面试03-工具调用错误的根源与治理.md

第四步  记忆系统
   memory/memory_manager.py → pg_manager.py → archiver.py
   配套笔记：面试05、10、11（记忆治理系列）

第五步  扩展系统
   mcp/gateway.py → skills/base_skill.py → agent/router.py
   配套笔记：06-MCP接入全流程.md、08-Skill与Tool的区别.md、15-三级路由详解.md
```

---

## 附：六大模块的「一句话记忆法」

| 模块 | 一句话 |
|---|---|
| QueryEngine | 负责"想"：跑 Reason-Action-Observation 循环，五层刹车保证停得下来 |
| Tool System | 负责"做"：工具唯一入口，注册/去重/限次 + 执行期 Guard→Retry→沙箱 |
| Permission | 负责"能不能"：Key 解析成 Principal，源头/执行前/数据层三道防线 |
| Context | 负责"串和裁"：task_id 串双库，PromptBuilder 按 token 预算贪心裁剪 |
| Memory | 负责"记"：短期原文保精准、长期向量跨会话，写入和读取双向治理 |
| Extensibility | 负责"扩"：MCP 远程工具按需加载，Skill 用 SOP 编排多个工具 |
