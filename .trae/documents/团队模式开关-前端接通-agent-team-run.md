# 前端"团队模式"开关：接通 /agent/team/run

## Context（为什么做）

当前前端聊天框只调 `POST /chat`（单 Agent LangGraph 链路），多 Agent 团队入口 `POST /agent/team/run` 只能用 curl 调。用户要求在前端加一个"团队模式"开关：开启后聊天请求发到多 Agent 接口，前端可视化展示"主管拆任务 → 并行执行 → 汇总终审"全过程。

关键事实（已探明）：
- 前端只有一份：`static/index.html` + `static/js/app.js`，真后端 [server.py#L188-L193](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/api/server.py#L188-L193) 和 Mock 服务 demo_server.py 都托管同一份
- 发消息函数 `sendMessage()`：[app.js#L306-L435](file:///e:/workspace/AI/AI-project/ai_agent_project/static/js/app.js#L306-L435)，现用 fetch POST `/chat` + SSE 流式渲染
- 认证：请求头 `X-API-Key`（`get_current_user` 已支持，/agent/team/run 无需改动）
- `TeamRunRequest` 请求体：`query`（必填）、`session_id`（可选）、`images`、`backend`
- `RunResult` 响应（[models.py#L272-L307](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/multiagent/models.py#L272-L307)）：`plan`（thought/subtasks 含 assignee、depends_on）、`outputs[]`（agent/role/success/content/duration_ms/steps）、`final_answer`、`review`、`status`、`duration_ms`、`replan_count`、`error`
- 团队接口是**同步 JSON**（无 SSE），团队模式渲染走"等待提示 → 一次性渲染结果"

## 改动内容

### 1. static/index.html — 加开关控件

输入区（L105-113 input-wrapper 内、textarea 之前）加一个开关：

```html
<label class="team-toggle" title="开启后由多Agent团队协作：主管拆任务 → 并行执行 → 汇总终审">
    <input type="checkbox" id="teamMode"> 团队模式
</label>
```

在 `<style>`（index.html 顶部内联样式或 static/css）补 `.team-toggle` 小样式（竖排居中、勾选框高亮），不动现有布局。

### 2. static/js/app.js — sendMessage() 分流 + 团队渲染

**分流**：`sendMessage()` 开头（L312 `appendMessage` 之前）判断开关：

```js
const teamMode = $("#teamMode").prop("checked");
```

- 普通模式：走现有 `/chat` SSE 逻辑，**一行不动**
- 团队模式：调用新函数 `sendTeamRun(query, aiMsg)`（新建 aiMsg 消息气泡后进入）

**新函数 `sendTeamRun(query, aiMsg)`**：
1. `setLoading("团队协作中：主管拆任务 → 并行执行 → 汇总终审...")`
2. `fetch(state.apiBase + "/agent/team/run", { method: "POST", headers: {"Content-Type": "application/json", "X-API-Key": state.apiKey}, body: JSON.stringify({query, session_id: state.currentSessionId || null}) })`
3. 非 2xx 时按状态码提示：503="多Agent模块未启用"、401="API Key 无效"、其余透出 detail
4. 成功后调 `renderTeamResult(result, $content, $meta)`：

```
📋 主管思路: <plan.thought>
子任务:
  t1 → researcher_1: 查Q3/Q4营收 （可立即开始）
  t2 → analyst_1: 计算环比 （依赖 [t1]）
执行产出:
  ✅ researcher_1(researcher) 步数=2 耗时=3200ms
     <content 前 120 字>
  ...
────────────
<final_answer 用现有 renderMarkdown() 渲染>
```

5. `$meta` 渲染：`状态: {status} | 耗时: {duration_ms}ms | 产出: {outputs.length} 个`（有 replan_count > 0 时加"重规划 N 次"）
6. 会话号沿用 done 事件同款逻辑：响应里 `query`/无 session 字段 → 用返回 JSON 中没有 session_id 时保持 `tmp-` 会话即可（与现状一致，不多做）

**样式**：产出区用现有 `renderToolCallsBox` 的盒子风格（浅色底圆角块），不新造组件体系。

### 3. demo_server.py — 加 Mock 团队端点（保持 web-demo 可用）

现有 Mock `/chat` 在 [demo_server.py#L101-L148](file:///e:/workspace/AI/AI-project/ai_agent_project/demo_server.py#L101-L148)。旁边加：

- `POST /agent/team/run`：读 `session_id/query`，返回固定 JSON（照 RunResult 字段拼一份演示数据：2 个 subtask 带 depends_on、2 个 outputs 成功、final_answer、status="completed"、duration_ms≈1800、review 通过）
- 目的：web-demo 页面里开关打开也能演示，不 404

### 4. 不改的文件

- `modules/api/server.py` — `/agent/team/run` 端点已存在，不动
- `modules/multiagent/**` — 模型与编排逻辑不动
- `/chat` 现有逻辑与 7 种 SSE 事件处理 — 一行不动

## 验证

1. `python main.py serve` → 浏览器开 `http://localhost:8000/` → 输入 `sk-admin-003` 连接
2. **不开开关**：问"你好" → 走原 /chat 流式，行为与改动前一致
3. **开开关**：问"查一下Q3和Q4的营收，然后计算环比增长率" → 等待提示出现 → 渲染出主管思路、2 个子任务（含依赖标注）、执行产出列表、最终答案、meta 行
4. 开关开着问"你好" → PlanResult 直接答形态：subtasks 为空，页面只显示 final_answer（验证直接答路径不报错）
5. `python main.py web-demo` → 开开关发任意消息 → 显示 Mock 团队结果
6. `python -m py_compile demo_server.py` 编译通过

## 风险提示

- 团队模式最长 120s（RunBudget 总超时），fetch 无超时限制会一直等，页面有 loading 提示即可接受；浏览器本身不会主动断
- 调试断点暂停期间墙钟继续走，budget 提前耗尽可能返回"预算耗尽"错误——这是已知现象，非本次改动引入
