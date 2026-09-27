"""
前端预览服务器（Mock 模式，基于 Starlette）
=============================================
用于快速启动并验证 jQuery 前端界面，不依赖 PostgreSQL / Milvus / DeepSeek / FastAPI。
所有 API 返回 Mock 数据，方便纯前端调试。零额外依赖（仅需 starlette + uvicorn）。

启动方式：
    python demo_server.py
访问：
    http://localhost:8000
"""
# 【什么】Mock 预览服务的依赖
# 【作用】Starlette 提供路由与静态文件、StreamingResponse 负责 SSE 流式输出
import os
import json
import uuid
import asyncio
import logging
from pathlib import Path
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse, FileResponse, PlainTextResponse, StreamingResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")   # 监听地址，默认对外全开
APP_PORT = int(os.environ.get("APP_PORT", "8000")) # 端口，默认 8000

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
_logger = logging.getLogger("web-demo")

STATIC_DIR = Path(__file__).resolve().parent / "static"   # 前端静态资源目录

# 【什么】演示用的 Key → 用户映射
# 【作用】 替代真实 RBAC：三把 Key 对应三个等级，前端切换身份时用它鉴权
MOCK_USERS = {
    "sk-user-normal-001": {"user_id": "U001", "name": "张三", "level": "normal"},      # 普通用户
    "sk-user-advanced-002": {"user_id": "U002", "name": "李四", "level": "advanced"},  # 高级用户
    "sk-admin-003": {"user_id": "U003", "name": "管理员", "level": "admin"},            # 管理员
}

MOCK_KB_STATS = {"total_chunks": 1280, "collection": "knowledge_base"}   # 固定的假知识库统计


# 【什么】统一鉴权：读 X-API-Key 请求头
# 【作用】 返回 (用户, 错误响应) 二元组；调用方只需判断 err 是否为 None
def _auth(request):
    """简单鉴权：从 Header 读取 X-API-Key。"""
    api_key = request.headers.get("x-api-key", "")
    if not api_key:
        return None, JSONResponse({"detail": "缺少 X-API-Key"}, status_code=401)   # 401：没带 Key
    user = MOCK_USERS.get(api_key)
    if not user:
        return None, JSONResponse({"detail": "API Key 无效"}, status_code=403)     # 403：带了但不认识
    return user, None


# 【什么】首页路由：返回静态 index.html
# 【作用】 让 http://localhost:8000 直接打开前端界面；文件缺失时给 404 文本
async def index(request):
    """前端首页。"""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return PlainTextResponse("前端界面未找到", status_code=404)


# 【什么】健康检查路由
# 【作用】 给前端/探针一个不需要鉴权的存活检测点
async def health(request):
    return JSONResponse({"status": "ok", "service": "ai-agent-demo"})


# 【什么】知识库统计路由
# 【作用】 返回写死的假统计，让前端统计卡片有数据可渲染
async def kb_stats(request):
    user, err = _auth(request)
    if err:
        return err
    return JSONResponse(MOCK_KB_STATS)


# 【什么】入库路由（Mock）
# 【作用】 收上传文件但只回一个随机分片数，用于演示前端上传流程是否通顺
async def ingest(request):
    user, err = _auth(request)
    if err:
        return err
    form = await request.form()
    file = form.get("file")
    filename = file.filename if file else "unknown"   # 没取到文件时兜底叫 unknown
    import random
    return JSONResponse({
        "status": "success",
        "chunks": random.randint(20, 80),   # 随机 20~80 片，纯演示用
        "file": filename,
    })


# 【什么】对话路由：用 SSE 把「思考 → 工具 → 回答」分阶段推给前端
# 【作用】 模拟真实 /chat 的流式体验，前端按 event 类型渲染不同区块
async def chat(request):
    """流式对话接口（SSE）。"""
    user, err = _auth(request)
    if err:
        return err
    body = await request.json()
    session_id = body.get("session_id") or str(uuid.uuid4())   # 前端没传会话号就现生成一个
    query = body.get("query", "")

    answer, tool_calls, iterations = _generate_mock_response(query)

    # 【什么】SSE 事件生成器
    # 【作用】 按阶段 yield 事件，每步 sleep 一下，让前端看得到「正在思考/正在调用」的过程感
    async def event_generator():
        """SSE 事件生成器：分阶段流式输出。"""
        yield _sse_event("status", {"message": "Agent 正在思考...", "phase": "thinking"})
        await asyncio.sleep(0.8)

        if tool_calls:
            for tc in tool_calls:
                yield _sse_event("tool_start", {"name": tc["name"], "arguments": tc["arguments"]})   # 工具开始
                await asyncio.sleep(0.5)
                yield _sse_event("tool_end", {"name": tc["name"], "result": tc["result"], "success": tc["success"]})   # 工具结束
                await asyncio.sleep(0.3)

        yield _sse_event("answer_start", {"iterations": iterations})
        chars = list(answer)
        step = 2   # 每次推 2 个字，中文逐字推出更有打字机效果
        for i in range(0, len(chars), step):
            chunk = "".join(chars[i:i + step])
            yield _sse_event("answer_chunk", {"content": chunk})
            delay = 0.02 if i % 10 != 0 else 0.04   # 每 10 个字符多停一下，节奏更像真人
            await asyncio.sleep(delay)

        yield _sse_event("answer_end", {"summary": query[:30]})   # 摘要取提问前 30 字
        yield _sse_event("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",   # SSE 标准 MIME
        headers={
            "Cache-Control": "no-cache",        # 禁止缓存，否则流会被中间层攒着不发
            "Connection": "keep-alive",         # 保持长连接
            "X-Accel-Buffering": "no",          # 关掉 Nginx 缓冲，否则前端收不到分片
        },
    )


# 【什么】把一个事件名与数据拼成 SSE 报文
# 【作用】 统一格式：event 行 + data 行 + 空行结尾，缺一不可，否则前端收不到
def _sse_event(event: str, data: dict) -> str:
    """构造 SSE 消息格式。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# 【什么】历史记录路由
# 【作用】 返回空列表占位，让前端历史面板不报错（预览模式不存数据）
async def history(request):
    user, err = _auth(request)
    if err:
        return err
    session_id = request.path_params["session_id"]   # 路径参数：/history/{session_id}
    return JSONResponse({"session_id": session_id, "history": []})


# 【什么】工具日志路由
# 【作用】 返回空的审计日志列表，占位用
async def tool_logs(request):
    user, err = _auth(request)
    if err:
        return err
    return JSONResponse({"status": "audit", "logs": []})


# 【什么】待复核列表路由
# 【作用】 返回空的复核列表，占位用
async def pending_reviews(request):
    user, err = _auth(request)
    if err:
        return err
    return JSONResponse({"reviews": []})


# 【什么】按提问关键词挑一种假回答
# 【作用】 让前端四种渲染效果（算式/业务数据/MCP 搜索/通用回答）都能被演示到
def _generate_mock_response(query: str):
    """根据查询内容生成不同的 Mock 回答，演示前端渲染效果。"""
    if any(op in query for op in ["+", "-", "*", "/", "计算", "sqrt", "sin", "cos"]):   # 算式类：演示 calculator 工具
        answer = (
            "这是一个数学计算任务。\n\n"
            "## 计算过程\n"
            f"你询问的是: **{query}**\n\n"
            "调用 `calculator` 工具计算结果：\n"
            '```json\n{"expression": "' + query + '", "result": 42}\n```\n\n'
            "根据表达式解析，最终结果为 42。\n\n"
            f"【摘要】计算 {query} 的结果为 42"
        )
        tool_calls = [{
            "name": "calculator",
            "arguments": {"expression": query},
            "result": "42",
            "success": True,
            "duration_ms": 8,
        }]
        return answer, tool_calls, 2

    if any(kw in query for kw in ["用户", "订单", "库存", "查询", "order", "user", "sku"]):   # 业务类：演示 data_query 工具
        answer = (
            "已为你查询到相关业务数据。\n\n"
            "## 查询结果\n"
            "调用 `data_query` 工具获取数据：\n"
            "- 用户ID: U001\n- 姓名: 张三\n- 部门: 技术部\n- 等级: 高级\n\n"
            "> 数据来源：企业内部数据库（脱敏后展示）\n\n"
            f"【摘要】查询 {query[:20]} 返回 1 条记录"
        )
        tool_calls = [{
            "name": "data_query",
            "arguments": {"data_type": "users", "record_id": "U001"},
            "result": "{'id': 'U001', 'name': '张三', 'dept': '技术部'}",
            "success": True,
            "duration_ms": 35,
        }]
        return answer, tool_calls, 2

    if any(kw in query for kw in ["搜索", "search", "最新", "新闻", "联网"]):   # 搜索类：演示 MCP 远程工具 web_search
        answer = (
            "通过网络搜索获取最新信息。\n\n"
            "## 搜索结果\n"
            "调用 MCP 远程工具 `web_search`：\n"
            f"1. **搜索词**: {query}\n"
            "2. **结果摘要**: 这是模拟的网络搜索结果，展示 MCP 工具调度能力。\n"
            "3. **数据来源**: 互联网（MCP远程工具）\n\n"
            f"【摘要】MCP搜索 {query[:20]} 完成"
        )
        tool_calls = [{
            "name": "web_search",
            "arguments": {"query": query},
            "result": "模拟搜索结果：互联网最新信息...",
            "success": True,
            "duration_ms": 1200,
        }]
        return answer, tool_calls, 2

    answer = (
        "已基于知识库和历史记忆为你解答。\n\n"
        "## 回答\n"
        f"你询问的是：**{query}**\n\n"
        "这是模拟的 Agent 回答。在实际部署中，这里会：\n"
        "1. 召回 **Milvus** 知识库相关文档\n"
        "2. 检索 **pgvector** 历史对话摘要\n"
        "3. 通过 **LangGraph ReAct** 推理\n"
        "4. 按需调用本地/MCP工具\n"
        "5. 生成完整答案+摘要+向量入库\n\n"
        "### 架构亮点\n"
        "- 双层向量存储（Milvus + PG/pgvector）\n"
        "- ReAct 循环 + Function Calling\n"
        "- MCP 标准化工具协议\n"
        "- 上下文动态压缩\n\n"
        f"【摘要】回答 {query[:20]} 的相关问题"
    )
    return answer, [], 1


# 【什么】组装 Starlette 应用：挂路由 + 静态目录 + CORS
# 【作用】 对外只暴露这一个工厂函数，main.py 的 web-demo 命令就是调它
def create_demo_app() -> Starlette:
    """创建 Mock 模式 Starlette 应用（前端预览用）。"""
    # 显式写成 list 而不是让它被推断成 list[Route]：下面还要往里 append Mount（静态目录挂载）
    routes: list = [
        Route("/", index),                                          # 首页
        Route("/health", health),                                   # 健康检查（免鉴权）
        Route("/knowledge/stats", kb_stats),                        # 知识库统计
        Route("/knowledge/ingest", ingest, methods=["POST"]),       # 入库（只收 POST）
        Route("/chat", chat, methods=["POST"]),                     # 对话（SSE 流式）
        Route("/history/{session_id}", history),                    # 历史记录
        Route("/tools/logs", tool_logs),                            # 工具日志
        Route("/reviews/pending", pending_reviews),                 # 待复核列表
    ]
    if STATIC_DIR.is_dir():
        routes.append(Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"))   # 静态资源
        _logger.info(f"静态文件目录: {STATIC_DIR}")

    middleware = [
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),   # 全放开 CORS，方便本地起前端
    ]
    return Starlette(routes=routes, middleware=middleware)


# 【什么】启动预览服务并打印演示 Key
# 【作用】 把三把 demo Key 打到控制台，前端直接复制就能切换身份
def run_demo():
    """启动前端预览服务。"""
    import uvicorn
    app = create_demo_app()
    print("\n" + "=" * 60)
    print("  企业级 AI Agent 前端预览服务（Mock 模式）")
    print("  访问地址: http://localhost:" + str(APP_PORT))
    print("  演示 API Key:")
    print("    sk-user-normal-001   (普通用户)")
    print("    sk-user-advanced-002 (高级用户)")
    print("    sk-admin-003         (管理员)")
    print("=" * 60 + "\n")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)


if __name__ == "__main__":
    run_demo()
