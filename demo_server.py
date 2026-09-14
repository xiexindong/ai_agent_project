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

# ---- 配置（自包含，不依赖 config.py，避免引入 loguru/pydantic-dotenv）----
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
_logger = logging.getLogger("web-demo")

# ---- 静态文件目录 ----
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ---- Mock 用户与 API Key ----
MOCK_USERS = {
    "sk-user-normal-001": {"user_id": "U001", "name": "张三", "level": "normal"},
    "sk-user-advanced-002": {"user_id": "U002", "name": "李四", "level": "advanced"},
    "sk-admin-003": {"user_id": "U003", "name": "管理员", "level": "admin"},
}

MOCK_KB_STATS = {"total_chunks": 1280, "collection": "knowledge_base"}


def _auth(request):
    """简单鉴权：从 Header 读取 X-API-Key。"""
    api_key = request.headers.get("x-api-key", "")
    if not api_key:
        return None, JSONResponse({"detail": "缺少 X-API-Key"}, status_code=401)
    user = MOCK_USERS.get(api_key)
    if not user:
        return None, JSONResponse({"detail": "API Key 无效"}, status_code=403)
    return user, None


async def index(request):
    """前端首页。"""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return PlainTextResponse("前端界面未找到", status_code=404)


async def health(request):
    return JSONResponse({"status": "ok", "service": "ai-agent-demo"})


async def kb_stats(request):
    user, err = _auth(request)
    if err:
        return err
    return JSONResponse(MOCK_KB_STATS)


async def ingest(request):
    user, err = _auth(request)
    if err:
        return err
    # 解析 multipart 文件上传
    form = await request.form()
    file = form.get("file")
    filename = file.filename if file else "unknown"
    import random
    return JSONResponse({
        "status": "success",
        "chunks": random.randint(20, 80),
        "file": filename,
    })


async def chat(request):
    """流式对话接口（SSE）。"""
    user, err = _auth(request)
    if err:
        return err
    body = await request.json()
    session_id = body.get("session_id") or str(uuid.uuid4())
    query = body.get("query", "")

    answer, tool_calls, iterations = _generate_mock_response(query)

    async def event_generator():
        """SSE 事件生成器：分阶段流式输出。"""
        # 阶段1：思考中
        yield _sse_event("status", {"message": "Agent 正在思考...", "phase": "thinking"})
        await asyncio.sleep(0.8)

        # 阶段2：工具调用（如果有）
        if tool_calls:
            for tc in tool_calls:
                yield _sse_event("tool_start", {"name": tc["name"], "arguments": tc["arguments"]})
                await asyncio.sleep(0.5)
                yield _sse_event("tool_end", {"name": tc["name"], "result": tc["result"], "success": tc["success"]})
                await asyncio.sleep(0.3)

        # 阶段3：逐字流式输出答案（打字机效果）
        yield _sse_event("answer_start", {"iterations": iterations})
        chars = list(answer)
        step = 2  # 每次输出2个字符
        for i in range(0, len(chars), step):
            chunk = "".join(chars[i:i + step])
            yield _sse_event("answer_chunk", {"content": chunk})
            # 不同字符速度略有差异，更自然
            delay = 0.02 if i % 10 != 0 else 0.04
            await asyncio.sleep(delay)

        # 阶段4：完成
        yield _sse_event("answer_end", {"summary": query[:30]})
        yield _sse_event("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event: str, data: dict) -> str:
    """构造 SSE 消息格式。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def history(request):
    user, err = _auth(request)
    if err:
        return err
    session_id = request.path_params["session_id"]
    return JSONResponse({"session_id": session_id, "history": []})


async def tool_logs(request):
    user, err = _auth(request)
    if err:
        return err
    return JSONResponse({"status": "audit", "logs": []})


async def pending_reviews(request):
    user, err = _auth(request)
    if err:
        return err
    return JSONResponse({"reviews": []})


def _generate_mock_response(query: str):
    """根据查询内容生成不同的 Mock 回答，演示前端渲染效果。"""
    # 数学计算 → 触发计算器工具
    if any(op in query for op in ["+", "-", "*", "/", "计算", "sqrt", "sin", "cos"]):
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

    # 数据查询 → 触发数据查询工具
    if any(kw in query for kw in ["用户", "订单", "库存", "查询", "order", "user", "sku"]):
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

    # 网络搜索 → 触发 MCP 工具
    if any(kw in query for kw in ["搜索", "search", "最新", "新闻", "联网"]):
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

    # 默认：知识库问答
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


def create_demo_app() -> Starlette:
    """创建 Mock 模式 Starlette 应用（前端预览用）。"""
    routes = [
        Route("/", index),
        Route("/health", health),
        Route("/knowledge/stats", kb_stats),
        Route("/knowledge/ingest", ingest, methods=["POST"]),
        Route("/chat", chat, methods=["POST"]),
        Route("/history/{session_id}", history),
        Route("/tools/logs", tool_logs),
        Route("/reviews/pending", pending_reviews),
    ]
    # 挂载静态文件
    if STATIC_DIR.is_dir():
        routes.append(Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"))
        _logger.info(f"静态文件目录: {STATIC_DIR}")

    middleware = [
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
    ]
    return Starlette(routes=routes, middleware=middleware)


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
