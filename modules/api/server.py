"""
FastAPI 后端服务模块
=====================
企业级 API 服务：对话接口、知识库管理、会话管理、工具日志审计。
所有接口统一鉴权、限流、多用户隔离。
"""
import os
import json
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from modules.common.logger import get_logger
from modules.api.auth import get_current_user, RequirePermission, UserLevel
from modules.agent.graph import AgentGraph
from modules.agent.react_agent import ReActAgent
from modules.agent.llm import LLMClient
from modules.tools.tool_registry import ToolRegistry
from modules.memory.memory_manager import MemoryManager
from modules.memory.pg_manager import PGManager
from config import get_settings

_logger = get_logger("api")


# ---- 请求/响应模型 ----
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    summary: str
    active_skill: str = ""
    iterations: int
    tool_calls: list


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: str


class AppContext:
    """应用全局上下文（单例组件）。"""
    llm: LLMClient = None
    memory: MemoryManager = None
    pg: PGManager = None
    rag: object = None
    tool_registry: ToolRegistry = None
    skill_registry: object = None
    agent_graph: AgentGraph = None


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""
    # ━━━ 创建 FastAPI 应用实例 ━━━
    # FastAPI() 内部干了 5 件事：① 继承 Starlette 装 ASGI 引擎
    # ② 初始化路由表 self.router ③ 装中间件栈 self.user_middleware=[]
    # ④ 准备 OpenAPI 文档生成器 ⑤ 注册 /docs /redoc 等默认路由
    #
    # title 和 version 不是给代码用的，是给 /docs 文档页用的：
    #   浏览器开 http://localhost:8000/docs，页面顶部显示：
    #     企业级 AI Agent 智能助手  1.0.0   ← title + version
    # 类比 Express：const app = express() —— 拿到空应用，等你往里加路由/中间件
    app = FastAPI(title="企业级 AI Agent 智能助手", version="1.0.0")

    # ━━━ 装跨域中间件（CORS）━━━
    # 【为什么要这个】浏览器同源策略：前端在 localhost:5500 请求 localhost:8000
    # 会被浏览器拦截（跨域），后端必须明确"允许来源"才放行
    #
    # 【参数拆解】
    #   CORSMiddleware       → FastAPI 自带的跨域处理中间件类
    #   allow_origins=["*"] → 允许哪些域名来访，"*" = 全放行
    #   allow_methods=["*"] → 允许 HTTP 方法，"*" = GET/POST/PUT/DELETE 全放行
    #   allow_headers=["*"] → 允许请求头，"*" = X-API-Key/Content-Type 全放行
    #
    # 【实际效果】
    #   ✗ 不加：前端 fetch('/chat') 浏览器报 CORS 错，请求被拦在浏览器侧
    #   ✓ 加：后端响应头带 Access-Control-Allow-Origin: *，浏览器放行
    #
    # 【生产环境】allow_origins=["*"] 是开发期方便，生产要换成具体域名：
    #   allow_origins=["http://localhost:5500", "https://your-frontend.com"]
    #
    # 类比 Express：app.use(cors())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 挂载静态文件（前端 jQuery 界面）
    static_dir = Path(__file__).resolve().parent.parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        _logger.info(f"静态文件已挂载: {static_dir}")
    return app


class APIServer:
    """API 服务封装：组件初始化 + 路由注册。"""

    def __init__(self):
        self.app = create_app()
        self.ctx = AppContext()
        self._init_components()
        self._register_routes()

    def _init_components(self):
        """初始化全局组件（LLM、记忆、工具、Agent图）。"""
        s = get_settings()
        _logger.info("初始化后端服务组件...")

        self.ctx.llm = LLMClient()
        self.ctx.pg = PGManager()
        self.ctx.memory = MemoryManager(llm_client=self.ctx.llm)

        # 工具注册
        self.ctx.tool_registry = ToolRegistry()
        self.ctx.tool_registry.register_batch(self.ctx.tool_registry.get_default_tools())

        # 技能注册（SOP + 工具绑定）
        from modules.skills import SkillRegistry, get_default_skills
        self.ctx.skill_registry = SkillRegistry()
        self.ctx.skill_registry.register_batch(get_default_skills())
        _logger.info(f"技能已加载: {len(self.ctx.skill_registry.list_skills())} 个")

        # MCP 远程工具接入（可选）
        try:
            from modules.mcp.mcp_client import MCPClient, MCPToolAdapter
            mcp_client = MCPClient(
                server_command=["python", "-m", "modules.mcp.mcp_server"],
                name="enterprise-mcp",
            )
            mcp_client.connect()
            tools = mcp_client.discover_tools()
            for tool_info in tools:
                adapter = MCPToolAdapter(mcp_client, tool_info)
                self.ctx.tool_registry.register(adapter)
            self.ctx.mcp_client = mcp_client
        except Exception as e:
            _logger.warning(f"MCP Server 接入降级（本地工具仍可用）: {e}")
            self.ctx.mcp_client = None

        # RAG（Milvus 不可用时降级）
        try:
            from modules.rag.rag_chain import RAGChain
            self.ctx.rag = RAGChain()
        except Exception as e:
            _logger.warning(f"RAG 模块降级（Milvus不可用）: {e}")
            self.ctx.rag = None

        # Agent 图
        react_agent = ReActAgent(
            llm_client=self.ctx.llm,
            tool_registry=self.ctx.tool_registry,
            rag_chain=self.ctx.rag,
            memory_manager=self.ctx.memory,
            skill_registry=self.ctx.skill_registry,
        )
        self.ctx.agent_graph = AgentGraph(react_agent)
        _logger.info("后端服务组件初始化完成")

    def _register_routes(self):
        """注册 API 路由。"""
        app = self.app
        ctx = self.ctx

        @app.get("/health")
        async def health():
            return {"status": "ok", "service": "ai-agent"}

        @app.get("/")
        async def index():
            """前端首页：返回 jQuery 界面。"""
            static_dir = Path(__file__).resolve().parent.parent.parent / "static"
            index_file = static_dir / "index.html"
            if index_file.is_file():
                return FileResponse(str(index_file))
            raise HTTPException(status_code=404, detail="前端界面未找到")

        @app.post("/chat")
        async def chat(
            # req = 请求体参数：FastAPI 看到类型是 Pydantic 模型（class ChatRequest(BaseModel)，见上方第32行），
            # 会自动把前端发来的 JSON body（app.js 第344行 {query, session_id}）反序列化成 ChatRequest 对象，
            # 并按类型注解做校验：query 必填且必须是字符串，session_id 可省略（默认 None）。
            # 字段缺失/类型错误时 FastAPI 直接返回 422，不会进入函数体。
            # 类比 Express 的 body-parser：const { query, session_id } = req.body，但多了自动校验
            req: ChatRequest,
            # Depends(...) = FastAPI 依赖注入：执行 chat() 之前先调用 get_current_user()，
            # 完成「读 X-API-Key → 验 Key 鉴权 → 滑动窗口限流」三步（见 auth.py 第93行），
            # 校验通过后返回用户信息 dict（含 user_id/name/level）；
            # 校验失败会直接抛 HTTP 异常并中断请求——401 没带 Key / 403 Key 无效 / 429 超频，
            # 根本不会进入下面的函数体。类比 Express 的鉴权中间件：router.post('/chat', authMiddleware, handler)
            user: dict = Depends(get_current_user),
        ):
            """
            对话接口（SSE 流式）：鉴权 → 会话隔离 → Agent推理 → 逐段推送事件。

            前端用 fetch + ReadableStream 读取，事件协议（text/event-stream）：
                event: status       思考状态提示
                event: tool_start   工具调用开始 {name, arguments}
                event: tool_end     工具调用完成 {name, result, success}
                event: answer_start 答案开始 {iterations}
                event: answer_chunk 答案分片 {content}（前端拼成打字机效果）
                event: answer_end   答案结束 {summary}
                event: done         全部完成 {session_id}（前端用它确认真实会话ID）
            """
            user_id = user["user_id"]
            is_new_session = not req.session_id
            session_id = req.session_id or str(uuid.uuid4())
            breakpoint()  # 【调试断点】/chat鉴权通过：可看 user_id / req.query / session_id

            # 创建会话（新会话）；已有会话做归属校验（防越权访问他人会话）
            if is_new_session:
                ctx.pg.create_session(session_id, user_id, title=req.query[:50])
            elif not ctx.pg.get_session_state(session_id, user_id):
                raise HTTPException(status_code=403, detail="无权访问该会话")

            # 重置工具调用记录（防死循环的去重/计数归零）
            ctx.tool_registry.reset_session(session_id)

            def sse_event(event: str, data: dict) -> str:
                """按 SSE 协议拼一个事件帧：event/data 两行，事件间以空行分隔。"""
                return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

            async def event_generator():
                yield sse_event("status", {"message": "Agent 正在思考...", "phase": "thinking"})
                try:
                    # Agent 推理是同步阻塞调用（内部多次请求 LLM），丢到线程池避免卡住事件循环
                    breakpoint()  # 【调试断点】Agent推理即将开始：可看 req.query / session_id
                    result = await asyncio.to_thread(
                        ctx.agent_graph.run, req.query, session_id, user_id
                    )
                    breakpoint()  # 【调试断点】Agent推理完成：可看 result（final_answer / tool_call_logs / iteration）
                except Exception as e:
                    _logger.error(f"Agent推理异常: {e}")
                    yield sse_event("answer_start", {"iterations": 0})
                    yield sse_event("answer_chunk", {"content": f"服务内部错误：{e}"})
                    yield sse_event("answer_end", {"summary": ""})
                    yield sse_event("done", {"session_id": session_id})
                    return

                # 工具调用事件（Agent 已整体执行完，这里把日志按顺序补推给前端展示）
                for tc in result.get("tool_call_logs", []):
                    yield sse_event("tool_start", {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", {}),
                    })
                    yield sse_event("tool_end", {
                        "name": tc.get("name", ""),
                        "result": str(tc.get("result", "")),
                        "success": tc.get("success", True),
                    })

                answer = result.get("final_answer", "") or "（无回答）"
                yield sse_event("answer_start", {"iterations": result.get("iteration", 0)})
                # 答案切片推送（模拟打字机；真实逐 token 流式需 LLMClient 支持 stream=True）
                for i in range(0, len(answer), 24):
                    yield sse_event("answer_chunk", {"content": answer[i:i + 24]})
                    await asyncio.sleep(0.02)

                yield sse_event("answer_end", {"summary": result.get("llm_summary", "")})
                # 回传真实 session_id：前端首轮用临时ID占位，收到后替换为后端真实ID
                yield sse_event("done", {"session_id": session_id})

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @app.get("/history/{session_id}")
        async def get_history(
            session_id: str,
            limit: int = 20,
            user: dict = Depends(get_current_user),
        ):
            """获取会话历史（会话隔离）。"""
            rows = ctx.pg.get_short_term_history(session_id, user["user_id"], limit)
            return {"session_id": session_id, "history": rows}

        @app.post("/knowledge/ingest")
        async def ingest_document(
            file: UploadFile = File(...),
            user: dict = Depends(RequirePermission(UserLevel.ADMIN)),
        ):
            """知识库文档入库（管理员权限）。"""
            if not ctx.rag:
                raise HTTPException(status_code=503, detail="RAG模块未启用")
            # 保存临时文件
            import tempfile, os
            tmp_path = os.path.join(tempfile.gettempdir(), file.filename)
            with open(tmp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            count = ctx.rag.ingest_document(tmp_path)
            os.remove(tmp_path)
            return {"status": "success", "chunks": count, "file": file.filename}

        @app.get("/knowledge/stats")
        async def knowledge_stats(user: dict = Depends(get_current_user)):
            """知识库统计信息。"""
            if ctx.rag:
                return ctx.rag.get_knowledge_stats()
            return {"status": "RAG模块未启用"}

        @app.get("/skills")
        async def list_skills(user: dict = Depends(get_current_user)):
            """获取已注册技能清单及其绑定的函数工具。"""
            return {
                "total": len(ctx.skill_registry.list_skills()),
                "skills": [s.to_dict() for s in ctx.skill_registry.list_skills()],
            }

        @app.get("/tools/logs")
        async def tool_logs(
            session_id: Optional[str] = None,
            limit: int = 50,
            user: dict = Depends(RequirePermission(UserLevel.ADVANCED)),
        ):
            """工具调用审计日志（高级用户+权限）。"""
            return {"status": "audit", "session_id": session_id, "limit": limit}

        @app.get("/reviews/pending")
        async def pending_reviews(
            user: dict = Depends(RequirePermission(UserLevel.ADMIN)),
        ):
            """待人工审核列表（管理员权限）。"""
            return {"reviews": ctx.pg.get_pending_reviews()}

    def run(self):
        """启动 API 服务。"""
        # ━━━ 真正启动 HTTP 服务的方法 ━━━
        # 前面 __init__ / _init_components / _register_routes 三步只是"造车 + 装货"：
        #   ① FastAPI() 造应用对象（车） ② 装配 6 大组件（货） ③ 注册 7 个路由（装货上车）
        # 但车还停在车库里——FastAPI 自己不会启动服务（没有 listen 方法）。
        # 这一行 import uvicorn 才是真正"找司机来开车"的步骤。
        #
        # 【FastAPI vs uvicorn 分工】
        #     FastAPI = 应用层：定义路由、处理请求、生成文档  → 类比"装好货的车"
        #     uvicorn  = 服务器层：开端口、收 socket、转 ASGI  → 类比"司机点火开车"
        #     FastAPI 把网络层外包给 uvicorn（职责分离设计），
        #     所以 FastAPI 对象本身不会启动服务，必须 uvicorn.run(app, ...) 来跑。
        #
        # 【对照 Express】
        #     Express：app.listen(8000)              ← 车和司机合一，一步启动
        #     FastAPI：uvicorn.run(app, port=8000)   ← 车和司机分离，两步启动
        import uvicorn

        # get_settings() = 读 .env 配置单例（端口/主机/密钥都从这里拿）
        # s.app_host / s.app_port 来自 .env 的 APP_HOST / APP_PORT 配置项
        s = get_settings()
        _logger.info(f"API服务启动: {s.app_host}:{s.app_port}")
        # ━━━ 点火启动：让 uvicorn 在 host:port 开 socket 监听 ━━━
        # 调用后进程一直阻塞在 uvicorn 事件循环里，收到 HTTP 请求 → 转给 self.app（FastAPI）
        # → FastAPI 路由匹配 → 调对应处理函数（如 @app.post("/chat") 的 chat()）
        # 直到 Ctrl+C 才退出。
        uvicorn.run(self.app, host=s.app_host, port=s.app_port)
