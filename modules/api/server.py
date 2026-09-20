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
from modules.api.auth import get_current_user, RequirePermission, RequirePerm, UserLevel
from modules.agent.graph import AgentGraph
from modules.agent.react_agent import ReActAgent
from modules.agent.llm import LLMClient
from modules.tools.tool_registry import ToolRegistry
from modules.memory.memory_manager import MemoryManager
from modules.memory.pg_manager import PGManager
from config import get_settings

_logger = get_logger("api")


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
    app = FastAPI(title="企业级 AI Agent 智能助手", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
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

        self.ctx.tool_registry = ToolRegistry()
        self.ctx.tool_registry.register_batch(self.ctx.tool_registry.get_default_tools())

        from modules.skills import SkillRegistry, get_default_skills
        self.ctx.skill_registry = SkillRegistry()
        self.ctx.skill_registry.register_batch(get_default_skills())
        _logger.info(f"技能已加载: {len(self.ctx.skill_registry.list_skills())} 个")

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

        try:
            from modules.rag.rag_chain import RAGChain
            self.ctx.rag = RAGChain()
        except Exception as e:
            _logger.warning(f"RAG 模块降级（Milvus不可用）: {e}")
            self.ctx.rag = None

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

        @app.get("/me")
        async def me(user: dict = Depends(get_current_user)):
            """当前登录者的权限快照：排障时先问"我是谁、我有什么权限"。"""
            p = user.get("principal")
            return {
                "user": {k: v for k, v in user.items() if k != "principal"},
                "permissions": p.to_dict() if p else None,
            }

        @app.get("/admin/audit")
        async def audit_log(limit: int = 50, user: dict = Depends(RequirePerm("api:audit:*"))):
            """审计日志：谁在什么时候对什么资源做了什么，deny 记录是重点。"""
            from modules.enterprise.mysql_client import get_mysql_client
            try:
                rows = get_mysql_client().query_all(
                    "SELECT user_code, dept_code, action, resource, result, created_at "
                    "FROM sys_audit_log ORDER BY id DESC LIMIT %s",
                    (limit,),
                )
                return {"total": len(rows), "items": rows}
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"审计库不可用: {e}")

        @app.get("/admin/bill")
        async def bill(user: dict = Depends(RequirePerm("api:admin:*"))):
            """按部门汇总 LLM token 消耗：成本治理与预算控制的数据源。"""
            from modules.enterprise.mysql_client import get_mysql_client
            try:
                rows = get_mysql_client().query_all(
                    "SELECT dept_code, COUNT(*) AS calls, SUM(total_tokens) AS tokens "
                    "FROM sys_llm_usage GROUP BY dept_code ORDER BY tokens DESC"
                )
                return {"items": rows}
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"用量库不可用: {e}")

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
            req: ChatRequest,
            user: dict = Depends(dependency=get_current_user),
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
            principal = user.get("principal")
            is_new_session = not req.session_id
            session_id = req.session_id or str(uuid.uuid4())

            if is_new_session:
                ctx.pg.create_session(session_id, user_id, title=req.query[:50])
            elif not ctx.pg.get_session_state(session_id, user_id):
                raise HTTPException(status_code=403, detail="无权访问该会话")

            ctx.tool_registry.reset_session(session_id)

            def sse_event(event: str, data: dict) -> str:
                """按 SSE 协议拼一个事件帧：event/data 两行，事件间以空行分隔。"""
                return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

            async def event_generator():
                yield sse_event("status", {"message": "Agent 正在思考...", "phase": "thinking"})
                try:
                    result = await asyncio.to_thread(
                        ctx.agent_graph.run, req.query, session_id, user_id, principal
                    )
                except Exception as e:
                    _logger.error(f"Agent推理异常: {e}")
                    yield sse_event("answer_start", {"iterations": 0})
                    yield sse_event("answer_chunk", {"content": f"服务内部错误：{e}"})
                    yield sse_event("answer_end", {"summary": ""})
                    yield sse_event("done", {"session_id": session_id})
                    return

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
                for i in range(0, len(answer), 24):
                    yield sse_event("answer_chunk", {"content": answer[i:i + 24]})
                    await asyncio.sleep(0.02)

                yield sse_event("answer_end", {"summary": result.get("llm_summary", "")})
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
        import uvicorn

        s = get_settings()
        _logger.info(f"API服务启动: {s.app_host}:{s.app_port}")
        uvicorn.run(self.app, host=s.app_host, port=s.app_port)
