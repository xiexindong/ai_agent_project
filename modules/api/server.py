"""
FastAPI 后端服务模块
=====================
企业级 API 服务：对话接口、知识库管理、会话管理、工具日志审计。
所有接口统一鉴权、限流、多用户隔离。
"""
import os
import json
import uuid
import time
import asyncio
import threading
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from modules.common.logger import get_logger
from modules.common.context import new_task_id, set_request_context
from modules.api.auth import (
    get_current_user,
    RequirePermission,
    RequirePerm,
    UserLevel,
    AuthManager,
)
from modules.agent.graph import AgentGraph
from modules.agent.react_agent import ReActAgent
from modules.agent.llm import LLMClient
from modules.tools.tool_registry import ToolRegistry
from modules.memory.memory_manager import MemoryManager
from modules.memory.pg_manager import PGManager
from modules.multimodal.media import MultiModalInput, ImagePart
from modules.multiagent.harness import build_harness
from modules.multiagent.models import EvalCase, RunResult
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


class TeamRunRequest(BaseModel):
    """多 Agent 团队运行请求。images 传 data URL 或 http(s) 图片链接。"""

    query: str
    session_id: Optional[str] = None
    images: list[str] = []
    backend: str = "native"  # native（自研Harness）| crewai（角色扮演编排，需装可选依赖）


class AgentEvalRequest(BaseModel):
    """多 Agent 评测请求。cases 为空时用内置默认用例跑一次连通性。"""

    cases: list[dict] = []


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: str


class ReviewRequest(BaseModel):
    action: str
    question: str = ""
    standard_answer: str = ""
    note: str = ""


class AppContext:
    """应用全局上下文（单例组件）。"""
    llm: LLMClient = None
    memory: MemoryManager = None
    pg: PGManager = None
    rag: object = None
    tool_registry: ToolRegistry = None
    skill_registry: object = None
    agent_graph: AgentGraph = None
    harness: object = None
    crew_backend: object = None


def _reconcile_loop(interval_hours: int):
    """后台对账循环：睡 → 跑一次 → 有不一致就告警。

    为什么必须有这一层：跨库没有事务，MySQL 写失败是静默的（audit 里只
    warning 不抛），没人主动查就永远发现不了。定时对账是唯一的兜底。
    大厂用 MQ + 独立对账服务，阶段1 先用一个 daemon 线程顶上。
    """
    while True:
        time.sleep(interval_hours * 3600)
        try:
            from modules.enterprise.reconcile import reconcile

            result = reconcile(hours=max(interval_hours * 2, 24))
            if not result.get("ok") or result.get("pg_only") or result.get("mysql_only"):
                _logger.warning(f"定时对账发现不一致: {result}")
        except Exception as e:
            _logger.warning(f"定时对账异常: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动/停止钩子：拉起后台对账线程（RECONCILE_INTERVAL_HOURS=0 关闭）。"""
    interval = int(get_settings().reconcile_interval_hours)
    if interval > 0:
        threading.Thread(target=_reconcile_loop, args=(interval,), daemon=True).start()
        _logger.info(f"跨库对账后台任务已启动: 每 {interval} 小时一次")
    yield


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""
    app = FastAPI(title="企业级 AI Agent 智能助手", version="1.0.0", lifespan=lifespan)

    origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
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

        # 走工厂而不是直接 new RAGChain：RAG_BACKEND=llamaindex 时接管，
        # 库没装会自动降级回自研链路（降级在工厂里记 warning，这里只兜底异常）
        try:
            from modules.rag.llamaindex_adapter import build_rag
            self.ctx.rag = build_rag()
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

        # 多 Agent 团队：共用上面的 LLM 与工具注册器，团队组成走默认 4 人配置
        try:
            if get_settings().multiagent_enabled:
                self.ctx.harness = build_harness(
                    llm=self.ctx.llm, tool_registry=self.ctx.tool_registry
                )
                _logger.info("多 Agent 团队已装配: 4 人（主管/检索/分析/审核）")
        except Exception as e:
            _logger.warning(f"多 Agent 模块降级（/agent/* 接口不可用）: {e}")
            self.ctx.harness = None
        _logger.info("后端服务组件初始化完成")

    def _register_routes(self):
        """注册 API 路由。"""
        app = self.app
        ctx = self.ctx

        @app.get("/health")
        async def health():
            """真探活：逐项探测 PG / MySQL / Milvus，核心依赖不可用即 503。

            硬编码返回 ok 的探针是假健康——依赖全挂容器也不会重启，等于没有探针。
            """
            checks = {}
            try:
                ctx.pg._reconnect_if_needed()
                ctx.pg._cur.execute("SELECT 1")
                checks["postgres"] = "ok"
            except Exception as e:
                checks["postgres"] = f"down: {e}"

            if get_settings().rbac_enabled:
                try:
                    from modules.enterprise.mysql_client import get_mysql_ro_client
                    get_mysql_ro_client().query_one("SELECT 1")
                    checks["mysql"] = "ok"
                except Exception as e:
                    checks["mysql"] = f"down: {e}"
            else:
                checks["mysql"] = "disabled"

            checks["milvus"] = "ok" if ctx.rag else "down: RAG模块未启用"

            degraded = any(v != "ok" and v != "disabled" for v in checks.values())
            payload = {"status": "degraded" if degraded else "ok", "service": "ai-agent", "checks": checks}
            if checks["postgres"] != "ok":
                raise HTTPException(status_code=503, detail=payload)
            return payload

        @app.get("/me")
        async def me(user: dict = Depends(get_current_user)):
            """当前登录者的权限快照：排障时先问"我是谁、我有什么权限"。"""
            p = user.get("principal")
            return {
                "user": {k: v for k, v in user.items() if k != "principal"},
                "permissions": p.to_dict() if p else None,
            }

        @app.get("/admin/audit")
        async def audit_log(
            limit: int = 50,
            session_id: str = "",
            task_id: str = "",
            user: dict = Depends(RequirePerm("api:audit:*")),
        ):
            """审计日志：谁在什么时候对什么资源做了什么，deny 记录是重点。

            传 session_id 拉同一会话的全部审计；传 task_id 拉同一轮对话的全部审计
            （PG 的 chat_history / tool_call_log 落的是同一个号）。
            这两个号是跨库排障唯一的可靠串联键。
            """
            from modules.enterprise.mysql_client import get_mysql_ro_client
            try:
                db = get_mysql_ro_client()
                sql = (
                    "SELECT user_code, dept_code, session_id, task_id, action, resource, result, created_at "
                    "FROM sys_audit_log"
                )
                if task_id:
                    rows = db.query_all(
                        sql + " WHERE task_id=%s ORDER BY id DESC LIMIT %s", (task_id, limit)
                    )
                elif session_id:
                    rows = db.query_all(
                        sql + " WHERE session_id=%s ORDER BY id DESC LIMIT %s", (session_id, limit)
                    )
                else:
                    rows = db.query_all(sql + " ORDER BY id DESC LIMIT %s", (limit,))
                return {"total": len(rows), "items": rows}
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"审计库不可用: {e}")

        @app.get("/admin/bill")
        async def bill(user: dict = Depends(RequirePerm("api:admin:*"))):
            """按部门汇总 LLM token 消耗：成本治理与预算控制的数据源。"""
            from modules.enterprise.mysql_client import get_mysql_ro_client
            try:
                rows = get_mysql_ro_client().query_all(
                    "SELECT dept_code, COUNT(*) AS calls, SUM(total_tokens) AS tokens "
                    "FROM sys_llm_usage GROUP BY dept_code ORDER BY tokens DESC"
                )
                return {"items": rows}
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"用量库不可用: {e}")

        @app.get("/admin/offboard/preview")
        async def offboard_preview(user: dict = Depends(RequirePerm("api:admin:*"))):
            """离职数据对账：只看不改，报告哪些离职员工的会话仍未归档。"""
            from modules.enterprise import lifecycle
            result = lifecycle.preview()
            if not result.get("ok"):
                raise HTTPException(status_code=503, detail=result.get("error"))
            return result

        @app.post("/admin/offboard/run")
        async def offboard_run(
            retention_days: int = 0, user: dict = Depends(RequirePerm("api:admin:*"))
        ):
            """执行离职数据处置：离职员工会话归档 + 超保留期数据清理。"""
            from modules.enterprise import lifecycle
            result = lifecycle.run_once(retention_days or None)
            if not result.get("ok"):
                raise HTTPException(status_code=503, detail=result.get("errors") or result)
            return result

        @app.get("/admin/reconcile")
        async def cross_db_reconcile(
            hours: int = 24, user: dict = Depends(RequirePerm("api:admin:*"))
        ):
            """跨库对账：按 task_id 比对 MySQL 与 PG，列出单边缺失的号。

            pg_only（PG 有、MySQL 无）= 审计/用量丢失，是合规与账单风险；
            mysql_only（MySQL 有、PG 无）= 会话已清理，或 PG 写入失败。
            """
            from modules.enterprise.reconcile import reconcile

            result = reconcile(hours)
            if not result.get("ok"):
                raise HTTPException(status_code=503, detail=result.get("errors") or result)
            return result

        @app.post("/admin/archive/run")
        async def archive_run(
            retention_days: int = 0, user: dict = Depends(RequirePerm("api:admin:*"))
        ):
            """冷热分离：超过保留期的会话先导出归档文件，再从 PG 删除。"""
            from modules.memory.archiver import archive_expired

            result = archive_expired(retention_days or 0)
            if not result.get("ok"):
                raise HTTPException(status_code=503, detail=result.get("errors") or result)
            return result

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
                event: done         全部完成 {session_id, task_id}（task_id 是本轮的业务流水号，
                                    凭它能在 MySQL 审计/用量与 PG 会话之间来回查）
            """
            user_id = user["user_id"]
            principal = user.get("principal")
            is_new_session = not req.session_id
            session_id = req.session_id or str(uuid.uuid4())

            # ------------------------------------------------------------------
            # 【task_id 只在入口生成一次】它是本轮对话的业务流水号：
            #   PG 的 chat_history / tool_call_log 与 MySQL 的 sys_audit_log /
            #   sys_llm_usage 落的是同一个号，定时对账才能两边比对。
            #
            # 【为什么放进 ContextVar 而不是层层传参】
            #   Agent 跑在 asyncio.to_thread 的子线程里，context 会复制过去，
            #   落库处（pg_manager.save_history / audit.log_audit）自动取到；
            #   靠传参的话，中间漏一层就是空号，而且不报错、最难查。
            # ------------------------------------------------------------------
            task_id = new_task_id()
            set_request_context(task_id=task_id, session_id=session_id, user_code=user_id)

            if is_new_session:
                ctx.pg.create_session(session_id, user_id, title=req.query[:50])
            elif not AuthManager.check_session_owner(user_id, session_id, ctx.pg):
                raise HTTPException(status_code=403, detail="无权访问该会话")

            ctx.tool_registry.reset_session(session_id)
            ctx.llm.set_usage_context(
                user_code=user_id,
                dept_code=user.get("dept_code", ""),
                session_id=session_id,
                task_id=task_id,
            )

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
                    yield sse_event("done", {"session_id": session_id, "task_id": task_id})
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
                yield sse_event("done", {"session_id": session_id, "task_id": task_id})

            return StreamingResponse(
                event_generator(),

                media_type="text/event-stream",

                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @app.post("/agent/team/run", response_model=RunResult)
        async def team_run(req: TeamRunRequest, user: dict = Depends(get_current_user)):
            """多 Agent 团队协作：主管拆任务 → 并行执行 → 汇总 → 终审。

            【安全约束：图片不接受服务器本地路径】
                若允许传 "/etc/passwd" 这类路径，客户端就能把服务器任意文件
                读进模型上下文（路径穿越 + 信息泄露）。这里只放行
                data URL（客户端自己编码的）和 http(s) 链接（模型侧去取）。
            """
            if not ctx.harness:
                raise HTTPException(status_code=503, detail="多Agent模块未启用")
            # 后端选择：请求级 > 全局配置 > native。请求级能覆盖是为了 A/B 对比，
            # 但默认仍是 native —— 换编排引擎是行为变更，不该由客户端静默触发。
            backend = (req.backend or get_settings().multiagent_backend or "native").lower()
            runner = ctx.harness
            if backend == "crewai":
                if ctx.crew_backend is None:
                    from modules.multiagent.crewai_backend import CrewAIBackend
                    ctx.crew_backend = CrewAIBackend(llm=ctx.llm, tool_registry=ctx.tool_registry)
                runner = ctx.crew_backend
            parts = []
            for src in req.images or []:
                if not src.startswith(("data:", "http://", "https://")):
                    raise HTTPException(
                        status_code=400,
                        detail="图片只接受 data URL 或 http(s) 链接，不接受服务器本地路径",
                    )
                try:
                    parts.append(ImagePart(source=src))
                except ValidationError as e:
                    raise HTTPException(status_code=400, detail=f"图片非法: {e.errors()[0]['msg']}")
            mm = MultiModalInput(instruction=req.query, parts=parts) if parts else None
            session_id = req.session_id or f"team-{uuid.uuid4().hex[:8]}"
            return runner.run(req.query, mm=mm, session_id=session_id)

        @app.post("/agent/eval")
        async def agent_eval(
            req: AgentEvalRequest,
            user: dict = Depends(RequirePermission(UserLevel.ADMIN)),
        ):
            """多 Agent 回归评测（管理员）：关键词命中 + 完成率 + 速度。

            改了提示词或调了团队配置后跑一次，通过率掉了说明退化。
            cases 为空时跑内置连通性用例（只验链路，不验回答质量）。
            """
            if not ctx.harness:
                raise HTTPException(status_code=503, detail="多Agent模块未启用")
            cases = [EvalCase(**c) for c in req.cases] if req.cases else [
                EvalCase(name="默认连通性", query="用一句话说明你能做什么", expected_keywords=["可以"]),
            ]
            return ctx.harness.evaluate(cases).model_dump()

        @app.get("/agent/backends")
        async def agent_backends(user: dict = Depends(get_current_user)):
            """可选框架后端探活：当前配的哪个、库装没装。

            排障时第一件事就是分清"配了但没装"和"装了但没配"——
            前者表现为静默降级（不报错、结果不一样），最容易误判。
            """
            from modules.multiagent.crewai_backend import is_available as crewai_ok
            from modules.rag.llamaindex_adapter import is_available as li_ok

            s = get_settings()
            return {
                "rag_backend": s.rag_backend,
                "multiagent_backend": s.multiagent_backend,
                "available": {"llamaindex": li_ok(), "crewai": crewai_ok()},
            }

        @app.get("/history/{session_id}")
        async def get_history(
            session_id: str,
            limit: int = 20,
            user: dict = Depends(get_current_user),
        ):
            """获取会话历史（会话隔离）。"""
            if not AuthManager.check_session_owner(user["user_id"], session_id, ctx.pg):
                raise HTTPException(status_code=403, detail="无权访问该会话")
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

        @app.post("/reviews/{review_id}/review")
        async def submit_review(
            review_id: int,
            req: ReviewRequest,
            user: dict = Depends(RequirePermission(UserLevel.ADMIN)),
        ):
            """人工审核处置：approved / rejected，审核通过的标准答案反哺标准答案库（Few-shot 来源）。"""
            if req.action not in ("approved", "rejected"):
                raise HTTPException(status_code=400, detail="action 只能是 approved 或 rejected")
            if not ctx.pg.update_review_status(review_id, req.action, user["user_id"], req.note):
                raise HTTPException(status_code=404, detail="待审核项不存在或已处理")

            answer_id = None
            if req.action == "approved" and req.standard_answer.strip():
                answer_id = ctx.pg.add_standard_answer(
                    question=req.question.strip(),
                    answer=req.standard_answer.strip(),
                    source_review_id=review_id,
                )
            return {"review_id": review_id, "action": req.action, "standard_answer_id": answer_id}

    def run(self):
        """启动 API 服务。"""
        import uvicorn

        s = get_settings()
        _logger.info(f"API服务启动: {s.app_host}:{s.app_port}")
        uvicorn.run(self.app, host=s.app_host, port=s.app_port)
