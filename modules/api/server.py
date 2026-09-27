"""
FastAPI 后端服务模块
=====================
企业级 API 服务：对话接口、知识库管理、会话管理、工具日志审计。
所有接口统一鉴权、限流、多用户隔离。
"""
import os
import sys
import json
import uuid
import time
import asyncio
import threading
from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
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

# 【什么】只在类型检查阶段执行的导入（TYPE_CHECKING 恒为 False，运行时不导入）
# 【作用】给 AppContext 字段标注真实类型，避免退化成 object 导致调用方法被判错；
#        运行时不导入，防止拉起 pymilvus / crewai 等重型可选依赖，也避免循环导入
if TYPE_CHECKING:
    from modules.rag.rag_chain import RAGChain
    from modules.rag.llamaindex_adapter import LlamaIndexRAG
    from modules.skills.skill_registry import SkillRegistry
    from modules.multiagent.harness import AgentHarness
    from modules.multiagent.crewai_backend import CrewAIBackend
    from modules.mcp.gateway import MCPGateway
    from modules.mcp.mcp_client import MCPClient

_logger = get_logger("api")


# 【什么】POST /chat 的请求体
# 【作用】带 session_id 表示续聊，不带则由服务端新开一个会话
class ChatRequest(BaseModel):
    session_id: Optional[str] = None   # 已有会话号；不传 = 新会话
    query: str                         # 用户这一轮的输入


# 【什么】对话结果的响应体
# 【作用】把一轮对话的产出字段固定下来，前端按名字取值
class ChatResponse(BaseModel):
    session_id: str           # 会话号
    answer: str               # 最终答案
    summary: str              # 本轮摘要，供长期记忆使用
    active_skill: str = ""    # 命中的技能名；未命中是空串
    iterations: int           # ReAct 迭代轮数
    tool_calls: list          # 本轮工具调用记录


# 【什么】POST /agent/team/run 的请求体
# 【作用】描述要交给团队完成的任务、可选图片与编排后端
class TeamRunRequest(BaseModel):
    """多 Agent 团队运行请求。images 传 data URL 或 http(s) 图片链接。"""

    query: str                         # 交给团队的任务
    session_id: Optional[str] = None   # 会话号；不传时生成 team-xxxxxxxx
    images: list[str] = []             # 图片来源；只收 data URL 或 http(s) 链接
    backend: str = "native"            # native（自研Harness）| crewai（角色扮演编排，需装可选依赖）


# 【什么】POST /agent/eval 的请求体
# 【作用】cases 为空时只跑内置连通性用例，验链路不验质量
class AgentEvalRequest(BaseModel):
    """多 Agent 评测请求。cases 为空时用内置默认用例跑一次连通性。"""

    cases: list[dict] = []   # 评测用例；空列表 = 只跑内置默认用例


# 【什么】会话信息的响应体
# 【作用】新建会话后回给前端会话号、归属用户与创建时间
class SessionResponse(BaseModel):
    session_id: str   # 会话号
    user_id: str      # 归属用户工号
    created_at: str   # 创建时间


# 【什么】POST /reviews/{review_id}/review 的请求体
# 【作用】承载审核动作；通过时可用标准答案反哺标准答案库
class ReviewRequest(BaseModel):
    action: str                 # 审核动作：approved / rejected
    question: str = ""          # 对应问题；与标准答案一起入库
    standard_answer: str = ""   # 人工标准答案，作为 Few-shot 来源
    note: str = ""              # 审核备注


# 【什么】全局组件的持有者（进程级单例）
# 【作用】路由通过 ctx.xxx 拿到 LLM / 记忆 / 工具 / Agent 图等共享组件
class AppContext:
    """应用全局上下文（单例组件）。"""
    # 【类型说明】字段一律 Optional[...] + None：这些组件在 lifespan 里才装配，
    #   声明成裸类型却赋 None 会让类型检查器报赋值不匹配；声明成 object 则调用方法时全报错
    llm: Optional[LLMClient] = None             # 单例 LLM 客户端
    memory: Optional[MemoryManager] = None      # 单例记忆管理器
    pg: Optional[PGManager] = None              # 单例 Postgres 管理器
    # RAG_BACKEND=llamaindex 时工厂返回 LlamaIndexRAG，两种实现接口一致，故注解成联合类型
    rag: Optional[Union["RAGChain", "LlamaIndexRAG"]] = None  # 单例 RAG 链路
    tool_registry: Optional[ToolRegistry] = None  # 单例工具注册表
    tool_harness: Optional[Any] = None            # 工具运行时（能力/校验/审批/沙箱/预算）
    skill_registry: Optional["SkillRegistry"] = None  # 单例技能注册表
    gateway: Optional["MCPGateway"] = None      # MCP 技能网关（多 server 纳管，远程技能不直连 LLM）
    mcp_client: Optional["MCPClient"] = None    # 兼容旧字段：直连模式下的 MCP 客户端（现由网关管理）
    agent_graph: Optional[AgentGraph] = None    # 单例 Agent 图
    harness: Optional["AgentHarness"] = None    # 单例多 Agent 运行时
    crew_backend: Optional["CrewAIBackend"] = None  # 单例角色扮演编排后端


# 【什么】后台对账循环：睡够间隔 → 跑一次对账 → 有不一致就告警
# 【作用】跨库无事务，MySQL 写失败是静默的，定时对账是发现不一致的兜底手段
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


# 【什么】FastAPI 生命周期钩子：启动时拉起对账线程
# 【作用】interval=0 时不启线程；daemon=True 保证主进程退出时线程不拖住退出
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动/停止钩子：拉起后台对账线程（RECONCILE_INTERVAL_HOURS=0 关闭）。"""
    interval = int(get_settings().reconcile_interval_hours)
    if interval > 0:
        threading.Thread(target=_reconcile_loop, args=(interval,), daemon=True).start()
        _logger.info(f"跨库对账后台任务已启动: 每 {interval} 小时一次")
    yield


# 【什么】创建 FastAPI 应用实例
# 【作用】配好 CORS 与静态目录挂载，并挂上 lifespan 钩子；路由由 APIServer 注册
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


# 【什么】API 服务封装：建 app、初始化组件、注册路由
# 【作用】main.py 只需 APIServer().run() 就能起一个完整服务
class APIServer:
    """API 服务封装：组件初始化 + 路由注册。"""

    # 【什么】构造：建 FastAPI app 并装配全部组件与路由
    # 【作用】构造完成即处于可服务状态；任何组件失败都走降级而不是中断启动
    def __init__(self): # self.app: FastAPI
        self.app = create_app()
        self.ctx: AppContext = AppContext()
        self._init_components() # 初始化全局组件
        self._register_routes() # 注册路由

    # 【什么】初始化全局组件（LLM、记忆、工具、技能、网关、Agent 图）
    # 【作用】每个可选能力都用 try 包住，失败只记 warning 并降级，不让服务起不来
    def _init_components(self):
        """初始化全局组件（LLM、记忆、工具、Agent图）。"""
        s = get_settings()
        _logger.info("初始化后端服务组件...")

        self.ctx.llm = LLMClient()
        self.ctx.pg = PGManager()
        self.ctx.memory = MemoryManager(llm_client=self.ctx.llm)

        # ------------------------------------------------------------------
        # 【工具 Harness】能力声明 / 调用前校验 / 人工审批 / 沙箱 / 预算 / 结果过滤
        #   装配失败或开关关闭时 harness=None，ToolRegistry 退化为改造前行为。
        #   gate 只在开启 HITL 时注入：高风险工具（high/critical）需要真人拍板，
        #   没开 HITL 时按 tool_on_no_gate 处理（默认 deny，安全优先）。
        # ------------------------------------------------------------------
        harness = None
        try:
            if s.tool_harness_enabled:
                from modules.harness import build_tool_harness
                gate = None
                if s.hitl_enabled:
                    from modules.hitl import get_global_gate
                    gate = get_global_gate()
                harness = build_tool_harness(s, gate=gate)
        except Exception as e:
            _logger.warning(f"工具 Harness 装配失败（按无治理运行）: {e}")
            harness = None
        self.ctx.tool_harness = harness

        self.ctx.tool_registry = ToolRegistry(harness=harness)
        self.ctx.tool_registry.register_batch(self.ctx.tool_registry.get_default_tools())
        if harness is not None:
            # 绑定后 SchemaMiddleware 才能按工具名取到 JSON Schema（延迟注入避免循环依赖）
            harness.bind_registry(self.ctx.tool_registry)
            _logger.info(f"工具 Harness 已装配: 中间件={harness.chain.names()} | 闸门={'开' if harness.gate else '关'}")

        from modules.skills import SkillRegistry, get_default_skills
        self.ctx.skill_registry = SkillRegistry()
        self.ctx.skill_registry.register_batch(get_default_skills())
        _logger.info(f"技能已加载: {len(self.ctx.skill_registry.list_skills())} 个")

        # ------------------------------------------------------------------
        # 【MCP 技能网关 + Tool-RAG + 分层路由】与 main.py 同一套装配
        #   改造前：discover_tools() 全量注册 → 每个请求都把全部工具 schema 发给模型
        #   改造后：远程技能只进「目录 + 向量索引」，每轮由路由决定带哪几个
        #
        #   【启动命令为什么用 sys.executable】写死 "python" 在虚拟环境里可能指向
        #   别的解释器，MCP 子进程起来后 import 不到本项目模块，握手直接失败。
        # ------------------------------------------------------------------
        self.ctx.gateway = None
        router = None
        try:
            from modules.mcp.gateway import MCPGateway
            from modules.tools.tool_rag import ToolRAGIndex
            from modules.agent.router import AgentRouter
            from modules.common.circuit_breaker import CircuitBreaker

            if s.mcp_enabled:
                self.ctx.gateway = MCPGateway.from_settings(
                    s,
                    breaker=CircuitBreaker(s.cb_failure_threshold, s.cb_recovery_seconds),
                    default_command=[sys.executable, "-m", "modules.mcp.mcp_server"],
                )
                _logger.info(f"MCP 网关已就绪: servers={self.ctx.gateway.server_names} | "
                             f"传输={self.ctx.gateway.transports}")

            tool_index = None
            if s.tool_rag_enabled:
                # 召回 top_k 个候选，相似度低于 min_score 的直接丢掉（配置：TOOL_RAG_TOP_K / TOOL_RAG_MIN_SCORE）
                tool_index = ToolRAGIndex(top_k=s.tool_rag_top_k, min_score=s.tool_rag_min_score)
                tool_index.index_tools(self.ctx.tool_registry.list_tools())
                if self.ctx.gateway:
                    tool_index.index_tools(self.ctx.gateway.catalog())
                _logger.info(f"Tool-RAG 索引已建立: {tool_index.stats()}")

            router = AgentRouter(
                skill_registry=self.ctx.skill_registry,
                tool_index=tool_index,
                gateway=self.ctx.gateway,
                top_k=s.tool_rag_top_k,            # 语义召回候选数
                max_tools=s.max_tools_per_turn,    # 单轮最多带进 prompt 的工具数（配置：MAX_TOOLS_PER_TURN）
            )
        except Exception as e:
            _logger.warning(f"技能网关/路由降级（按全部工具运行）: {e}")
            router = None

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
            router=router,
            tool_harness=self.ctx.tool_harness,     # 预算账本与拦截逻辑与 registry 共用同一实例
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

    # 【什么】注册全部 API 路由（本文件的主要部分）
    # 【作用】路由以闭包形式挂到 self.app 上，共享同一个 ctx
    def _register_routes(self):
        """注册 API 路由。"""
        app = self.app
        ctx = self.ctx

        # 【什么】GET /health —— 真探活（免鉴权，供容器探针调用）
        # 【作用】逐项探测 PG / MySQL / Milvus；PG 不可用直接 503，让编排能重启容器
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

        # 【什么】GET /me —— 当前登录者的权限快照
        # 【作用】排障第一步："我是谁、我有什么权限"；principal 单独展开成 dict
        @app.get("/me")
        async def me(user: dict = Depends(get_current_user)):
            """当前登录者的权限快照：排障时先问"我是谁、我有什么权限"。"""
            p = user.get("principal")
            return {
                "user": {k: v for k, v in user.items() if k != "principal"},
                "permissions": p.to_dict() if p else None,
            }

        # 【什么】GET /admin/audit —— 查审计日志（需 api:audit:* 权限）
        # 【作用】按 task_id 或 session_id 串联一次请求在两库留下的记录
        @app.get("/admin/audit")
        async def audit_log(
            limit: int = 50,       # 最多返回多少条，默认 50
            session_id: str = "",  # 按会话号过滤；留空不过滤
            task_id: str = "",     # 按流水号过滤；留空不过滤（优先级高于 session_id）
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

        # 【什么】GET /admin/bill —— 按部门汇总 LLM token 消耗（需 api:admin:* 权限）
        # 【作用】成本治理与预算控制的数据源；按 tokens 倒序，花得最多的部门在最前
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

        # 【什么】GET /admin/offboard/preview —— 离职数据对账预览（只报告不改）
        # 【作用】列出哪些离职员工的会话还没归档，给管理员决定要不要执行
        @app.get("/admin/offboard/preview")
        async def offboard_preview(user: dict = Depends(RequirePerm("api:admin:*"))):
            """离职数据对账：只看不改，报告哪些离职员工的会话仍未归档。"""
            from modules.enterprise import lifecycle
            result = lifecycle.preview()
            if not result.get("ok"):
                raise HTTPException(status_code=503, detail=result.get("error"))
            return result

        # 【什么】POST /admin/offboard/run —— 执行离职数据处置
        # 【作用】离职员工会话归档 + 超保留期数据清理；真正落库的一步
        @app.post("/admin/offboard/run")
        async def offboard_run(
            retention_days: int = 0,   # 保留天数；0 表示不传，用配置里的默认值
            user: dict = Depends(RequirePerm("api:admin:*"))
        ):
            """执行离职数据处置：离职员工会话归档 + 超保留期数据清理。"""
            from modules.enterprise import lifecycle
            result = lifecycle.run_once(retention_days or None)
            if not result.get("ok"):
                raise HTTPException(status_code=503, detail=result.get("errors") or result)
            return result

        # 【什么】GET /admin/reconcile —— 跨库对账（按 task_id 比对两库）
        # 【作用】列出单边缺失的号：pg_only 是审计丢失，mysql_only 是会话已清理或写库失败
        @app.get("/admin/reconcile")
        async def cross_db_reconcile(
            hours: int = 24,   # 回溯最近多少小时，默认 24
            user: dict = Depends(RequirePerm("api:admin:*"))
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

        # 【什么】POST /admin/archive/run —— 冷热分离归档
        # 【作用】超过保留期的会话先导出归档文件，再从 PG 里删掉
        @app.post("/admin/archive/run")
        async def archive_run(
            retention_days: int = 0,   # 保留天数；0 表示取配置里的默认值
            user: dict = Depends(RequirePerm("api:admin:*"))
        ):
            """冷热分离：超过保留期的会话先导出归档文件，再从 PG 删除。"""
            from modules.memory.archiver import archive_expired

            result = archive_expired(retention_days or 0)
            if not result.get("ok"):
                raise HTTPException(status_code=503, detail=result.get("errors") or result)
            return result

        # 【什么】Human-in-the-Loop 审批端点：/hitl/pending、/hitl/reply、/hitl/records
        # 【作用】给"人"这一侧一个可操作的入口：看待办、点决策、查留痕。
        #   闸门必须用进程级共享的那个（get_global_gate），
        #   否则人在这边点批准、Agent 链路等在另一个 store 上，永远收不到。
        #   注册失败只告警不抛出：审批是增强能力，不能因为它拖垮主服务。
        try:
            from modules.api.hitl_routes import register_hitl_routes
            from modules.hitl import get_global_gate

            register_hitl_routes(app, get_global_gate(), RequirePerm("api:admin:*"))
            _logger.info("HITL 审批端点已注册: /hitl/pending /hitl/stream /hitl/reply /hitl/records")
        except Exception as e:
            _logger.warning(f"HITL 端点注册失败（不影响主流程）: {e}")

        # 【什么】GET / —— 前端首页（免鉴权）
        # 【作用】返回 static/index.html；文件不存在时 404 而不是返回空页
        @app.get("/")
        async def index():
            """前端首页：返回 jQuery 界面。"""
            static_dir = Path(__file__).resolve().parent.parent.parent / "static"
            index_file = static_dir / "index.html"
            if index_file.is_file():
                return FileResponse(str(index_file))
            raise HTTPException(status_code=404, detail="前端界面未找到")

        # 【什么】POST /chat —— 对话主接口，SSE 流式返回
        # 【作用】鉴权 → 会话归属校验 → Agent 推理 → 按事件逐段推给前端
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
                # 会话标题取用户输入的前 50 个字，够列表展示又不把整段提问塞进标题
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

            # 【什么】按 SSE 协议拼一个事件帧
            # 【作用】"event: 名\ndata: JSON\n\n"，结尾空行是帧分隔符，少它前端收不到事件
            def sse_event(event: str, data: dict) -> str:
                """按 SSE 协议拼一个事件帧：event/data 两行，事件间以空行分隔。"""
                return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

            # 【什么】SSE 事件生成器：按推理进度 yield 一串事件帧
            # 【作用】跑在线程池里执行 Agent（不阻塞事件循环），异常时也保证把 done 帧发出去
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
                # 每片 24 个字，片间停 20 毫秒：既有打字机效果，又不至于把连接刷爆
                for i in range(0, len(answer), 24):
                    yield sse_event("answer_chunk", {"content": answer[i:i + 24]})
                    await asyncio.sleep(0.02)

                yield sse_event("answer_end", {"summary": result.get("llm_summary", "")})
                yield sse_event("done", {"session_id": session_id, "task_id": task_id})

            return StreamingResponse(
                event_generator(),

                media_type="text/event-stream",   # SSE 固定类型，前端按 text/event-stream 解析

                # no-cache 防止中间层缓存流；X-Accel-Buffering: no 关掉 Nginx 的响应缓冲
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # 【什么】POST /agent/team/run —— 多 Agent 团队协作
        # 【作用】主管拆任务 → 并行执行 → 汇总 → 终审；图片只收 data URL 或 http(s) 链接
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
            # 没传会话号时生成 team-＋8 位十六进制，避免多轮团队会话互相覆盖
            session_id = req.session_id or f"team-{uuid.uuid4().hex[:8]}"
            return runner.run(req.query, mm=mm, session_id=session_id)

        # 【什么】POST /agent/eval —— 多 Agent 回归评测（管理员）
        # 【作用】改提示词或团队配置后跑一次，通过率掉了说明退化
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

        # 【什么】GET /agent/backends —— 可选框架后端探活
        # 【作用】分清"配了没装"和"装了没配"，前者表现为静默降级
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

        # 【什么】GET /history/{session_id} —— 取会话历史（带归属校验）
        # 【作用】只返回本人会话的历史；不是自己的会话直接 403
        @app.get("/history/{session_id}")
        async def get_history(
            session_id: str,
            limit: int = 20,   # 最多返回多少条，默认 20
            user: dict = Depends(get_current_user),
        ):
            """获取会话历史（会话隔离）。"""
            if not AuthManager.check_session_owner(user["user_id"], session_id, ctx.pg):
                raise HTTPException(status_code=403, detail="无权访问该会话")
            rows = ctx.pg.get_short_term_history(session_id, user["user_id"], limit)
            return {"session_id": session_id, "history": rows}

        # 【什么】POST /knowledge/ingest —— 文档入库（管理员）
        # 【作用】上传文件先落临时目录再切片入库；kb_code 决定写进哪个知识库
        @app.post("/knowledge/ingest")
        async def ingest_document(
            file: UploadFile = File(...),      # 上传的文件，必填
            kb_code: str = Form("public"),     # 目标知识库编码，默认 public
            user: dict = Depends(RequirePermission(UserLevel.ADMIN)),
        ):
            """知识库文档入库（管理员权限）。

            kb_code: 目标知识库编码。多集合模式下决定写进哪个 Milvus 物理集合；
                     单集合模式下只作为 kb_code 标量字段写入，不影响落哪个集合。
            """
            if not ctx.rag:
                raise HTTPException(status_code=503, detail="RAG模块未启用")
            import tempfile, os
            # file.filename 可能是 None（客户端没带文件名），兜底一个名字，避免拼路径时类型不兼容
            tmp_path = os.path.join(tempfile.gettempdir(), file.filename or "upload.tmp")
            with open(tmp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            count = ctx.rag.ingest_document(tmp_path, kb_code=kb_code)
            os.remove(tmp_path)
            return {"status": "success", "chunks": count, "file": file.filename, "kb_code": kb_code}

        # 【什么】GET /knowledge/stats —— 知识库统计
        # 【作用】只统计当前用户可见范围内的库，可见范围外的连数量都不暴露
        @app.get("/knowledge/stats")
        async def knowledge_stats(user: dict = Depends(get_current_user)):
            """知识库统计信息（按当前用户可见的知识库范围统计）。

            传 kb_codes 是为了不让普通用户看到"全公司文档总量"——
            可见范围之外的库连数量都不该暴露。
            """
            if ctx.rag:
                principal = user.get("principal")
                kb_codes = getattr(principal, "kb_codes", None) or None
                return ctx.rag.get_knowledge_stats(kb_codes=kb_codes)
            return {"status": "RAG模块未启用"}

        # 【什么】GET /skills —— 已注册技能清单
        # 【作用】列出技能总数与每个技能绑定的函数工具，供前端展示与排障
        @app.get("/skills")
        async def list_skills(user: dict = Depends(get_current_user)):
            """获取已注册技能清单及其绑定的函数工具。"""
            return {
                "total": len(ctx.skill_registry.list_skills()),
                "skills": [s.to_dict() for s in ctx.skill_registry.list_skills()],
            }

        # 【什么】GET /tools/logs —— 工具调用审计（需 ADVANCED 及以上等级）
        # 【作用】按会话号查工具调用记录；当前只回显入参，实际取数在 PG 侧
        @app.get("/tools/logs")
        async def tool_logs(
            session_id: Optional[str] = None,   # 按会话号过滤；不传不过滤
            limit: int = 50,                    # 最多返回多少条，默认 50
            user: dict = Depends(RequirePermission(UserLevel.ADVANCED)),
        ):
            """工具调用审计日志（高级用户+权限）。"""
            return {"status": "audit", "session_id": session_id, "limit": limit}

        # 【什么】GET /reviews/pending —— 待人工审核列表（管理员）
        # 【作用】把 PG 里待审核的问答取出来，供人工判定并反哺标准答案
        @app.get("/reviews/pending")
        async def pending_reviews(
            user: dict = Depends(RequirePermission(UserLevel.ADMIN)),
        ):
            """待人工审核列表（管理员权限）。"""
            return {"reviews": ctx.pg.get_pending_reviews()}

        # ------------------------------------------------------------------
        # 【流水线端点】独立模块（modules/api/pipeline_routes.py），
        #   注册失败只记日志：流水线是增强能力，不该让主服务起不来。
        # ------------------------------------------------------------------
        try:
            from modules.api.pipeline_routes import register_pipeline_routes
            register_pipeline_routes(app)
            _logger.info("流水线端点已注册: POST /pipeline/run, POST /pipeline/tools/index, GET /pipeline/health")
        except Exception as e:
            _logger.warning(f"流水线端点注册失败: {e}")

        # 【什么】POST /reviews/{review_id}/review —— 提交人工审核结果（管理员）
        # 【作用】approved 且填了标准答案时，把标准答案存进标准答案库
        @app.post("/reviews/{review_id}/review")
        async def submit_review(
            review_id: int,       # 待审核项 id，来自 /reviews/pending
            req: ReviewRequest,   # 审核动作与标准答案
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

    # 【什么】启动 API 服务（uvicorn）
    # 【作用】按配置的 APP_HOST / APP_PORT 监听，阻塞当前进程
    def run(self):
        """启动 API 服务。"""
        import uvicorn

        s = get_settings()
        _logger.info(f"API服务启动: {s.app_host}:{s.app_port}")
        uvicorn.run(self.app, host=s.app_host, port=s.app_port)
