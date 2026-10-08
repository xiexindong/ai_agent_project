"""
FastAPI 后端服务模块
=====================
企业级 API 服务：对话接口、知识库管理、会话管理、工具日志审计。
所有接口统一鉴权、限流、多用户隔离。
"""
# ---- 标准库：文件/系统/序列化/唯一号/时间/异步/线程 ----
import os  # 操作系统接口：环境变量、路径拼接等
import sys  # 解释器接口：取 sys.executable 作为 MCP 子进程启动命令
import json  # 序列化：SSE 事件帧的 data 字段要 dumps 成 JSON 字符串
import uuid  # 唯一号生成：新会话 session_id、团队会话号
import time  # 时间函数：后台对账线程的 sleep
import asyncio  # 异步运行时：to_thread 跑同步 Agent、sleep 控制打字机节奏
import threading  # 线程：对账循环跑在 daemon 线程里
from pathlib import Path  # 面向对象路径：定位 static 目录
from typing import Any, Optional, Union, TYPE_CHECKING  # 类型标注：可选/联合/仅类型检查期导入
from contextlib import asynccontextmanager  # 异步上下文管理器：定义 lifespan 启动/停止钩子
# ---- FastAPI 全家桶：应用/依赖注入/异常/上传表单/跨域/静态文件/响应类型 ----
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form  # 框架核心件
from fastapi.middleware.cors import CORSMiddleware  # 跨域中间件：浏览器前端直连必备
from fastapi.staticfiles import StaticFiles  # 静态文件子应用：托管 /static 下的前端资源
from fastapi.responses import FileResponse, StreamingResponse  # 文件响应与 SSE 流式响应
# ---- Pydantic：请求/响应体基类，以及上传图片校验失败时拿到的异常类型 ----
from pydantic import BaseModel, ValidationError  # 请求体模型基类 + 校验异常（图片 data URL 校验）
# ---- 项目内公共件：日志器、业务流水号与请求上下文 ----
from modules.common.logger import get_logger  # 统一日志器工厂
from modules.common.context import new_task_id, set_request_context  # 业务流水号生成与请求上下文写入
# ---- 统一鉴权：依赖注入函数、权限/等级要求、会话归属校验 ----
from modules.api.auth import (
    get_current_user,  # 依赖注入：解析 token 得到当前用户
    RequirePermission,  # 依赖注入：要求指定用户等级
    RequirePerm,  # 依赖注入：要求指定权限码
    UserLevel,  # 用户等级枚举（ADMIN/ADVANCED/...）
    AuthManager,  # 鉴权管理器：会话归属校验等静态方法
)
# ---- Agent：LangGraph 编排外壳、ReAct 业务引擎、大模型客户端 ----
from modules.agent.graph import AgentGraph  # LangGraph 编排外壳：路由实际调用的入口
from modules.agent.react_agent import ReActAgent  # ReAct 推理引擎：思考-行动-观察循环
from modules.agent.llm import LLMClient  # 大模型客户端：统一封装模型调用与用量落库
# ---- 工具注册、双层记忆与 PG ----
from modules.tools.tool_registry import ToolRegistry  # 工具注册表：注册/去重/限次/治理
from modules.memory.memory_manager import MemoryManager  # 双层记忆：短期历史 + 长期摘要
from modules.memory.pg_manager import PGManager  # Postgres 管理器：会话/历史/审核/标准答案
# ---- 多模态（团队接口的图片输入）、多 Agent 编排与数据模型 ----
from modules.multimodal.media import MultiModalInput, ImagePart  # 多模态输入模型与图片部件
from modules.multiagent.harness import build_harness  # 多 Agent 运行时工厂
from modules.multiagent.models import EvalCase, RunResult  # 评测用例与运行结果模型
# ---- 全局配置单例 ----
from config import get_settings  # 配置单例工厂：全模块共享同一份 Settings

# 【什么】只在类型检查阶段执行的导入（TYPE_CHECKING 恒为 False，运行时不导入）
# 【作用】给 AppContext 字段标注真实类型，避免退化成 object 导致调用方法被判错；
#        运行时不导入，防止拉起 pymilvus / crewai 等重型可选依赖，也避免循环导入
if TYPE_CHECKING:
    from modules.rag.rag_chain import RAGChain  # 自研 RAG 链路（仅类型标注用）
    from modules.rag.llamaindex_adapter import LlamaIndexRAG  # LlamaIndex 适配 RAG 链路
    from modules.skills.skill_registry import SkillRegistry  # 技能注册表
    from modules.multiagent.harness import AgentHarness  # 多 Agent 运行时
    from modules.multiagent.crewai_backend import CrewAIBackend  # CrewAI 编排后端
    from modules.mcp.gateway import MCPGateway  # MCP 技能网关
    from modules.mcp.mcp_client import MCPClient  # MCP 直连客户端（旧字段兼容）

# 本模块统一日志器：名 "api"，日志前缀据此显示，也可按模块名做日志过滤
_logger = get_logger("api")  # 模块级日志器单例


# 【什么】POST /chat 的请求体
# 【作用】带 session_id 表示续聊，不带则由服务端新开一个会话
class ChatRequest(BaseModel):  # Pydantic 请求体模型
    session_id: Optional[str] = None   # 已有会话号；不传 = 新会话
    query: str                         # 用户这一轮的输入


# 【什么】对话结果的响应体
# 【作用】把一轮对话的产出字段固定下来，前端按名字取值
class ChatResponse(BaseModel):  # Pydantic 响应体模型
    session_id: str           # 会话号
    answer: str               # 最终答案
    summary: str              # 本轮摘要，供长期记忆使用
    active_skill: str = ""    # 命中的技能名；未命中是空串
    iterations: int           # ReAct 迭代轮数
    tool_calls: list          # 本轮工具调用记录


# 【什么】POST /agent/team/run 的请求体
# 【作用】描述要交给团队完成的任务、可选图片与编排后端
class TeamRunRequest(BaseModel):  # 团队运行请求体
    """多 Agent 团队运行请求。images 传 data URL 或 http(s) 图片链接。"""

    query: str                         # 交给团队的任务
    session_id: Optional[str] = None   # 会话号；不传时生成 team-xxxxxxxx
    images: list[str] = []             # 图片来源；只收 data URL 或 http(s) 链接
    backend: str = "native"            # native（自研Harness）| crewai（角色扮演编排，需装可选依赖）


# 【什么】POST /agent/eval 的请求体
# 【作用】cases 为空时只跑内置连通性用例，验链路不验质量
class AgentEvalRequest(BaseModel):  # 评测请求体
    """多 Agent 评测请求。cases 为空时用内置默认用例跑一次连通性。"""
    cases: list[dict] = []   # 评测用例；空列表 = 只跑内置默认用例


# 【什么】会话信息的响应体
# 【作用】新建会话后回给前端会话号、归属用户与创建时间
class SessionResponse(BaseModel):  # 会话信息响应体
    session_id: str   # 会话号
    user_id: str      # 归属用户工号
    created_at: str   # 创建时间


# 【什么】POST /reviews/{review_id}/review 的请求体
# 【作用】承载审核动作；通过时可用标准答案反哺标准答案库
class ReviewRequest(BaseModel):  # 人工审核请求体
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
    while True:  # 无限循环：daemon 线程随主进程退出而终止
        time.sleep(interval_hours * 3600)  # 睡够一个对账周期（小时转秒）
        try:
            from modules.enterprise.reconcile import reconcile  # 延迟导入：避免模块加载时的循环依赖

            result = reconcile(hours=max(interval_hours * 2, 24))  # 对账窗口至少 24 小时，覆盖可能的延迟
            if not result.get("ok") or result.get("pg_only") or result.get("mysql_only"):  # 任一侧有单边数据即异常
                _logger.warning(f"定时对账发现不一致: {result}")  # 告警但不中断：对账是监控手段不是阻断器
        except Exception as e:
            _logger.warning(f"定时对账异常: {e}")  # 吞掉异常保循环：对账线程挂了主服务也不能停


# 【什么】FastAPI 生命周期钩子：启动时拉起对账线程
# 【作用】interval=0 时不启线程；daemon=True 保证主进程退出时线程不拖住退出
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动/停止钩子：拉起后台对账线程（RECONCILE_INTERVAL_HOURS=0 关闭）。"""
    interval = int(get_settings().reconcile_interval_hours)  # 从配置读对账间隔（小时）
    if interval > 0:  # 0 表示关闭对账，不启线程
        threading.Thread(target=_reconcile_loop, args=(interval,), daemon=True).start()  # daemon 线程：主进程退出时自动回收
        _logger.info(f"跨库对账后台任务已启动: 每 {interval} 小时一次")  # 启动日志确认对账线程就位
    yield  # yield 之后是停止钩子（当前无清理逻辑）


# 【什么】创建 FastAPI 应用实例
# 【作用】配好 CORS 与静态目录挂载，并挂上 lifespan 钩子；路由由 APIServer 注册
def create_app() -> FastAPI:  # 应用工厂函数
    """创建 FastAPI 应用实例。"""
    app = FastAPI(title="企业级 AI Agent 智能助手", version="1.0.0", lifespan=lifespan)  # 建实例并挂生命周期钩子

    # 配置里按逗号写允许的前端来源，这里逐个去空白；空串项丢掉，避免匹配到非法来源
    origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],   # 一个都没配时放开全部来源（开发友好；生产应显式配置）
        allow_methods=["*"],              # 放行全部 HTTP 方法（GET/POST/PUT/DELETE…）
        allow_headers=["*"],              # 放行全部请求头（含 Authorization 鉴权头）
    )
    # 静态目录固定在项目根/static：__file__ 是 modules/api/server.py，上溯三级到项目根
    static_dir = Path(__file__).resolve().parent.parent.parent / "static"
    if static_dir.is_dir():
        # 挂成子应用：/static/xxx 的请求直接由 StaticFiles 发文件，不进路由函数
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
        self.app = create_app()              # 裸 FastAPI 实例（CORS/静态/lifespan 已就绪，还没有业务路由）
        self.ctx: AppContext = AppContext()  # 进程级组件容器，路由闭包都从它取共享单例
        self._init_components() # 初始化全局组件（LLM/记忆/工具/技能/网关/图）
        self._register_routes() # 注册路由（必须在组件就绪后，路由要捕获 ctx）

    # 【什么】初始化全局组件（LLM、记忆、工具、技能、网关、Agent 图）
    # 【作用】每个可选能力都用 try 包住，失败只记 warning 并降级，不让服务起不来
    def _init_components(self):
        """初始化全局组件（LLM、记忆、工具、Agent图）。"""
        s = get_settings()
        _logger.info("初始化后端服务组件...")

        # 三个基础组件先就位，后面所有装配都围绕它们接线
        self.ctx.llm = LLMClient()                       # 大模型客户端（对话/工具决策/摘要全走它）
        self.ctx.pg = PGManager()                        # PG 连接管理（会话/历史/向量读写）
        self.ctx.memory = MemoryManager(llm_client=self.ctx.llm)  # 双层记忆：摘要生成需要复用同一个 LLM 客户端

        # ------------------------------------------------------------------
        # 【工具 Harness】能力声明 / 调用前校验 / 人工审批 / 沙箱 / 预算 / 结果过滤
        #   装配失败或开关关闭时 harness=None，ToolRegistry 退化为改造前行为。
        #   gate 只在开启 HITL 时注入：高风险工具（high/critical）需要真人拍板，
        #   没开 HITL 时按 tool_on_no_gate 处理（默认 deny，安全优先）。
        # ------------------------------------------------------------------
        harness = None   # 先按 None（无治理）兜底：下面任何一步装配失败，服务仍能正常启动
        try:
            # 总开关关闭时整段跳过：Harness 是增强项不是依赖项，关掉行为回到改造前
            if s.tool_harness_enabled:
                from modules.harness import build_tool_harness   # 函数内延迟 import：启动失败也不影响本模块加载
                gate = None                                      # 人工审批闸门；None = 不启用 HITL
                if s.hitl_enabled:
                    # 开启 Human-in-the-loop：取全局唯一的人工闸门（审批通知/等待/落库都由它负责）
                    from modules.hitl import get_global_gate
                    gate = get_global_gate()
                # 按配置装配完整 Harness（策略/中间件/预算/沙箱/结果过滤 + Guard/Retry），并注入人工闸门
                harness = build_tool_harness(s, gate=gate)
        except Exception as e:
            # 装配失败（配置错误/依赖缺失等）只告警不抛出：保证 API 服务始终可启动，按无治理降级
            _logger.warning(f"工具 Harness 装配失败（按无治理运行）: {e}")
            harness = None
        self.ctx.tool_harness = harness   # 挂到全局上下文：Agent/路由等其他组件从这里取用

        # 工具注册与 Harness 同一行接线：harness=None 时注册器退化为"注册+去重+限次"
        self.ctx.tool_registry = ToolRegistry(harness=harness)
        # 注册内置工具（calculator/data_query）：先注册、后绑定，顺序不能反
        self.ctx.tool_registry.register_batch(self.ctx.tool_registry.get_default_tools())
        if harness is not None:
            # 绑定后 SchemaMiddleware 才能按工具名取到 JSON Schema（延迟注入避免循环依赖）
            harness.bind_registry(self.ctx.tool_registry)
            # 打印装配结果：中间件清单确认治理链生效，闸门状态确认人工审批是否就位
            _logger.info(f"工具 Harness 已装配: 中间件={harness.chain.names()} | 闸门={'开' if harness.gate else '关'}")

        # 技能系统独立装配（与工具并行，互不依赖）：函数内延迟 import 与上面保持同一写法
        from modules.skills import SkillRegistry, get_default_skills
        self.ctx.skill_registry = SkillRegistry()
        # 注册内置技能包（含触发词/SOP/推荐工具），供 Agent 每轮开始前按意图匹配
        self.ctx.skill_registry.register_batch(get_default_skills())
        # 打印技能数量：启动日志里据此确认技能包加载成功（数量异常偏少说明装配有问题）
        _logger.info(f"技能已加载: {len(self.ctx.skill_registry.list_skills())} 个")

        # ------------------------------------------------------------------
        # 【MCP 技能网关 + Tool-RAG + 分层路由】与 main.py 同一套装配
        #   改造前：discover_tools() 全量注册 → 每个请求都把全部工具 schema 发给模型
        #   改造后：远程技能只进「目录 + 向量索引」，每轮由路由决定带哪几个
        #
        #   【启动命令为什么用 sys.executable】写死 "python" 在虚拟环境里可能指向
        #   别的解释器，MCP 子进程起来后 import 不到本项目模块，握手直接失败。
        # ------------------------------------------------------------------
        self.ctx.gateway = None   # 网关先置空：mcp_enabled=False 时保持 None，路由据此跳过远程工具
        router = None             # 分层路由器先置空：装配失败时 Agent 退回"全量工具"模式
        try:
            from modules.mcp.gateway import MCPGateway
            from modules.tools.tool_rag import ToolRAGIndex
            from modules.agent.router import AgentRouter
            from modules.common.circuit_breaker import CircuitBreaker

            if s.mcp_enabled:
                # from_settings 按配置里的 server 清单纳管；熔断器在连续失败时自动开路，防止拖垮主链路
                self.ctx.gateway = MCPGateway.from_settings(
                    s,
                    breaker=CircuitBreaker(s.cb_failure_threshold, s.cb_recovery_seconds),
                    default_command=[sys.executable, "-m", "modules.mcp.mcp_server"],
                )
                # servers=纳管的服务名；transports=每个服务用 stdio 还是 http，启动时核对配置是否生效
                _logger.info(f"MCP 网关已就绪: servers={self.ctx.gateway.server_names} | "
                             f"传输={self.ctx.gateway.transports}")

            tool_index = None   # 语义索引先置空：tool_rag_enabled=False 时路由只走关键词
            if s.tool_rag_enabled:
                # 召回 top_k 个候选，相似度低于 min_score 的直接丢掉（配置：TOOL_RAG_TOP_K / TOOL_RAG_MIN_SCORE）
                tool_index = ToolRAGIndex(top_k=s.tool_rag_top_k, min_score=s.tool_rag_min_score)
                tool_index.index_tools(self.ctx.tool_registry.list_tools())  # 本地工具先入索引
                if self.ctx.gateway:
                    tool_index.index_tools(self.ctx.gateway.catalog())  # 远程工具目录也入索引（只存语义，不连进程）
                _logger.info(f"Tool-RAG 索引已建立: {tool_index.stats()}")

            # 路由器把"技能匹配 + 工具语义召回 + 网关目录"收敛成每轮一个工具名单
            router = AgentRouter(
                skill_registry=self.ctx.skill_registry,
                tool_index=tool_index,
                gateway=self.ctx.gateway,
                top_k=s.tool_rag_top_k,            # 语义召回候选数
                max_tools=s.max_tools_per_turn,    # 单轮最多带进 prompt 的工具数（配置：MAX_TOOLS_PER_TURN）
            )
        except Exception as e:
            # 网关/索引/路由任一失败：记 warning 并让 router=None，Agent 按"全量工具可用"继续跑
            _logger.warning(f"技能网关/路由降级（按全部工具运行）: {e}")
            router = None

        # 走工厂而不是直接 new RAGChain：RAG_BACKEND=llamaindex 时接管，
        # 库没装会自动降级回自研链路（降级在工厂里记 warning，这里只兜底异常）
        try:
            from modules.rag.llamaindex_adapter import build_rag
            self.ctx.rag = build_rag()   # 工厂内部按 RAG_BACKEND 决定返回自研链路还是 LlamaIndex 适配
        except Exception as e:
            # Milvus 没起/依赖没装：RAG 置空，知识库相关接口会回 503，对话主链路不受影响
            _logger.warning(f"RAG 模块降级（Milvus不可用）: {e}")
            self.ctx.rag = None

        # 把前面装配好的单例全部注入 ReAct 引擎；顺序无关，缺哪个传 None 就走对应降级
        react_agent = ReActAgent(
            llm_client=self.ctx.llm,
            tool_registry=self.ctx.tool_registry,
            rag_chain=self.ctx.rag,
            memory_manager=self.ctx.memory,
            skill_registry=self.ctx.skill_registry,
            router=router,
            tool_harness=self.ctx.tool_harness,     # 预算账本与拦截逻辑与 registry 共用同一实例
        )
        self.ctx.agent_graph = AgentGraph(react_agent)   # 包成 LangGraph：路由实际调用的是 agent_graph.run

        # 多 Agent 团队：共用上面的 LLM 与工具注册器，团队组成走默认 4 人配置
        # ------------------------------------------------------------------
        # 【这段在装什么】上面 324-333 行装的是"单 Agent 主链路"（/chat 用）：
        #   一个 ReActAgent 从头干到尾，不会拆任务。这里再装一套"团队模式"
        #   （/agent/team/run 用）：主管拆任务 → 两个 Worker 并行干 → 审核员
        #   把关。两条入口、两套运行时，互不影响。
        #
        # 【为什么要这个？】复杂问题（"查 Q3/Q4 营收再算环比"）塞给一个 Agent，
        #   上下文越滚越大，容易顾此失彼甚至编数据。拆成子任务后，每个
        #   Worker 只背自己那份最小上下文（上下文隔离），还能并行执行，
        #   主管专职"拆"和"汇总"。
        #
        # 【"共用 LLM 与工具注册器"是什么意思】团队不是另起炉灶：
        #   build_harness 拿到的 llm / tool_registry 与主链路是同一个单例，
        #   Worker 调 calculator、data_query 时走的并发闸、重试、权限过滤、
        #   审计日志和 /chat 完全一致——不会出现"团队里的工具调用不受治理"
        #   的两套口径。JS 类比：像主 HTTP 服务之外再起一个 worker 池，
        #   但两边共用同一个数据库连接池和 logger，而不是各配一套。
        #
        # 【"默认 4 人"是哪 4 人】（agents.py 的 build_default_team）
        #   supervisor 主管：拆任务、汇总答案，不亲自干活
        #   researcher_1 检索员：查事实，带 data_query 工具、可看图
        #   analyst_1 分析员：计算对比，带 calculator 工具
        #   reviewer_1 审核员：给汇总答案打分，不合格可打回
        #
        # 【为什么整段包 try/except】团队是"可选增强"，不是主服务：
        #   装配失败只把 harness 置 None（/agent/* 返回 503），
        #   而 /chat 照常服务。若让异常往外抛，整个 API 会起不来——
        #   为一个可选能力赔上主服务，是典型的过度耦合。
        #
        # 【开关】MULTIAGENT_ENABLED 默认 True；置 False 时整段跳过、
        #   harness 保持 None，连 build_harness 的装配开销都不发生。
        # ------------------------------------------------------------------
        try:
            if get_settings().multiagent_enabled:
                # 团队复用主链路的 LLM 和工具注册器：不另起一套，保证工具治理口径一致
                self.ctx.harness = build_harness(
                    llm=self.ctx.llm, tool_registry=self.ctx.tool_registry
                )
                # 【委派状态要打在启动日志里】它默认关闭，
                #   不看这行就无法判断接口返回的 RunResult.delegations 为什么一直是空
                s2 = get_settings()
                _logger.info(
                    "多 Agent 团队已装配: 4 人（主管/检索/分析/审核） | 委派="
                    + ("开" if s2.multiagent_delegation_enabled else "关")
                    + (f"（深度<={s2.multiagent_delegation_max_depth} "
                       f"总次数<={s2.multiagent_delegation_max_per_run} "
                       f"单人<={s2.multiagent_delegation_max_per_agent}）"
                       if s2.multiagent_delegation_enabled else "")
                )
        except Exception as e:
            # 团队装配失败：/agent/* 接口返回 503，不影响 /chat 主服务
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
            checks = {}   # 每项依赖一条结果："ok" 正常，"down: 原因" 故障，"disabled" 未启用
            try:
                ctx.pg._reconnect_if_needed()     # 断线先重连：保证探的是"现在能不能连上"
                ctx.pg._cur.execute("SELECT 1")   # 最轻量的探活 SQL，不依赖任何业务表
                checks["postgres"] = "ok"
            except Exception as e:
                checks["postgres"] = f"down: {e}"

            # MySQL 只有开启 RBAC 才是必需依赖；没开就标 disabled，不参与降级判定
            if get_settings().rbac_enabled:
                try:
                    from modules.enterprise.mysql_client import get_mysql_ro_client
                    get_mysql_ro_client().query_one("SELECT 1")   # 只读连接探活，不产生写入
                    checks["mysql"] = "ok"
                except Exception as e:
                    checks["mysql"] = f"down: {e}"
            else:
                checks["mysql"] = "disabled"

            # Milvus 没有独立探活：RAG 组件装配成功（启动时连过一次）即视为可用
            checks["milvus"] = "ok" if ctx.rag else "down: RAG模块未启用"

            # 除 ok/disabled 之外都算降级：disabled 是主动关闭不算故障
            degraded = any(v != "ok" and v != "disabled" for v in checks.values())
            payload = {"status": "degraded" if degraded else "ok", "service": "ai-agent", "checks": checks}
            if checks["postgres"] != "ok":
                # PG 是会话/记忆的核心依赖：直接 503，K8s 探针看到非 2xx 才会重启容器
                raise HTTPException(status_code=503, detail=payload)
            return payload

        # 【什么】GET /me —— 当前登录者的权限快照
        # 【作用】排障第一步："我是谁、我有什么权限"；principal 单独展开成 dict
        @app.get("/me")
        async def me(user: dict = Depends(get_current_user)):
            """当前登录者的权限快照：排障时先问"我是谁、我有什么权限"。"""
            p = user.get("principal")   # Principal 对象单独取出：它带 to_dict，其他字段是普通字符串
            return {
                "user": {k: v for k, v in user.items() if k != "principal"},  # 身份基础字段（user_id/dept_code…）
                "permissions": p.to_dict() if p else None,   # 权限码/数据域/脱敏强度/可见知识库全展开
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
                db = get_mysql_ro_client()   # 只读客户端：查询审计不该走写连接
                # 公共列与基表；WHERE 条件按入参动态追加
                sql = (
                    "SELECT user_code, dept_code, session_id, task_id, action, resource, result, created_at "
                    "FROM sys_audit_log"
                )
                if task_id:
                    # 精确到一轮对话：参数化查询 %s，杜绝审计查询本身被注入
                    rows = db.query_all(
                        sql + " WHERE task_id=%s ORDER BY id DESC LIMIT %s", (task_id, limit)
                    )
                elif session_id:
                    # 精确到一次会话：同一 session_id 下可能跨很多轮
                    rows = db.query_all(
                        sql + " WHERE session_id=%s ORDER BY id DESC LIMIT %s", (session_id, limit)
                    )
                else:
                    # 两个过滤条件都没传：返回全局最近的审计
                    rows = db.query_all(sql + " ORDER BY id DESC LIMIT %s", (limit,))
                return {"total": len(rows), "items": rows}
            except Exception as e:
                # 审计库挂了明确报 503：不能返回空列表让管理员误以为"没有日志"
                raise HTTPException(status_code=503, detail=f"审计库不可用: {e}")

        # 【什么】GET /admin/bill —— 按部门汇总 LLM token 消耗（需 api:admin:* 权限）
        # 【作用】成本治理与预算控制的数据源；按 tokens 倒序，花得最多的部门在最前
        @app.get("/admin/bill")
        async def bill(user: dict = Depends(RequirePerm("api:admin:*"))):
            """按部门汇总 LLM token 消耗：成本治理与预算控制的数据源。"""
            from modules.enterprise.mysql_client import get_mysql_ro_client
            try:
                # 按部门分组：调用次数 + token 总数；ORDER BY tokens DESC 让花费最高的部门排最前
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
            result = lifecycle.preview()   # 只读扫描：对比在职名单与未归档会话，不做任何删除
            if not result.get("ok"):
                # 扫描本身失败（如 MySQL 不可用）转 503，detail 带上具体原因
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
            # None 时由 lifecycle 内部取配置默认保留期；传了天数则以请求参数为准
            result = lifecycle.run_once(retention_days or None)
            if not result.get("ok"):
                # 部分失败也回 503：detail 优先展示 errors 明细，让管理员知道哪几步没做完
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

            result = reconcile(hours)   # 按 task_id 集合比对两库，算出 pg_only / mysql_only
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

            # 0 表示取配置默认保留期；归档动作内部顺序是"先导出文件成功、再删 PG 行"
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
                # FileResponse 直接读文件发回，Content-Type 按 .html 自动判定
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
            user_id = user["user_id"]                 # 工号：会话归属与落库的用户标识
            principal = user.get("principal")         # 权限快照：透传给 Agent 做工具过滤和数据隔离
            is_new_session = not req.session_id       # 没带会话号 = 新会话（后面要先建会话再开跑）
            session_id = req.session_id or str(uuid.uuid4())   # 续聊用原号；新会话现造一个 UUID

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
                # 续聊必须校验归属：防止拿别人的 session_id 直接读历史（越权）
                raise HTTPException(status_code=403, detail="无权访问该会话")

            # 清空该会话上的去重账本/结果缓存：新一轮对话允许重新调用相同工具
            ctx.tool_registry.reset_session(session_id)
            # 把计费上下文挂到 LLM：每次模型调用落用量表时自动带上这四个号，不需要业务侧再传
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
                # 第一帧先发状态：前端立刻展示"思考中"，不用等首轮推理结束
                yield sse_event("status", {"message": "Agent 正在思考...", "phase": "thinking"})
                try:
                    # Agent 是同步重逻辑：丢到线程池执行，不卡住 FastAPI 的事件循环
                    # 【to_thread 的参数结构】第 1 个是函数引用（只传不调用），
                    # 后面 req.query / session_id / user_id / principal 全是它的实参，
                    # 线程池里的工作线程拿到后会真正执行 run(query, session_id, user_id, principal)。
                    # await 期间协程挂起、事件循环继续服务其他请求；线程跑完结果回填，从这里继续。
                    result = await asyncio.to_thread(
                        ctx.agent_graph.run, req.query, session_id, user_id, principal
                    )
                except Exception as e:
                    # 推理异常也要把协议走完：发错误答案 + done 帧，前端不会永远转圈
                    _logger.error(f"Agent推理异常: {e}")
                    yield sse_event("answer_start", {"iterations": 0})
                    yield sse_event("answer_chunk", {"content": f"服务内部错误：{e}"})
                    yield sse_event("answer_end", {"summary": ""})
                    yield sse_event("done", {"session_id": session_id, "task_id": task_id})
                    return

                # Agent 跑完后一次性回放工具流水：每笔先发 start 再发 end，前端按序渲染
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

                # 没有答案时给占位文案：前端不用处理空字符串边界
                answer = result.get("final_answer", "") or "（无回答）"
                yield sse_event("answer_start", {"iterations": result.get("iteration", 0)})
                # 每片 24 个字，片间停 20 毫秒：既有打字机效果，又不至于把连接刷爆
                for i in range(0, len(answer), 24):
                    yield sse_event("answer_chunk", {"content": answer[i:i + 24]})
                    await asyncio.sleep(0.02)

                # 收尾两帧：summary 供前端展示，done 标志流结束（前端据此关闭 loading）
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
                # 模块降级时给明确 503：和"服务挂了"区分开
                raise HTTPException(status_code=503, detail="多Agent模块未启用")
            # 后端选择：请求级 > 全局配置 > native。请求级能覆盖是为了 A/B 对比，
            # 但默认仍是 native —— 换编排引擎是行为变更，不该由客户端静默触发。
            backend = (req.backend or get_settings().multiagent_backend or "native").lower()
            runner = ctx.harness
            if backend == "crewai":
                # 懒加载 CrewAI 后端：没装可选依赖的用户不会在启动时被拖累
                if ctx.crew_backend is None:
                    from modules.multiagent.crewai_backend import CrewAIBackend
                    ctx.crew_backend = CrewAIBackend(llm=ctx.llm, tool_registry=ctx.tool_registry)
                runner = ctx.crew_backend
            parts = []
            for src in req.images or []:
                # 白名单前缀校验：只放行 data URL 和 http(s)，本地路径一律拒绝
                if not src.startswith(("data:", "http://", "https://")):
                    raise HTTPException(
                        status_code=400,
                        detail="图片只接受 data URL 或 http(s) 链接，不接受服务器本地路径",
                    )
                try:
                    # ImagePart 内部再做一次格式合法性校验（如 data URL 的前缀/base64）
                    parts.append(ImagePart(source=src))
                except ValidationError as e:
                    # 取第一条校验错误回给客户端，避免整段冗长错误结构
                    raise HTTPException(status_code=400, detail=f"图片非法: {e.errors()[0]['msg']}")
            # 有图片才构造多模态输入；纯文本任务保持 None，走普通链路
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
            # 传了用例就逐个转成 EvalCase（Pydantic 会校验字段）；没传走内置连通性用例
            cases = [EvalCase(**c) for c in req.cases] if req.cases else [
                EvalCase(name="默认连通性", query="用一句话说明你能做什么", expected_keywords=["可以"]),
            ]
            # evaluate 返回 RunResult；model_dump 转成普通 dict 才能直接做 JSON 响应
            return ctx.harness.evaluate(cases).model_dump()

        # 【什么】GET /agent/backends —— 可选框架后端探活
        # 【作用】分清"配了没装"和"装了没配"，前者表现为静默降级
        @app.get("/agent/backends")
        async def agent_backends(user: dict = Depends(get_current_user)):
            """可选框架后端探活：当前配的哪个、库装没装。

            排障时第一件事就是分清"配了但没装"和"装了但没配"——
            前者表现为静默降级（不报错、结果不一样），最容易误判。
            """
            # is_available 只检查库装没装（importlib），不真正连接服务，接口开销极小
            from modules.multiagent.crewai_backend import is_available as crewai_ok
            from modules.rag.llamaindex_adapter import is_available as li_ok

            s = get_settings()
            return {
                "rag_backend": s.rag_backend,                 # 配置里想用的 RAG 后端
                "multiagent_backend": s.multiagent_backend,   # 配置里想用的编排后端
                "available": {"llamaindex": li_ok(), "crewai": crewai_ok()},  # 实际装没装
            }

        # 【什么】GET /sessions —— 取当前用户的会话列表（侧边栏）
        # 【作用】只返回本人未归档会话，按最近活跃倒序；点开某条时前端再懒加载 /history/{id}
        @app.get("/sessions")
        async def list_sessions(
            limit: int = 50,    # 单次最多返回多少条，默认 50
            user: dict = Depends(get_current_user),
        ):
            """获取当前用户的会话列表（仅封面信息，不含对话明细）。"""
            # SQL 层带 user_id 过滤：列表查询同样做会话隔离，不暴露他人会话
            rows = ctx.pg.list_user_sessions(user["user_id"], limit)
            return {"sessions": rows}

        # 【什么】GET /history/{session_id} —— 取会话历史（带归属校验）
        # 【作用】只返回本人会话的历史；不是自己的会话直接 403
        @app.get("/history/{session_id}")
        async def get_history(
            session_id: str,
            limit: int = 20,   # 最多返回多少条，默认 20
            user: dict = Depends(get_current_user),
        ):
            """获取会话历史（会话隔离）。"""
            # 归属校验先行：不是本人会话，连历史条数都不返回
            if not AuthManager.check_session_owner(user["user_id"], session_id, ctx.pg):
                raise HTTPException(status_code=403, detail="无权访问该会话")
            # 取短期历史（默认最近 20 条），user_id 在 SQL 层再做一次过滤，双保险
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
                # RAG 降级时不允许入库：否则文件处理了却没处存，还可能误报成功
                raise HTTPException(status_code=503, detail="RAG模块未启用")
            import tempfile, os
            # file.filename 可能是 None（客户端没带文件名），兜底一个名字，避免拼路径时类型不兼容
            tmp_path = os.path.join(tempfile.gettempdir(), file.filename or "upload.tmp")
            # 上传内容先完整落到临时文件：RAG 入库按"文件路径"读取，不直接吃字节流
            with open(tmp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            count = ctx.rag.ingest_document(tmp_path, kb_code=kb_code)  # 切片+向量化+写入，返回切片数
            os.remove(tmp_path)   # 临时文件用完即删，避免临时目录堆积
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
                # None = 未开启 RBAC，不做范围过滤；否则只统计本人可见知识库
                kb_codes = getattr(principal, "kb_codes", None) or None
                return ctx.rag.get_knowledge_stats(kb_codes=kb_codes)
            # RAG 降级时回状态字段而不是 503：统计是只读展示，不该阻塞页面
            return {"status": "RAG模块未启用"}

        # 【什么】GET /skills —— 已注册技能清单
        # 【作用】列出技能总数与每个技能绑定的函数工具，供前端展示与排障
        @app.get("/skills")
        async def list_skills(user: dict = Depends(get_current_user)):
            """获取已注册技能清单及其绑定的函数工具。"""
            return {
                "total": len(ctx.skill_registry.list_skills()),
                # 每个技能转 dict：含触发词/SOP/绑定工具，前端直接渲染，无需二次解析
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
            # 直接从 PG 取 status=pending 的记录；鉴权已由 RequirePermission(ADMIN) 完成
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
            # 动作白名单校验：只允许两种合法值，其他写法在入口就挡掉
            if req.action not in ("approved", "rejected"):
                raise HTTPException(status_code=400, detail="action 只能是 approved 或 rejected")
            # 更新审核状态；返回 False = 记录不存在或已被处理（防止重复审核）
            if not ctx.pg.update_review_status(review_id, req.action, user["user_id"], req.note):
                raise HTTPException(status_code=404, detail="待审核项不存在或已处理")

            answer_id = None
            # 仅"通过 + 填了标准答案"才反哺：驳回或留空都不进标准答案库
            if req.action == "approved" and req.standard_answer.strip():
                answer_id = ctx.pg.add_standard_answer(
                    question=req.question.strip(),
                    answer=req.standard_answer.strip(),
                    source_review_id=review_id,   # 保留来源：将来能追溯这条答案出自哪次审核
                )
            return {"review_id": review_id, "action": req.action, "standard_answer_id": answer_id}

    # 【什么】启动 API 服务（uvicorn）
    # 【作用】按配置的 APP_HOST / APP_PORT 监听，阻塞当前进程
    def run(self):
        """启动 API 服务。"""
        import uvicorn   # 延迟 import：不用 serve 功能时不需要安装/加载 uvicorn

        s = get_settings()
        _logger.info(f"API服务启动: {s.app_host}:{s.app_port}")
        # 直接传 app 对象（非 import 字符串）：uvicorn 不做 reloader，配置全由本进程控制
        uvicorn.run(self.app, host=s.app_host, port=s.app_port)
