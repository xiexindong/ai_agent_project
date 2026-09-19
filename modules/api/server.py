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
            is_new_session = not req.session_id
            session_id = req.session_id or str(uuid.uuid4())

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
                    # ==================== asyncio.to_thread 详解 ====================
                    # 【先看被包装的那个东西】
                    #   ctx.agent_graph.run 是 AgentGraph 的方法（graph.py:256），
                    #   注意它是普通的 **def，不是 async def** —— 纯同步阻塞函数。
                    #   它内部走 LangGraph 的 _graph.invoke（graph.py:273），串行执行
                    #   prepare → reason → act → answer 这条链，每一次 reason / act 都可能
                    #   往 DeepSeek / 智谱 / Tavily 发一次 HTTP 请求。一轮 ReAct 平均几秒，
                    #   多轮下来整段执行常常十几到几十秒。
                    #
                    # 【如果这里直接写 result = ctx.agent_graph.run(...) 会怎样】
                    #   这一行现在处在 async def event_generator() 里，跑在 uvicorn 的
                    #   **事件循环线程**上。同步阻塞几十秒 = 事件循环停止转动 = 整个服务卡死：
                    #     - 同一个 uvicorn 进程里其他所有并发请求全部停摆
                    #       （连 /health 都 ping 不通，前端仪表那几个接口也一样）
                    #     - 正在进行的其他 SSE 流也跟着冻住
                    #   而 uvicorn 默认单 worker，服务就等于"一人聊天，全员陪等"。
                    #   这是 async 框架里最常见的踩坑：把同步阻塞代码混进协程里。
                    #
                    #   【为什么 to_thread 能救】
                    #   asyncio.to_thread(f, *args) 做两件事：
                    #     ① 把 f(*args) 提交给 asyncio 默认的 ThreadPoolExecutor 执行
                    #        （默认池大小 min(32, cpu_count + 4)）
                    #     ② 返回一个 coroutine，await 它就能拿到 f 的返回值
                    #   关键在于：① 发生在**别的线程**，事件循环线程只是"登记了一个任务"就
                    #   立刻让出控制权去处理别人的请求了。await 挂起的这段时间里，
                    #   事件循环是空转可用的。
                    #
                    #   它等价于下面这段老写法（Python 3.9 之前没有 to_thread 就得这么写）：
                    #       loop = asyncio.get_running_loop()
                    #       result = await loop.run_in_executor(
                    #           None,  # None = 用默认线程池
                    #           lambda: ctx.agent_graph.run(req.query, session_id, user_id)
                    #       )
                    #   to_thread 只是它的语法糖：不用自己搞 lambda/partial 包一层，
                    #   参数直接跟在后面传，而且**支持关键字参数**：to_thread(f, a, k=v)。
                    #
                    #   【这里第 2/3/4 个参数的作用】
                    #   to_thread 第一个参数是被调函数，后面的是原样转发给它参数。
                    #   所以下面这一句完全等价于在线程里跑：
                    #       ctx.agent_graph.run(req.query, session_id, user_id)
                    #   → 对应 graph.py:256 的签名 run(self, query, session_id, user_id)
                    #     （self 已经由对象绑定好了，所以这里只传 3 个）
                    #
                    #   【依赖 invoke 回来的 result 是什么】
                    #   一个 dict，下面 326/337/338/344 行会从中取这些 key（来源 graph.py:273-279）：
                    #     final_answer    最终答案文本
                    #     tool_call_logs  本轮所有工具调用记录（name/arguments/result/success）
                    #     iteration       实际推理轮数
                    #     llm_summary     一句话摘要
                    #   LangGraph 的 invoke 本身就是同步的 —— 这也是为什么这里要用
                    #   to_thread 包它，而不是直接 await。
                    #   （LangGraph 也有异步版本 ainvoke，但那要求 Agent 里每个节点
                    #    都改成 async def 才行，本项目没走那条路。）
                    #
                    #   【为什么多线程管用 —— GIL 的问题】
                    #   Agent 慢在**网络等待**（等 LLM 回包），不在算算术。
                    #   requests / SDK 底层 socket recv 时会释放 GIL，所以另一个线程能继续跑，
                    #   事件循环线程也就活了。→ 这种 IO 密集场景，to_thread 是正解。
                    #
                    #   【反例：如果它是 CPU 密集，多线程救不了】
                    #   假设 Agent 里有段纯 Python 密集计算（大量向量运算、正则回溯），
                    #   CPython 的 GIL 保证同一时刻只有一个线程执行字节码，移到线程一样堵 ——
                    #   那种情况得换进程池：
                    #       from concurrent.futures import ProcessPoolExecutor
                    #       await loop.run_in_executor(ProcessPoolExecutor(), cpu_job)
                    #   （代价：进程池传参要能 pickle，graph/ctx 这种塞对象的东西传不过去）
                    #
                    #   【一个当前实现的局限】
                    #   asyncio.to_thread 提交出去的任务**不可取消**：哪怕客户端已经断开
                    #   （前端 AbortController 取消 fetch，app.js:337/348），这个线程里的
                    #   LLM 调用还是会跑完才结束，白白烧掉一次 API 费用。
                    #   要做到"断开即中断"，得用可中断的方案（async def + httpx 异步客户端
                    #   + task.cancel），本项目目前没做。
                    #   （简单起见此处不做补偿：单次请求失败不会污染会话状态，
                    #    会话已在上面 223-226 行创建/校验过，工具记录也已在 229 行 reset 清零。）
                    result = await asyncio.to_thread(
                        ctx.agent_graph.run, req.query, session_id, user_id
                    )
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

            # ━━━ 为什么这里不是 return JSON，而是 StreamingResponse？━━━
            # 普通 return {"answer": ...} 是"算完一次性吐出"：Agent 跑 20 秒期间，
            # 前端只能干等着转圈，什么都看不到。
            # StreamingResponse 则把这个 HTTP 响应**保持不关闭**，ASGI 服务器一边从生成器里
            # 取一帧、一边就写进 TCP 发出去。前端能在 Agent 还在跑的时候就收到第一批事件。
            #
            # 【它就是 SSE（Server-Sent Events）】
            #   没有升级协议、没有握手，就是一个 Content-Type 特殊的"超长 HTTP 响应"：
            #     Content-Type: text/event-stream
            #   体格式是若干次 "event: xxx\ndata: {...}\n\n"（由上面 231 行的 sse_event 拼），
            #   每次后面跟一个空行表示这一帧结束。
            #
            # 【为什么选 SSE 而不是 WebSocket】
            #   本场景只有"服务端 → 客户端"这一个方向的推送就够了（第二个方向走的是普通
            #   HTTP 请求本身），而 SSE 相比 WebSocket 有两点省心：
            #     ① 走普通 HTTP，不需要 WS 握手和协议升级，nginx/网关不用额外配置
            #     ② /chat 的请求体是 POST JSON（{query, session_id}），SSE 照常带 body；
            #        WebSocket 只能在握手那一步传，发 JSON body 得自己再包一层协议
            #   前端侧对应 app.js:338 —— 用 fetch + ReadableStream 逐块读，同样不用引任何库。
            #
            #   → 反过来记：真正需要"客户端高频主动发消息"（协同编辑、在线游戏）时才上 WebSocket。
            return StreamingResponse(
                # 【注意这里的括号】必须写 event_generator()，不能写成 event_generator。
                #   event_generator 本身是 async def，直接调用才得到"异步生成器对象"
                #   （一个可被 async for 迭代的东西）；只写名字传过去的是个函数对象，
                #   Starlette 遍历它会 TypeError。
                #   同理也不需要 await —— await 会把生成器整个抽干成一个结果，
                #   那就退化成"一次性返回"了，流式效果全丢。
                #
                # 【谁在驱动它】Starlette 拿到这个 async generator 后逐个 await 取值。
                #   所以生成器内部的 await asyncio.sleep(0.02)（342 行）能真正把控制权
                #   交还事件循环 —— 这一瞬间同一个 uvicorn 进程还能去处理别人的 HTTP 请求，
                #   不会因为你在"打字"就把整个服务堵死。这正是 313 行要用 asyncio.to_thread
                #   把同步阻塞的 Agent 推理丢去线程池的同一个道理。
                event_generator(),

                # 【这一行决定了它是 SSE】告诉浏览器"这是事件流，别按普通响应处理"。
                #   前端 fetch 拿到 headers 后，response.body 才是一个可读流（app.js:352 getReader）。
                media_type="text/event-stream",

                # 两个响应头，缺一个流式就可能变成"卡住不动"：
                #
                # Cache-Control: no-cache
                #   禁止浏览器和中间各级缓存把这份响应存下来。
                #   流式响应的内容是一次性的、随时间推进的，被缓存毫无意义且会串数据。
                #
                # X-Accel-Buffering: no
                #   专治 nginx 反向代理。X-Accel-Buffering 是 nginx 的**私有响应头**，不是 HTTP 标准。
                #   nginx 默认会攒够 proxy_buffer_size 再往下游发；设为 no 就是明确告诉它
                #   "别攒，收到就立刻转发"。
                #   症状很有迷惑性：本地直连 uvicorn 一切正常，一挂上 nginx 就发现前端不再
                #   逐条冒字、而是等几十秒后一次性刷出来 —— 不是后端慢，是中间被 buffer 住了。
                #   （Apache 对应的是关掉 mod_proxy 缓冲，或用 "ProxyPass ... flushpackets=on"）
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
