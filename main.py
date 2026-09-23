"""
企业级 AI Agent 智能助手平台 - 项目入口
=========================================

架构总览（5 层）：
    入口层（main.py）        ← 你正在看的文件，负责命令分发
    Agent 层（LangGraph）    ← 大脑：编排 prepare→reason↔act→answer 五节点循环
    能力层（Skills + Tools） ← 技能(SOP流程) + 工具(Function Calling) + MCP(外部工具)
    知识层（RAG + Memory）   ← Milvus 知识库 + PostgreSQL 记忆库
    基础层（common）         ← 日志、异常、安全校验、token 计数

启动方式：
    # 交互式对话（CLI）
    python main.py chat

    # 启动 API 服务（FastAPI）
    python main.py serve

    # 知识库文档入库（支持单文件或整个目录）
    python main.py ingest <文件路径>
    python main.py ingest <目录路径>

    # 查看知识库统计
    python main.py stats

    # MCP Server 独立启动（供其他 Agent 通过 JSON-RPC 调用工具）
    python main.py mcp-server

    # 查看技能清单 / 测试意图匹配
    python main.py skills
    python main.py skills -q "帮我算一下 3*7"

    # 前端预览（不依赖数据库，全 Mock 数据）
    python main.py web-demo

    # 离职数据处置（跨库对账：MySQL 离职名单 → PG 会话归档/清理）
    python main.py offboard
    python main.py offboard --apply

环境要求：
    1. PostgreSQL + pgvector（docker run） — 存对话记忆与业务表
    2. Milvus（docker standalone）          — 存知识库文档向量
    3. .env 配置完成（参考 .env.example）    — 密钥、连接串、模型参数
"""
import sys
import uuid
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_settings, setup_langsmith
from modules.common.logger import get_logger

_logger = get_logger("main")


# 【什么】chat 子命令：起一个交互式 CLI 会话
# 【作用】 装配 LLM/记忆/工具/技能/RAG/路由六类组件，交给 LangGraph 跑 ReAct 循环
def cmd_chat(args):
    """
    交互式命令行对话。

    装配流程：把 6 大组件逐个初始化，注入 ReActAgent，再交给 LangGraph 编排。
    组件之间通过构造函数注入（依赖注入），全部为单例，整个 CLI 生命周期复用同一实例。
    """
    # 【什么】函数内延迟导入
    # 【作用】 只有执行 chat 才加载这些重依赖，跑 stats/skills 等子命令不会被拖慢
    from modules.agent.graph import AgentGraph
    from modules.agent.react_agent import ReActAgent
    from modules.agent.llm import LLMClient
    from modules.tools.tool_registry import ToolRegistry
    from modules.skills import SkillRegistry, get_default_skills
    from modules.memory.memory_manager import MemoryManager
    from modules.memory.pg_manager import PGManager

    _logger.info("启动交互式对话模式...")
    s = get_settings()

    llm = LLMClient()                        # LLM 客户端：封装模型调用与重试
    pg = PGManager()                         # PG：会话与记忆存储
    memory = MemoryManager(llm_client=llm)   # 记忆管理：摘要 + 向量召回

    tool_registry = ToolRegistry()           # 工具注册表
    tool_registry.register_batch(tool_registry.get_default_tools())   # 注册内置工具

    skill_registry = SkillRegistry()         # 技能注册表（SOP 流程）
    skill_registry.register_batch(get_default_skills())               # 注册内置技能
    _logger.info(f"技能已加载: {len(skill_registry.list_skills())} 个")

    # ------------------------------------------------------------------
    # 【MCP 技能网关 + Tool-RAG + 分层路由】三件套在这里装配
    #   改造前：discover_tools() 全量注册 → 每轮把全部工具 schema 发给模型
    #   改造后：网关只登记目录（工具描述进向量索引，不进 prompt），
    #           AgentRouter 每轮按语义召回 top-k，只有被召回的才连 server + 注册
    #
    #   【装配失败怎么降级】网关/索引/路由任一环节出错，router 保持为 None，
    #   ReActAgent 会退回「全部工具」模式，行为与改造前一致 —— 新机制是增强，不是依赖。
    # ------------------------------------------------------------------
    gateway = None   # MCP 网关：纳管远程工具目录；装配失败就一直是 None
    router = None    # 分层路由：每轮决定哪些工具进 prompt；None = 退回全量工具模式
    try:
        from modules.mcp.gateway import MCPGateway
        from modules.tools.tool_rag import ToolRAGIndex
        from modules.agent.router import AgentRouter
        from modules.common.circuit_breaker import CircuitBreaker

        if s.mcp_enabled:
            gateway = MCPGateway.from_settings(
                s,
                breaker=CircuitBreaker(s.cb_failure_threshold, s.cb_recovery_seconds),
                default_command=[sys.executable, "-m", "modules.mcp.mcp_server"],
            )
            _logger.info(f"MCP 网关已就绪: servers={gateway.server_names}")

        tool_index = None
        if s.tool_rag_enabled:
            tool_index = ToolRAGIndex(top_k=s.tool_rag_top_k, min_score=s.tool_rag_min_score)
            # 索引内容 = 本地工具 + 远程目录（只有描述，不是调用契约）
            tool_index.index_tools(tool_registry.list_tools())
            if gateway:
                tool_index.index_tools(gateway.catalog())
            _logger.info(f"Tool-RAG 索引已建立: {tool_index.stats()}")

        router = AgentRouter(
            skill_registry=skill_registry,
            tool_index=tool_index,
            gateway=gateway,
            top_k=s.tool_rag_top_k,
            max_tools=s.max_tools_per_turn,
        )
    except Exception as e:
        _logger.warning(f"技能网关/路由降级（按全部工具运行）: {e}")
        router = None

    rag = None   # RAG 链；Milvus 不可用时保持 None，Agent 走无知识库模式
    try:
        from modules.rag.rag_chain import RAGChain
        rag = RAGChain()
    except Exception as e:
        _logger.warning(f"RAG降级（Milvus不可用）: {e}")

    # 【什么】组装 Agent 本体并包一层 LangGraph
    # 【作用】 依赖注入后由 AgentGraph 负责 prepare→reason↔act→answer 的状态流转
    react_agent = ReActAgent(
        llm_client=llm,
        tool_registry=tool_registry,
        rag_chain=rag,
        memory_manager=memory,
        skill_registry=skill_registry,
        router=router,
    )
    agent_graph = AgentGraph(react_agent)

    user_id = args.user or "cli-user"                # --user 未给时的默认用户
    session_id = args.session or str(uuid.uuid4())   # --session 未给时随机生成一个会话号
    pg.create_session(session_id, user_id, title="CLI对话")   # 落一条会话记录，后续每轮按它存消息

    print("\n" + "=" * 60)
    print("  企业级 AI Agent 智能助手（交互模式）")
    print("  输入问题开始对话 | 输入 quit/exit 退出 | 输入 new 开启新会话")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input(f"[{user_id} | {session_id[:8]}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("再见！")
            break
        if query.lower() == "new":
            tool_registry.reset_session(session_id)
            session_id = str(uuid.uuid4())
            pg.create_session(session_id, user_id, title="CLI对话")
            print(f"已开启新会话: {session_id[:8]}\n")
            continue

        tool_registry.reset_session(session_id)

        print("思考中...\n")
        result = agent_graph.run(query, session_id, user_id)

        answer = result.get("final_answer", "（无回答）")
        print(f"\n{'─' * 50}")
        print(f"回答: {answer}")
        print(f"{'─' * 50}")
        print(f"摘要: {result.get('llm_summary', '')}")
        active_skill = result.get("active_skill", "")
        print(f"激活技能: {active_skill or '（通用助手流程）'}")
        print(f"推理轮次: {result.get('iteration', 0)} | "
              f"工具调用: {len(result.get('tool_call_logs', []))} 次")
        if result.get("memory_compressed"):
            print("（历史记忆已自动压缩）")
        print()

    # 网关负责收尾它拉起的所有 MCP 子进程（改造前这里是 mcp.disconnect()）
    if gateway:
        gateway.shutdown()
    pg.close()


# 【什么】serve 子命令：启动 FastAPI 服务
# 【作用】 由 APIServer 内部完成同样的组件装配，再对外提供 REST 接口
def cmd_serve(args):
    """
    启动 FastAPI API 服务。

    APIServer 内部完成和 cmd_chat 一样的组件装配（LLM/PG/Memory/Tools/Skills/RAG），
    然后挂载 7 个 REST 路由：/chat /history /knowledge/ingest /knowledge/stats
    /tools/logs /reviews/pending /skills，带三级权限鉴权 + 滑动窗口限流。
    """
    from modules.api.server import APIServer
    _logger.info("启动 API 服务模式...")
    server = APIServer()   # 组件装配 + 路由挂载都在构造函数里
    server.run()           # 阻塞运行，直到 Ctrl+C


# 【什么】ingest 子命令：知识库文档入库
# 【作用】 加载 → 切片 → 向量化 → 写 Milvus，单文件或整个目录都能吃
def cmd_ingest(args):
    """
    知识库文档入库。

    入库链路：DocumentLoader 加载(TXT/MD/PDF) → TextSplitter 切片(500字/片+80字重叠)
    → EmbeddingService 智谱向量化 → Milvus insert+flush
    支持单文件或整个目录批量入库。
    """
    from modules.rag.rag_chain import RAGChain
    from modules.memory.pg_manager import PGManager

    pg = PGManager()
    try:
        rag = RAGChain()
    except Exception as e:
        _logger.error(f"RAG模块初始化失败（请确认Milvus运行）: {e}")
        return

    path = Path(args.path)
    if not path.exists():
        _logger.error(f"路径不存在: {path}")
        return

    if path.is_dir():
        count = rag.ingest_directory(str(path))   # 目录：批量入库
        print(f"批量入库完成: {count} 个分片")
    else:
        count = rag.ingest_document(str(path))    # 单文件入库
        print(f"入库完成: {count} 个分片")

    stats = rag.get_knowledge_stats()
    print(f"知识库总量: {stats}")
    pg.close()


# 【什么】stats 子命令：查看知识库统计
# 【作用】 打印 Milvus 中的分片总数与来源文件数，用来确认入库是否成功
def cmd_stats(args):
    """查看知识库统计（Milvus 集合中的分片总数、来源文件数等）。"""
    from modules.rag.rag_chain import RAGChain
    try:
        rag = RAGChain()
        stats = rag.get_knowledge_stats()
        print(f"知识库统计: {stats}")
    except Exception as e:
        _logger.error(f"RAG不可用: {e}")   # Milvus 没起时只记一条 error，不让命令崩


# 【什么】offboard 子命令：离职数据处置
# 【作用】 默认只打印预览，加 --apply 才真正归档并清理该租户的会话数据
def cmd_offboard(args):
    """离职数据处置：默认只预览，加 --apply 才真正归档与清理。"""
    import json
    from modules.enterprise import lifecycle

    if not args.apply:
        print(json.dumps(lifecycle.preview(args.tenant), ensure_ascii=False, indent=2))
        print("\n以上为预览，加 --apply 才会真正归档与清理。")
        return
    print(json.dumps(
        lifecycle.run_once(args.days or None, args.tenant), ensure_ascii=False, indent=2
    ))


# 【什么】reconcile 子命令：跨库对账
# 【作用】 按 task_id 比对 MySQL 与 PG，列出只在单边存在的号，暴露数据不一致
def cmd_reconcile(args):
    """跨库对账：按 task_id 比对 MySQL 与 PG，列出单边缺失的号。"""
    import json
    from modules.enterprise.reconcile import reconcile

    print(json.dumps(reconcile(args.hours), ensure_ascii=False, indent=2))   # args.hours = 回看时间窗


# 【什么】archive 子命令：冷热分离归档
# 【作用】 把超期会话导出成归档文件；--apply 才从 PG 删除，默认是安全的只读预览
def cmd_archive(args):
    """冷热分离：超期会话导出归档文件（--apply 才从 PG 删除）。"""
    import json
    from modules.memory.archiver import archive_session, archive_expired

    if args.session:
        print(archive_session(args.session) or "（该会话没有可归档数据）")   # 指定单会话：只导出，不删库
        return
    if not args.apply:
        print("归档后会删除 PG 数据，加 --apply 才执行；先不加 --apply 确认范围。")
        return
    print(json.dumps(archive_expired(args.days or 0), ensure_ascii=False, indent=2))


# 【什么】mcp-server 子命令：单独把本项目跑成一个 MCP 服务端
# 【作用】 以 stdio + JSON-RPC 2.0 暴露工具，给本项目的 MCPClient 或外部 Agent 客户端调用
def cmd_mcp_server(args):
    """
    独立启动 MCP Server。

    以 stdio 模式运行，通过 JSON-RPC 2.0 对外暴露工具（web_search/get_current_time 等）。
    供本项目的 MCPClient 拉起，也可被其他支持 MCP 协议的 Agent 客户端调用。
    """
    from modules.mcp.mcp_server import MCPServer
    print("MCP Server 启动中（stdio模式）...")
    MCPServer().run()


# 【什么】web-demo 子命令：起一个纯 Mock 的前端预览服务
# 【作用】 不连 PG/Milvus/DeepSeek 也能看界面与对话流程，用于演示与前端联调
def cmd_web_demo(args):
    """
    启动前端预览服务（Mock 模式）。

    不需要 PostgreSQL / Milvus / DeepSeek，全部返回 Mock 数据，
    用于快速预览前端交互界面和对话流程。
    """
    from demo_server import run_demo
    run_demo()


# 【什么】skills 子命令：查看技能清单 / 试一次意图匹配
# 【作用】 不加载 LLM 与数据库，纯本地调试技能配置与匹配效果
def cmd_skills(args):
    """
    查看已注册技能清单，可选测试意图匹配。

    不加载 LLM / 数据库，只初始化 SkillRegistry，用于快速查看技能配置和调试匹配效果。
    加 --query 参数时，会模拟一次意图匹配并输出命中技能的 SOP 上下文。
    """
    from modules.skills import SkillRegistry, get_default_skills
    skill_registry = SkillRegistry()
    skill_registry.register_batch(get_default_skills())
    print("\n" + "=" * 60)
    print("  技能（Skills）清单")
    print("=" * 60)
    print(skill_registry.describe_all())
    print("-" * 60)
    if args.query:
        match = skill_registry.match(args.query)
        if match:
            print(f"\n提问「{args.query}」匹配结果:")
            print(match.skill.format_context(match))
        else:
            print(f"\n提问「{args.query}」未命中任何技能，将走通用助手流程")
    print()


# 【什么】命令行入口：注册子命令并分发
# 【作用】 一个命令对一个 cmd_xxx 处理器，等价于 Express 的 app.get('/chat', handler)
def main():
    """
    命令行入口。

    使用 argparse 注册 7 个子命令，根据用户输入分发到对应的 cmd_xxx 函数。
    类比 Express 的 app.get('/chat', handler) — 路由 + 处理器模式。
    """
    setup_langsmith()   # 必须第一行执行：LangChain 只在启动时读一次追踪开关

    parser = argparse.ArgumentParser(
        description="企业级 AI Agent 智能助手平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,   # 保留帮助文本里的换行与缩进
    )
    sub = parser.add_subparsers(dest="command", help="可用命令")   # dest="command"：子命令名存进 args.command


    p_chat = sub.add_parser("chat", help="交互式对话")

    p_chat.add_argument("--user", "-u", default="cli-user", help="用户ID")

    p_chat.add_argument("--session", "-s", default="", help="会话ID（留空自动生成）")

    sub.add_parser("serve", help="启动API服务")

    p_ingest = sub.add_parser("ingest", help="知识库文档入库")

    p_ingest.add_argument("path", help="文件或目录路径")

    sub.add_parser("stats", help="知识库统计")

    sub.add_parser("mcp-server", help="独立启动MCP Server")

    sub.add_parser("web-demo", help="启动前端预览服务（Mock模式，不依赖数据库）")

    p_skills = sub.add_parser("skills", help="查看技能清单/测试技能匹配")

    p_skills.add_argument("--query", "-q", default="", help="输入一句提问，测试技能意图匹配")

    p_offboard = sub.add_parser("offboard", help="离职数据处置（默认预览，--apply 执行）")

    p_offboard.add_argument("--apply", action="store_true", help="真正执行归档与清理")

    p_offboard.add_argument("--days", type=int, default=0, help="保留天数，默认取 DATA_RETENTION_DAYS")

    p_offboard.add_argument("--tenant", default="", help="租户编码，留空表示全部租户")

    p_reconcile = sub.add_parser("reconcile", help="跨库对账（按 task_id 比对两库）")

    p_reconcile.add_argument("--hours", type=int, default=24, help="回看时间窗（小时）")

    p_archive = sub.add_parser("archive", help="冷热分离：超期会话导出归档文件")

    p_archive.add_argument("--apply", action="store_true", help="导出后从 PG 删除")

    p_archive.add_argument("--days", type=int, default=0, help="保留天数，默认取 SESSION_HOT_DAYS")

    p_archive.add_argument("--session", default="", help="只导出这一个会话，不删库")


    args = parser.parse_args()

    # 【什么】子命令名 → 处理器函数的路由表
    # 【作用】 分发时按 args.command 查表；没命中就打印帮助，不静默退出
    commands = {
        "chat": cmd_chat,              # 交互式对话
        "serve": cmd_serve,            # 启动 API 服务
        "ingest": cmd_ingest,          # 知识库入库
        "stats": cmd_stats,            # 知识库统计
        "mcp-server": cmd_mcp_server,  # 独立启动 MCP Server
        "web-demo": cmd_web_demo,      # 前端 Mock 预览
        "skills": cmd_skills,          # 技能清单 / 匹配测试
        "offboard": cmd_offboard,      # 离职数据处置
        "reconcile": cmd_reconcile,    # 跨库对账
        "archive": cmd_archive,        # 冷热分离归档
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        print("\n快速开始：")
        print("  python main.py chat          # 交互式对话")
        print("  python main.py serve         # 启动API服务")
        print("  python main.py ingest ./docs  # 知识库入库")


if __name__ == "__main__":
    main()
