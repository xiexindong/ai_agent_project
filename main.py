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

环境要求：
    1. PostgreSQL + pgvector（docker run） — 存对话记忆与业务表
    2. Milvus（docker standalone）          — 存知识库文档向量
    3. .env 配置完成（参考 .env.example）    — 密钥、连接串、模型参数
"""
# ---- 标准库导入（Python 自带，无需 pip 安装）----
import sys          # 与 Python 解释器打交道：读取命令行参数、sys.executable（当前 Python 路径）
import uuid         # 生成全局唯一 ID：每次新对话用 uuid4() 生成不重复的 session_id
import argparse     # 命令行参数解析器：把 `python main.py chat -u zhangsan` 解析成程序能用的对象
from pathlib import Path  # 面向对象的路径工具：跨平台拼接/定位文件路径（Windows 反斜杠、Linux 正斜杠自动适配）

# 将本文件所在目录加入 sys.path，确保 `from modules.xxx import yyy` 可正常导入
# 类比 Node.js 的 NODE_PATH 或 tsconfig paths，让包查找路径包含项目根
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- 项目内模块导入（注意：重型组件都放在各 cmd_xxx 函数内部再导入，见下方说明）----
from config import get_settings, setup_langsmith
from modules.common.logger import get_logger

# 全局日志器（loguru 单例，控制台彩色 + 文件按天轮转）
_logger = get_logger("main")


def cmd_chat(args):
    """
    交互式命令行对话。

    装配流程：把 6 大组件逐个初始化，注入 ReActAgent，再交给 LangGraph 编排。
    组件之间通过构造函数注入（依赖注入），全部为单例，整个 CLI 生命周期复用同一实例。
    """
    # ━━━ 函数内导入（延迟导入 / lazy import）━━━
    # 为什么不写在文件顶部？这些模块依赖重（连数据库、加载模型），
    # 只有真正执行 chat 命令时才需要；写在函数内可让 skills/--help 等轻量命令秒启动，
    # 也能避免"只跑 web-demo 却被强制要求 PG 在线"。类比 JS 的动态 import()。
    from modules.agent.graph import AgentGraph
    from modules.agent.react_agent import ReActAgent
    from modules.agent.llm import LLMClient
    from modules.tools.tool_registry import ToolRegistry
    from modules.skills import SkillRegistry, get_default_skills
    from modules.memory.memory_manager import MemoryManager
    from modules.memory.pg_manager import PGManager

    _logger.info("启动交互式对话模式...")
    s = get_settings()  # 全局配置单例（.env 中的密钥、端口、阈值等都在这里）

    # ━━━ 第①步：基础连接（PG 数据库 + LLM 大模型客户端）━━━
    # LLMClient 是 DeepSeek 的单例客户端，封装了普通对话 / Function Calling / 结构化输出三种调用
    llm = LLMClient()
    # PGManager 是 PostgreSQL 单例，管 4 张表（会话/记忆/工具日志/审核队列）+ pgvector 向量检索
    pg = PGManager()
    # MemoryManager 把短期原文（最近 3 轮）和长期摘要（语义向量召回）组合成上下文，超 token 自动压缩
    memory = MemoryManager(llm_client=llm)

    # ━━━ 第②步：工具注册（Function Calling 工具）━━━
    # 先注册本地工具（calculator 计算器 + data_query 业务查询），它们直接在进程内执行
    tool_registry = ToolRegistry()
    tool_registry.register_batch(tool_registry.get_default_tools())

    # ━━━ 第③步：技能注册（Skill = SOP 作业流程 + 绑定工具）━━━
    # 技能在 ReAct 开始前按用户意图匹配，命中则注入专属 SOP 引导 LLM 按标准步骤调用工具
    # 与 Function Call 的分工：Skill 管"该怎么做"（流程），Function Call 管"具体调用"（执行）
    skill_registry = SkillRegistry()
    skill_registry.register_batch(get_default_skills())
    _logger.info(f"技能已加载: {len(skill_registry.list_skills())} 个")

    # ━━━ 第④步：MCP 外部工具接入（可降级）━━━
    # MCP 用子进程 + JSON-RPC stdio 通信，把外部工具（联网搜索/报时）包装成和本地工具一样的接口
    # MCPToolAdapter 是适配器模式：Agent 和 ToolRegistry 完全不感知工具在本地还是远程
    try:
        from modules.mcp.mcp_client import MCPClient, MCPToolAdapter
        mcp = MCPClient(
            server_command=[sys.executable, "-m", "modules.mcp.mcp_server"],
            name="enterprise-mcp",
        )
        mcp.connect()
        for tool_info in mcp.discover_tools():
            tool_registry.register(MCPToolAdapter(mcp, tool_info))
        _logger.info(f"MCP工具已接入: {len(tool_registry.list_tools())} 个工具")
    except Exception as e:
        # 全链路降级第 1 道：MCP 挂了不影响本地工具，Agent 仍可用 calculator / data_query 工作
        _logger.warning(f"MCP降级（本地工具仍可用）: {e}")
        mcp = None

    # ━━━ 第⑤步：RAG 知识库接入（可降级）━━━
    # RAGChain 串联"加载→切片→向量化→Milvus 粗召回→BGE 精排"全链路
    rag = None
    try:
        from modules.rag.rag_chain import RAGChain
        rag = RAGChain()
    except Exception as e:
        # 全链路降级第 2 道：Milvus 挂了 Agent 仍可纯靠模型 + 记忆对话，只是没有知识库参考
        _logger.warning(f"RAG降级（Milvus不可用）: {e}")

    # ━━━ 第⑥步：组装 ReActAgent + LangGraph 状态图 ━━━
    # ReActAgent 是核心推理器，6 个组件全部注入：
    #   llm（大脑）+ tool_registry（工具）+ skill_registry（技能）
    #   + rag（知识库）+ memory（记忆）→ 组成完整的 Agent
    react_agent = ReActAgent(
        llm_client=llm,
        tool_registry=tool_registry,
        rag_chain=rag,
        memory_manager=memory,
        skill_registry=skill_registry,
    )
    # AgentGraph 把 ReActAgent 的四步（prepare→reason↔act→answer）编排成 LangGraph 状态图
    # 图的运行入口是 graph.run(query, session_id, user_id)，返回最终状态的 dict
    agent_graph = AgentGraph(react_agent)

    # ━━━ 对话会话初始化 ━━━
    # user_id：命令行 --user 传入，没传则用默认 "cli-user"（`x or 默认值` 是 Python 常用兜底写法）
    user_id = args.user or "cli-user"
    # session_id：--session 没传时（空串为假值），用 uuid4 随机生成，保证每次新会话全局唯一
    session_id = args.session or str(uuid.uuid4())
    # 在 PG 的 chat_session 表登记一条会话记录（后续每轮对话通过 session_id 关联到它）
    pg.create_session(session_id, user_id, title="CLI对话")

    print("\n" + "=" * 60)
    print("  企业级 AI Agent 智能助手（交互模式）")
    print("  输入问题开始对话 | 输入 quit/exit 退出 | 输入 new 开启新会话")
    print("=" * 60 + "\n")

    # ━━━ 主对话循环 ━━━
    # 每轮循环 = 一次完整的 Agent 推理：prepare→reason↔act→answer→save
    # while True 是 REPL 模式（读取-求值-输出循环），和 Node 的 readline、Python 解释器本身同款
    while True:
        # input() 阻塞等待用户打字回车；提示符里显示用户ID和会话ID前8位方便辨认
        try:
            query = input(f"[{user_id} | {session_id[:8]}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D（EOF）或 Ctrl+C（中断）时优雅退出，而不是抛红色堆栈
            print("\n再见！")
            break

        # ---- 三条内置"指令"，不发给 LLM，本地直接处理 ----
        if not query:                  # 直接回车（空输入）：跳过本轮，继续等输入
            continue
        if query.lower() in ("quit", "exit"):  # 退出指令（大小写不敏感）
            print("再见！")
            break
        # "new" 指令：重置工具调用计数器（防死循环的去重记录归零），创建全新会话
        if query.lower() == "new":
            tool_registry.reset_session(session_id)
            session_id = str(uuid.uuid4())
            pg.create_session(session_id, user_id, title="CLI对话")
            print(f"已开启新会话: {session_id[:8]}\n")
            continue

        # 每轮提问前重置工具调用去重记录，允许同一工具在新问题中再次被调用
        tool_registry.reset_session(session_id)

        # ━━━ 执行 Agent（核心调用）━━━
        # agent_graph.run() 内部执行 LangGraph 状态图：
        #   prepare: 安全检查 + 技能匹配 + 记忆召回 + RAG 召回
        #   reason:  LLM 决策（直接回答 or 通过 Function Calling 调工具）
        #   act:     执行工具（本地 Tool 或远程 MCP），观察结果回到 reason 形成循环
        #   answer:  生成完整答案 + 一句话摘要
        #   save:    持久化到 PG（原文+摘要+向量+工具日志），状态序列化供断点续跑
        print("思考中...\n")
        result = agent_graph.run(query, session_id, user_id)

        # ━━━ 输出结果 ━━━
        # result 是 LangGraph 跑完后返回的最终状态 dict，键值由 modules/agent/state.py 定义
        # dict.get("键", 默认值)：键不存在时返回默认值而不报错，比 result["final_answer"] 更安全
        answer = result.get("final_answer", "（无回答）")
        print(f"\n{'─' * 50}")
        print(f"回答: {answer}")
        print(f"{'─' * 50}")
        print(f"摘要: {result.get('llm_summary', '')}")
        # 技能命中情况：命中则显示技能名（如"数学计算"），未命中显示通用流程
        active_skill = result.get("active_skill", "")
        print(f"激活技能: {active_skill or '（通用助手流程）'}")
        # 推理轮次和工具调用次数，方便观察 Agent 的"思考深度"
        print(f"推理轮次: {result.get('iteration', 0)} | "
              f"工具调用: {len(result.get('tool_call_logs', []))} 次")
        # 历史记忆超 4000 token 时会自动用 LLM 压缩，这里提示用户
        if result.get("memory_compressed"):
            print("（历史记忆已自动压缩）")
        print()

    # ━━━ 资源清理 ━━━
    # 关闭 MCP 子进程连接和 PG 数据库连接池
    if mcp:
        mcp.disconnect()
    pg.close()


def cmd_serve(args):
    """
    启动 FastAPI API 服务。

    APIServer 内部完成和 cmd_chat 一样的组件装配（LLM/PG/Memory/Tools/Skills/RAG），
    然后挂载 7 个 REST 路由：/chat /history /knowledge/ingest /knowledge/stats
    /tools/logs /reviews/pending /skills，带三级权限鉴权 + 滑动窗口限流。
    """
    from modules.api.server import APIServer
    _logger.info("启动 API 服务模式...")
    server = APIServer()
    server.run()


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
        # 入库强依赖 Milvus（向量必须有地方写），这里不像 chat 那样降级，直接终止并提示
        _logger.error(f"RAG模块初始化失败（请确认Milvus运行）: {e}")
        return  # 从函数返回 = 结束本命令

    # Path 对象把字符串路径包装成带方法的对象：exists()/is_dir() 等，跨平台通用
    path = Path(args.path)
    if not path.exists():
        _logger.error(f"路径不存在: {path}")
        return

    # 目录走批量入库，单文件走单文档入库，两者最终都拆成分片写入 Milvus
    if path.is_dir():
        count = rag.ingest_directory(str(path))
        print(f"批量入库完成: {count} 个分片")
    else:
        count = rag.ingest_document(str(path))
        print(f"入库完成: {count} 个分片")

    stats = rag.get_knowledge_stats()
    print(f"知识库总量: {stats}")
    pg.close()


def cmd_stats(args):
    """查看知识库统计（Milvus 集合中的分片总数、来源文件数等）。"""
    from modules.rag.rag_chain import RAGChain
    # 纯只读命令：Milvus 不可用时仅打印错误，不影响其他功能
    try:
        rag = RAGChain()
        stats = rag.get_knowledge_stats()
        print(f"知识库统计: {stats}")
    except Exception as e:
        _logger.error(f"RAG不可用: {e}")


def cmd_mcp_server(args):
    """
    独立启动 MCP Server。

    以 stdio 模式运行，通过 JSON-RPC 2.0 对外暴露工具（web_search/get_current_time 等）。
    供本项目的 MCPClient 拉起，也可被其他支持 MCP 协议的 Agent 客户端调用。
    """
    from modules.mcp.mcp_server import MCPServer
    print("MCP Server 启动中（stdio模式）...")
    MCPServer().run()


def cmd_web_demo(args):
    """
    启动前端预览服务（Mock 模式）。

    不需要 PostgreSQL / Milvus / DeepSeek，全部返回 Mock 数据，
    用于快速预览前端交互界面和对话流程。
    """
    from demo_server import run_demo
    run_demo()


def cmd_skills(args):
    """
    查看已注册技能清单，可选测试意图匹配。

    不加载 LLM / 数据库，只初始化 SkillRegistry，用于快速查看技能配置和调试匹配效果。
    加 --query 参数时，会模拟一次意图匹配并输出命中技能的 SOP 上下文。
    """
    from modules.skills import SkillRegistry, get_default_skills
    skill_registry = SkillRegistry()
    skill_registry.register_batch(get_default_skills())  # 注册内置 4 个技能
    # describe_all() 把技能清单拼成人类可读文本（名称/触发词/绑定工具/优先级）
    print("\n" + "=" * 60)
    print("  技能（Skills）清单")
    print("=" * 60)
    print(skill_registry.describe_all())
    print("-" * 60)
    # --query 参数：模拟一次意图匹配，预览技能 SOP 注入效果（不实际调 LLM）
    # 空串是假值，没传 -q 时整个 if 跳过，只打印清单
    if args.query:
        match = skill_registry.match(args.query)
        if match:
            print(f"\n提问「{args.query}」匹配结果:")
            print(match.skill.format_context(match))
        else:
            print(f"\n提问「{args.query}」未命中任何技能，将走通用助手流程")
    print()


def main():
    """
    命令行入口。

    使用 argparse 注册 7 个子命令，根据用户输入分发到对应的 cmd_xxx 函数。
    类比 Express 的 app.get('/chat', handler) — 路由 + 处理器模式。
    """
    # 初始化 LangSmith 链路追踪（可选，未配置密钥时自动跳过）
    setup_langsmith()

    # 【这行在干嘛？】创建一个「命令行菜单/前台接待员」。
    # 用户在终端敲的是一串纯文本，例如：python main.py chat -u zhangsan
    # 程序没法直接理解这串字，argparse 就像前台接待，帮你把它拆成结构化信息：
    #   要办什么业务？→ chat        （子命令，在下方 add_subparsers 处注册）
    #   带了什么参数？→ -u zhangsan（选项参数，在下方 chat 子命令处注册）
    # 拆好后统一放进 args 对象，程序用 args.command / args.user 读取。
    # JavaScript 类比：功能等同于 Node 生态的 commander.js / yargs：
    #   program.command('chat').option('-u, --user <id>').parse(process.argv)
    #
    # 两个配置项解释：
    #   description      —— 程序「自我介绍」，用户执行 python main.py -h 时显示在帮助页顶部
    #   formatter_class  —— 帮助文本的「排版规则」，决定多行 description 的换行/缩进怎么处理
    #
    #     【为什么要这个？】argparse 默认会把多行文字"压扁"成一长行，
    #     你写好的换行排版 -h 一看就全毁了。RawDescriptionHelpFormatter 的作用
    #     就是"原样保留"——你写啥样就显示啥样，不动手脚。
    #
    #     【对比例子】假设 description 是三行文字：
    #         description="第一行\n第二行\n第三行"
    #
    #     ✗ 不加 formatter_class（默认 HelpFormatter）：
    #         第一行 第二行 第三行     ← 换行被吃掉，挤成一坨
    #
    #     ✓ 加 RawDescriptionHelpFormatter：
    #         第一行                   ← 换行保住，排版不毁
    #         第二行
    #         第三行
    #
    #     【本项目说明】当前 description 是单行，效果暂不明显；这是项目标准写法，
    #     一旦未来把顶部 docstring（5 层架构 + 启动方式）传进 description，
    #     多行排版就能保住，用户 -h 时看到的是整齐的多行说明而非糊成一坨。
    parser = argparse.ArgumentParser(
        description="企业级 AI Agent 智能助手平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 创建"子命令"容器：dest="command" 表示解析后用户输入的子命令名会存到 args.command
    # 例：`python main.py chat` → args.command == "chat"，本函数末尾据此分发
    # 类比 npm scripts：main 是程序名，chat/serve/ingest 是后面跟的"动作名"
    sub = parser.add_subparsers(dest="command", help="可用命令")

    # ━━━ 子命令注册（每个子命令对应一个 cmd_xxx 处理函数）━━━
    # 名词扫盲（argparse 两种参数）：
    #   位置参数：不加横杠、按出现顺序取值，如 ingest 后面的文件路径（必填）
    #   选项参数：带 -- 前缀、可省略、有默认值，如 --user；单字母 -u 是它的简写
    # add_parser("名字", help="...")：声明一个子命令，help 显示在 -h 帮助里

    # chat: 交互式 CLI 对话（核心入口，完整装配 Agent 6 大组件）
    p_chat = sub.add_parser("chat", help="交互式对话")
    p_chat.add_argument("--user", "-u", default="cli-user", help="用户ID")
    #                    ↑长选项    ↑短选项 ↑没传时的默认值
    p_chat.add_argument("--session", "-s", default="", help="会话ID（留空自动生成）")

    # serve: 启动 FastAPI + 前端，对外提供 REST API（无额外参数）
    sub.add_parser("serve", help="启动API服务")

    # ingest: 知识库文档入库（文件 → 切片 → 向量化 → Milvus）
    p_ingest = sub.add_parser("ingest", help="知识库文档入库")
    p_ingest.add_argument("path", help="文件或目录路径")  # 不带横杠 = 位置参数，必填

    # stats: 查看 Milvus 知识库统计
    sub.add_parser("stats", help="知识库统计")

    # mcp-server: 独立启动 MCP 工具服务（stdio + JSON-RPC）
    sub.add_parser("mcp-server", help="独立启动MCP Server")

    # web-demo: 前端预览（零依赖，全 Mock 数据）
    sub.add_parser("web-demo", help="启动前端预览服务（Mock模式，不依赖数据库）")

    # skills: 查看技能清单 / 测试意图匹配（零依赖调试工具）
    p_skills = sub.add_parser("skills", help="查看技能清单/测试技能匹配")
    p_skills.add_argument("--query", "-q", default="", help="输入一句提问，测试技能意图匹配")

    # 真正解析命令行：读取 sys.argv（终端输入的原始字符串列表），产出 Namespace 对象
    # 例：`python main.py chat -u zhangsan`
    #     → Namespace(command='chat', user='zhangsan', session='')
    args = parser.parse_args()

    # 命令分发字典：子命令名 → 处理函数
    # 类比 Express 路由表：{ "/chat": chatHandler, "/serve": serveHandler, ... }
    commands = {
        "chat": cmd_chat,
        "serve": cmd_serve,
        "ingest": cmd_ingest,
        "stats": cmd_stats,
        "mcp-server": cmd_mcp_server,
        "web-demo": cmd_web_demo,
        "skills": cmd_skills,
    }

    # 按用户输入的子命令分发；无命令时打印帮助
    # 注意这里写的是函数名（不带括号）= 把函数本身存进字典；带括号 cmd_chat() 会立即执行
    if args.command in commands:
        commands[args.command](args)  # 查表取出函数并调用，把解析好的 args 传进去
    else:
        # args.command 为 None（只输入 python main.py）：打印用法提示，避免程序静默退出
        parser.print_help()
        print("\n快速开始：")
        print("  python main.py chat          # 交互式对话")
        print("  python main.py serve         # 启动API服务")
        print("  python main.py ingest ./docs  # 知识库入库")


# Python 入口守卫：只有"直接运行本文件"时才执行 main()
if __name__ == "__main__":
    # Python 惯用写法：仅在直接运行本文件时执行 main()，被 import 时不执行
    # 类比 Node.js 的 if (require.main === module) 入口守卫
    main()
