"""
统一配置加载模块
================
从 .env 读取全局配置，提供参数校验与单例访问。
所有密钥、数据库地址、模型参数集中管理，禁止硬编码。
"""
# 【什么】配置加载依赖
# 【作用】lru_cache 做单例缓存、pydantic-settings 按 alias 读环境变量、dotenv 负责读 .env
import os
from pathlib import Path
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


_PROJECT_DIR = Path(__file__).resolve().parent   # 本项目根目录（config.py 所在目录）
_PARENT_DIR = _PROJECT_DIR.parent                # 父目录，用于找多项目共用的 .env

# 【什么】按优先级加载 .env
# 【作用】把 .env 键值注入 os.environ，Settings 才能按 alias 读到；三级兜底保证总有加载动作
if (_PROJECT_DIR / ".env").exists():
    load_dotenv(_PROJECT_DIR / ".env")      # 优先：本项目自己的 .env
elif (_PARENT_DIR / ".env").exists():
    load_dotenv(_PARENT_DIR / ".env")       # 次之：父目录里的共用 .env
else:
    load_dotenv()                           # 兜底：交给 dotenv 自己向上查找


class Settings(BaseSettings):
    """全局配置单例，集中管理所有运行参数。

    继承 BaseSettings：实例化时自动从 os.environ 按 alias 读取配置；
    模块顶部 load_dotenv 已将 .env 内容注入 os.environ。
    """

    # ---- 模型：DeepSeek 主用 / 智谱备用 + 向量 ----
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")                      # 主模型密钥；为空则 LLM 调用直接失败
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")  # 主模型接口地址
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")              # 主模型名

    zhipu_api_key: str = Field(default="", alias="ZHIPU_API_KEY")                             # 备用模型与向量模型共用的密钥
    zhipu_base_url: str = Field(default="https://open.bigmodel.cn/api/paas/v4", alias="ZHIPU_BASE_URL")  # 智谱接口地址
    embedding_model: str = Field(default="embedding-2", alias="EMBEDDING_MODEL")              # 向量模型名
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")               # 向量维度：embedding-2 固定 1024 维，改了会和 Milvus 集合维度对不上

    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")                           # 联网搜索工具密钥

    # ---- PostgreSQL：记忆库 + 业务表（pgvector 存记忆向量）----
    pg_host: str = Field(default="localhost", alias="PG_HOST")
    pg_port: int = Field(default=5432, alias="PG_PORT")                                       # PG 默认端口
    pg_user: str = Field(default="postgres", alias="PG_USER")
    pg_password: str = Field(default="postgres", alias="PG_PASSWORD")
    pg_database: str = Field(default="ai_agent", alias="PG_DATABASE")                          # 库名

    # ---- Milvus：知识库向量 ----
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")                              # Milvus 默认端口
    milvus_collection: str = Field(default="knowledge_base", alias="MILVUS_COLLECTION")       # 单集合模式下的集合名

    # ---- 多集合（一个知识库一个 Milvus 物理集合）----
    # false（默认）：所有知识库共用一个集合，库之间靠 kb_code 标量字段区分（行为与改造前一致）
    # true：每个 kb_code 一个物理集合，集合名 = 前缀 + kb_code
    milvus_multi_collection: bool = Field(default=False, alias="MILVUS_MULTI_COLLECTION")     # False=共用一个集合靠 kb_code 区分
    milvus_collection_prefix: str = Field(default="kb_", alias="MILVUS_COLLECTION_PREFIX")    # 多集合模式下的集合名前缀，如 kb_public
    milvus_default_kb: str = Field(default="public", alias="MILVUS_DEFAULT_KB")               # 未指定知识库时的默认 kb_code

    # ---- MySQL：企业权限库（用户 / 角色 / 权限 / 知识库授权）----
    mysql_host: str = Field(default="localhost", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")                                 # MySQL 默认端口
    mysql_user: str = Field(default="root", alias="MYSQL_USER")
    mysql_password: str = Field(default="", alias="MYSQL_PASSWORD")
    mysql_database: str = Field(default="ai_agent_enterprise", alias="MYSQL_DATABASE")        # 权限库名
    rbac_enabled: bool = Field(default=True, alias="RBAC_ENABLED")                            # 权限总开关：False 时全部请求走内存 Key
    rbac_cache_ttl: int = Field(default=60, alias="RBAC_CACHE_TTL")                           # 权限缓存秒数，改权限最多延迟这么久生效
    kb_acl_enabled: bool = Field(default=True, alias="KB_ACL_ENABLED")                        # 知识库隔离：关掉就不按 kb_code 过滤
    mask_enabled: bool = Field(default=True, alias="MASK_ENABLED")                            # 敏感字段脱敏总开关

    # ---- LangSmith 全链路追踪 ----
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")                     # 追踪密钥；为空则 setup_langsmith 直接跳过
    langsmith_project: str = Field(default="ai-agent-prod", alias="LANGSMITH_PROJECT")        # 上报到哪个项目
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")           # 追踪开关（旧变量名）

    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")             # 单次 LLM 调用超时秒数
    llm_max_attempts: int = Field(default=3, alias="LLM_MAX_ATTEMPTS")                        # 最多尝试几次（含首次，即最多重试 2 次）
    fallback_model: str = Field(default="", alias="FALLBACK_MODEL")                           # 主模型不可用时的备用模型名；空=不切换
    # ---- 熔断：下游持续故障时直接跳过，省下每次白等的超时 ----
    cb_failure_threshold: int = Field(default=5, alias="CB_FAILURE_THRESHOLD")     # 连续失败几次后打开
    cb_recovery_seconds: float = Field(default=30.0, alias="CB_RECOVERY_SECONDS")  # 打开后多久放探针
    tool_timeout_seconds: float = Field(default=30.0, alias="TOOL_TIMEOUT_SECONDS")            # 单个工具执行的超时秒数

    # ---- 检索与记忆开关 ----
    rag_keyword_fallback: bool = Field(default=True, alias="RAG_KEYWORD_FALLBACK")              # 向量召回为空时，退回关键词匹配
    rag_recency_boost: bool = Field(default=True, alias="RAG_RECENCY_BOOST")                    # 检索打分里加时间新鲜度加权
    memory_keyword_check: bool = Field(default=True, alias="MEMORY_KEYWORD_CHECK")              # 记忆召回前先做关键词命中判断，省掉无谓检索
    review_auto_enabled: bool = Field(default=True, alias="REVIEW_AUTO_ENABLED")                # 回答是否自动过一遍 Review 复核
    checkpoint_enabled: bool = Field(default=True, alias="CHECKPOINT_ENABLED")                  # LangGraph 断点续跑开关

    max_react_iterations: int = Field(default=10, alias="MAX_REACT_ITERATIONS")                 # ReAct 最大循环轮数，超了强制出答案
    max_history_tokens: int = Field(default=4000, alias="MAX_HISTORY_TOKENS")                   # 召回的历史对话最多占多少 token
    # ---- 提示词装配：上下文（RAG/记忆/技能/工具清单）合计能占多少 token ----
    #   超了由 PromptBuilder 按优先级裁剪（权限 > 工具 > 观察 > 技能 > RAG > 记忆）。
    #   注意它管的是 prompt，与上面的 max_history_tokens（记忆召回的窗口）不是一回事。
    max_prompt_context_tokens: int = Field(default=3000, alias="MAX_PROMPT_CONTEXT_TOKENS")
    short_memory_rounds: int = Field(default=3, alias="SHORT_MEMORY_ROUNDS")                    # 原样带进 prompt 的最近几轮对话
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")                                        # 向量检索召回多少条
    rerank_top_k: int = Field(default=3, alias="RERANK_TOP_K")                                  # 重排后保留几条进 prompt
    similarity_threshold: float = Field(default=0.65, alias="SIMILARITY_THRESHOLD")             # 相似度下限，低于此值的片段直接丢弃
    session_expire_hours: int = Field(default=72, alias="SESSION_EXPIRE_HOURS")                 # 会话创建多久后算过期

    # ---- 技能匹配：关键词 → 向量 → LLM 三级 ----
    skill_high_threshold: float = Field(default=0.6, alias="SKILL_HIGH_THRESHOLD")              # 关键词得分 ≥ 此值：直接判定命中
    skill_low_threshold: float = Field(default=0.1, alias="SKILL_LOW_THRESHOLD")                # < 此值：直接判定未命中，不再问 LLM
    skill_vector_threshold: float = Field(default=0.75, alias="SKILL_VECTOR_THRESHOLD")         # 向量匹配的命中下限
    skill_vector_top_n: int = Field(default=3, alias="SKILL_VECTOR_TOP_N")                      # 向量匹配取前几个候选
    skill_llm_route_enabled: bool = Field(default=True, alias="SKILL_LLM_ROUTE_ENABLED")        # 前两级没结果时，是否让 LLM 兜底路由
    skill_llm_top_n: int = Field(default=3, alias="SKILL_LLM_TOP_N")                            # LLM 路由时给它几个候选
    skill_llm_confidence: float = Field(default=0.5, alias="SKILL_LLM_CONFIDENCE")              # LLM 路由置信度下限，低于则不绑定技能

    # ---- MCP 技能网关：远程技能统一纳管，不直连 LLM ----
    #   【为什么不直连】MCP 工具若启动时全部注册进 ToolRegistry，
    #   ReAct 每轮都会把全部工具的 schema 塞给模型（工具一多，prompt 与选择难度同步膨胀）。
    #   网关只做「纳管 + 建工具目录」，谁有资格进 prompt 由 AgentRouter 决定。
    mcp_enabled: bool = Field(default=True, alias="MCP_ENABLED")                                # MCP 总开关
    #   拉取工具目录后是否立刻断开子进程；True = 只有真正调用工具时才重新连
    mcp_lazy_connect: bool = Field(default=True, alias="MCP_LAZY_CONNECT")
    #   工具目录落盘缓存：进程重启后直接从缓存建索引，不必先拉起所有 server
    mcp_catalog_cache: str = Field(default="data/mcp_catalog.json", alias="MCP_CATALOG_CACHE")
    mcp_connect_retry: int = Field(default=2, alias="MCP_CONNECT_RETRY")                        # 连接 server 失败后的重试次数
    #   多 server 配置，JSON 数组：[{"name":"enterprise","command":["python","-m","x"]}]
    #   留空则回退到 main.py 里的默认单 server（向后兼容）
    mcp_servers_json: str = Field(default="", alias="MCP_SERVERS_JSON")
    #   远程 MCP Server（HTTP 传输）：远端机器部署常驻 Server，Agent 当 Client 连它
    #   只配这一个 URL 就能用，不必写完整 MCP_SERVERS_JSON
    mcp_remote_url: str = Field(default="", alias="MCP_REMOTE_URL")                              # 远端 /mcp 地址，空=走 stdio 子进程
    mcp_remote_token: str = Field(default="", alias="MCP_REMOTE_TOKEN")                          # Bearer 令牌（服务端设了 MCP_SERVER_TOKEN 才需要）
    # ---- Tool-RAG：按语义只召回本轮用得上的工具 ----
    tool_rag_enabled: bool = Field(default=True, alias="TOOL_RAG_ENABLED")                      # 关掉则每轮把全部工具塞给模型
    tool_rag_top_k: int = Field(default=5, alias="TOOL_RAG_TOP_K")                              # 语义召回几个工具
    #   【绝对下限】低于该相似度的工具不进 prompt：宁可少给一个，也不塞一堆噪声候选
    #     0.25 是实测标定的（智谱 embedding-2，1024 维）：
    #       正确候选的相似度落在 0.28~0.63，无关候选多在 0.14~0.20，
    #       取 0.25 能把两者分开；取 0.30 会砍掉"上海明天天气怎么样"→city_weather(0.2871)
    #       这种表述偏长但确实正确的候选。
    tool_rag_min_score: float = Field(default=0.25, alias="TOOL_RAG_MIN_SCORE")
    #   【相对下限】候选分数 ≥ 本轮最高分 * 该系数
    #     不同 embedding 模型的分数区间不同，纯绝对阈值换个模型就要重调；
    #     加一条相对阈值后，只要"第一名明显领先"就能把尾部噪声自动切掉。
    tool_rag_relative_ratio: float = Field(default=0.6, alias="TOOL_RAG_RELATIVE_RATIO")
    #   单轮最多暴露几个工具（技能绑定 + 语义召回合计），最后一道硬上限
    max_tools_per_turn: int = Field(default=8, alias="MAX_TOOLS_PER_TURN")

    # ---- 工具 Harness：能力声明 / 调用前校验 / 人工审批 / 沙箱 / 预算 / 结果过滤 ----
    # 【这一组配置在管什么】
    #   模型只有"发起申请"的权利，能不能执行由这里决定：
    #     风险等级 → 禁止执行 / 需人工批准 / 直接放行
    #     调用前   → 参数、权限、业务范围、限流、预算五道中间件
    #     执行时   → 按能力声明选沙箱；执行后 → 结果裁剪脱敏 + 审计
    tool_harness_enabled: bool = Field(default=True, alias="TOOL_HARNESS_ENABLED")            # 总开关；关掉=改造前行为
    tool_default_risk: str = Field(default="low", alias="TOOL_DEFAULT_RISK")                  # 工具未声明风险时的兜底等级
    #   形如 "web_search:medium,mysql_exec:critical"，按名字/前缀覆盖工具自己声明的等级
    tool_risk_overrides: str = Field(default="", alias="TOOL_RISK_OVERRIDES")
    tool_deny_levels: str = Field(default="critical", alias="TOOL_DENY_LEVELS")               # 这些等级一律禁止执行
    tool_approve_levels: str = Field(default="high,critical", alias="TOOL_APPROVE_LEVELS")    # 这些等级需人工批准
    tool_deny_tools: str = Field(default="", alias="TOOL_DENY_TOOLS")                         # 工具名黑名单，支持 drop_* 前缀
    tool_approve_tools: str = Field(default="", alias="TOOL_APPROVE_TOOLS")                   # 工具名审批白名单，支持前缀
    tool_on_no_gate: str = Field(default="deny", alias="TOOL_ON_NO_GATE")                     # 需审批但未启用闸门时：deny|allow
    # ---- 执行预算与循环控制 ----
    tool_max_calls_per_run: int = Field(default=8, alias="TOOL_MAX_CALLS_PER_RUN")            # 单次会话最多调几次工具
    tool_max_calls_per_tool: int = Field(default=5, alias="TOOL_MAX_CALLS_PER_TOOL")          # 单工具最多几次（防死循环）
    tool_token_budget: int = Field(default=6000, alias="TOOL_TOKEN_BUDGET")                   # 回灌上下文的累计 token 预算
    tool_run_timeout_seconds: float = Field(default=120.0, alias="TOOL_RUN_TIMEOUT_SECONDS")  # 单次运行墙钟超时
    tool_result_max_chars: int = Field(default=2000, alias="TOOL_RESULT_MAX_CHARS")           # 工具返回值裁剪上限
    # ---- 沙箱：none（进程内）/ subprocess（子进程）/ docker（容器）----
    tool_sandbox_mode: str = Field(default="none", alias="TOOL_SANDBOX_MODE")                 # 全局默认隔离方式，工具可单独声明
    tool_sandbox_timeout_seconds: float = Field(default=15.0, alias="TOOL_SANDBOX_TIMEOUT_SECONDS")
    tool_sandbox_image: str = Field(default="python:3.11-slim", alias="TOOL_SANDBOX_IMAGE")   # 容器沙箱镜像（mode=docker）
    # 远程沙箱（mode=remote）：把同一份类坐标 POST 给远端执行服务；连不上自动降级子进程
    tool_sandbox_remote_url: str = Field(default="", alias="TOOL_SANDBOX_REMOTE_URL")         # 远端 /run 地址，空=未启用
    tool_sandbox_remote_token: str = Field(default="", alias="TOOL_SANDBOX_REMOTE_TOKEN")     # 鉴权令牌，写到 X-Sandbox-Token 头
    tool_sandbox_remote_timeout: float = Field(default=15.0, alias="TOOL_SANDBOX_REMOTE_TIMEOUT")
    # ---- 工具调用限流（令牌桶）----
    tool_rate_per_sec: float = Field(default=5.0, alias="TOOL_RATE_PER_SEC")                  # 全局补充速率（次/秒）
    tool_rate_burst: int = Field(default=20, alias="TOOL_RATE_BURST")                         # 全局桶容量
    tool_rate_per_user: float = Field(default=2.0, alias="TOOL_RATE_PER_USER")                # 单用户补充速率
    tool_rate_burst_per_user: int = Field(default=8, alias="TOOL_RATE_BURST_PER_USER")        # 单用户桶容量

    # ---- 请求处理流水线（清洗 → 风控 → Token控制 → 意图识别 → 分支路由）----
    pipeline_risk_enabled: bool = Field(default=True, alias="PIPELINE_RISK_ENABLED")            # 风控节点（敏感词/注入检测）开关
    #   意图识别前允许的最大输入 token：超过就先压缩，避免长文本把分类 prompt 撑爆
    pipeline_max_input_tokens: int = Field(default=2000, alias="PIPELINE_MAX_INPUT_TOKENS")
    pipeline_compress_enabled: bool = Field(default=True, alias="PIPELINE_COMPRESS_ENABLED")    # 超长输入是否先压缩再送模型
    #   置信度下限：低于此值一律禁止调用任何工具（宁可让模型空手答，也不乱调工具）
    pipeline_min_confidence: float = Field(default=0.5, alias="PIPELINE_MIN_CONFIDENCE")
    #   Tool-RAG 召回数量（只召回本轮用得上的候选技能）
    pipeline_tool_top_k: int = Field(default=3, alias="PIPELINE_TOOL_TOP_K")
    #   pgvector 工具元数据表名
    tool_vector_table: str = Field(default="tool_catalog", alias="TOOL_VECTOR_TABLE")

    # ---- 权限库只读账号 + 数据生命周期 ----
    mysql_readonly_user: str = Field(default="", alias="MYSQL_READONLY_USER")                   # 只读账号；留空就退回上面的 mysql_user
    mysql_readonly_password: str = Field(default="", alias="MYSQL_READONLY_PASSWORD")           # 只读账号密码
    rbac_fail_closed: bool = Field(default=True, alias="RBAC_FAIL_CLOSED")                      # 权限库挂了：True 拒绝请求，False 降级放行
    reconcile_interval_hours: int = Field(default=24, alias="RECONCILE_INTERVAL_HOURS")         # 跨库对账的回看窗口（小时）
    session_hot_days: int = Field(default=90, alias="SESSION_HOT_DAYS")                         # 会话多久内算「热」，超出可归档
    archive_enabled: bool = Field(default=True, alias="ARCHIVE_ENABLED")                        # 冷热分离总开关
    archive_dir: str = Field(default="archives", alias="ARCHIVE_DIR")                           # 归档文件输出目录
    data_retention_days: int = Field(default=180, alias="DATA_RETENTION_DAYS")                  # 数据保留天数，离职清理按它算截止点
    offboard_archive_enabled: bool = Field(default=True, alias="OFFBOARD_ARCHIVE_ENABLED")      # 离职处置时是否顺带归档其会话

    # ---- 多模态（Vision 视觉模型）----
    vision_enabled: bool = Field(default=False, alias="VISION_ENABLED")                         # 视觉能力总开关，默认关
    vision_base_url: str = Field(default="https://api.openai.com/v1", alias="VISION_BASE_URL")  # OpenAI 兼容接口地址
    vision_api_key: str = Field(default="", alias="VISION_API_KEY")                             # 视觉模型密钥
    vision_model: str = Field(default="gpt-4o-mini", alias="VISION_MODEL")                      # 视觉模型名
    vision_temperature: float = Field(default=0.2, alias="VISION_TEMPERATURE")                  # 采样温度，视觉任务要稳所以偏低
    vision_max_tokens: int = Field(default=2048, alias="VISION_MAX_TOKENS")                     # 单次最多生成多少 token
    vision_timeout_seconds: float = Field(default=60.0, alias="VISION_TIMEOUT_SECONDS")         # 单次调用超时秒数
    vision_max_image_mb: int = Field(default=5, alias="VISION_MAX_IMAGE_MB")                    # 单张图片大小上限（MB）
    vision_max_images: int = Field(default=4, alias="VISION_MAX_IMAGES")                        # 单次请求最多几张图
    vision_max_file_chars: int = Field(default=8000, alias="VISION_MAX_FILE_CHARS")             # 随图附带的文本最多多少字符

    # ---- 多 Agent 团队 ----
    multiagent_enabled: bool = Field(default=True, alias="MULTIAGENT_ENABLED")                  # 多 Agent 团队模式开关
    multiagent_max_workers: int = Field(default=4, alias="MULTIAGENT_MAX_WORKERS")              # 并行执行的 worker 数
    multiagent_max_steps: int = Field(default=3, alias="MULTIAGENT_MAX_STEPS")                  # 单个 Agent 最多走几步
    multiagent_max_total_steps: int = Field(default=30, alias="MULTIAGENT_MAX_TOTAL_STEPS")     # 全团队总步数上限
    multiagent_timeout_seconds: float = Field(default=120.0, alias="MULTIAGENT_TIMEOUT_SECONDS")  # 团队总超时秒数
    multiagent_review_enabled: bool = Field(default=True, alias="MULTIAGENT_REVIEW_ENABLED")    # 团队结果是否再过一遍复核

    # ---- 多 Agent 协调（一致性 / 仲裁 / 重规划）----
    multiagent_consensus_enabled: bool = Field(default=True, alias="MULTIAGENT_CONSENSUS_ENABLED")    # 是否检测多个产出之间的一致性
    multiagent_consensus_threshold: float = Field(default=0.35, alias="MULTIAGENT_CONSENSUS_THRESHOLD")  # 相似度低于此值判为冲突
    multiagent_arbitration_enabled: bool = Field(default=True, alias="MULTIAGENT_ARBITRATION_ENABLED")  # 冲突时是否让审核者仲裁
    multiagent_share_findings: bool = Field(default=True, alias="MULTIAGENT_SHARE_FINDINGS")            # 是否把成员产出共享到消息总线
    multiagent_replan_enabled: bool = Field(default=False, alias="MULTIAGENT_REPLAN_ENABLED")           # 有人失败时是否让主管重规划（要多调一次 LLM）
    multiagent_max_replans: int = Field(default=1, alias="MULTIAGENT_MAX_REPLANS")                      # 最多重规划几次

    # ---- Human-in-the-Loop 人工介入 ----
    # 【为什么默认关】开启后链路会在敏感动作处停下来等人，
    #   没有人值守的场合会一直等到超时。这是需要人排班的运行模式，
    #   必须由部署方显式打开，不能靠默认值悄悄生效。
    hitl_enabled: bool = Field(default=False, alias="HITL_ENABLED")
    hitl_kinds: str = Field(default="", alias="HITL_KINDS")                       # 启用哪些中断点，逗号分隔；留空=全部
    hitl_sensitive_tools: str = Field(default="", alias="HITL_SENSITIVE_TOOLS")   # 需人工批准的工具名，逗号分隔；支持 mysql_* 前缀
    hitl_timeout_seconds: float = Field(default=60.0, alias="HITL_TIMEOUT_SECONDS")   # 单次等待上限，超时按默认动作兜底

    # ---- HITL 通知：中断抛出的那一刻怎么通知到人 ----
    # 【为什么需要单独一组开关】中断请求登记后只是躺在内存里等人来取，
    #   没有通知就是"系统自己按兜底动作处理了，人全程不知情"。
    #   console 面向本地/CLI，webhook 面向钉钉飞书，SSE 面向审批页（无需开关，连上就有）。
    hitl_notify_console: bool = Field(default=True, alias="HITL_NOTIFY_CONSOLE")                 # 中断时打印到终端(stderr)
    hitl_notify_webhook: str = Field(default="", alias="HITL_NOTIFY_WEBHOOK")                    # 机器人地址；留空=不推送
    hitl_notify_webhook_kind: str = Field(default="auto", alias="HITL_NOTIFY_WEBHOOK_KIND")      # auto/dingtalk/feishu/wecom/generic
    hitl_notify_webhook_timeout: float = Field(default=3.0, alias="HITL_NOTIFY_WEBHOOK_TIMEOUT") # 推送超时（后台线程，不吃等待时间）

    # ---- 中心化编排（限流 / 配额 / 防环 / 安全校验 / 观测）----
    # 【这套参数调什么】
    #   消息发不出去 → 先看 rate_per_sec / burst（限流）与 max_per_sender_round（配额）
    #   两个 Agent 互相踢皮球 → 调小 max_hops 与 pingpong_limit
    #   担心敏感信息外传 → 填 sensitive_words，并把 on_sensitive 设为 reject
    orchestration_enabled: bool = Field(default=True, alias="ORCHESTRATION_ENABLED")                      # 关掉则消息直连总线（不推荐）
    orchestration_max_hops: int = Field(default=2, alias="ORCHESTRATION_MAX_HOPS")                        # 一条对话链最多几跳
    orchestration_max_per_run: int = Field(default=100, alias="ORCHESTRATION_MAX_PER_RUN")                 # 单次运行消息总量
    orchestration_max_per_round: int = Field(default=30, alias="ORCHESTRATION_MAX_PER_ROUND")              # 单轮消息上限
    orchestration_max_per_sender_round: int = Field(default=3, alias="ORCHESTRATION_MAX_PER_SENDER_ROUND") # 同一发送者单轮上限
    orchestration_rate_per_sec: float = Field(default=20.0, alias="ORCHESTRATION_RATE_PER_SEC")            # 全局消息速率（条/秒）
    orchestration_burst: int = Field(default=40, alias="ORCHESTRATION_BURST")                              # 令牌桶容量（瞬时突发）
    orchestration_repeat_limit: int = Field(default=2, alias="ORCHESTRATION_REPEAT_LIMIT")                 # 相同内容最多几次
    orchestration_pingpong_limit: int = Field(default=2, alias="ORCHESTRATION_PINGPONG_LIMIT")             # 双向各几次算踢皮球
    orchestration_sensitive_words: str = Field(default="", alias="ORCHESTRATION_SENSITIVE_WORDS")          # 敏感词，逗号分隔
    orchestration_on_sensitive: str = Field(default="redact", alias="ORCHESTRATION_ON_SENSITIVE")          # redact|reject
    orchestration_max_content_chars: int = Field(default=4000, alias="ORCHESTRATION_MAX_CONTENT_CHARS")    # 单条正文上限
    orchestration_trace_enabled: bool = Field(default=True, alias="ORCHESTRATION_TRACE_ENABLED")           # 是否记消息级 span

    # ---- 可选框架桥接（未安装也照常跑，装了按开关切换）----
    # 两个后端都不设为默认：llama-index 用独立索引（切过去旧知识库就空了），
    # crewai 走自己的 litellm 模型配置（与 DEEPSEEK_API_KEY 不是一套）。
    # 后端切换属于数据/模型迁移级别的操作，必须显式配置，不能靠默认值悄悄发生。
    rag_backend: str = Field(default="native", alias="RAG_BACKEND")                 # native|llamaindex
    multiagent_backend: str = Field(default="native", alias="MULTIAGENT_BACKEND")   # native|crewai
    llamaindex_persist_dir: str = Field(default="storage/llamaindex", alias="LLAMAINDEX_PERSIST_DIR")  # llama-index 索引落盘目录

    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")                     # 允许的前端来源；* = 不限制

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")                       # API 监听地址
    app_port: int = Field(default=8000, alias="APP_PORT")                            # API 端口
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")                        # 日志级别

    model_config = SettingsConfigDict(extra="ignore")                                # .env 里多出来的键直接忽略，不报错

    @property
    def pg_dsn(self) -> str:
        # 【什么】拼接 PostgreSQL 连接串
        # 【作用】 PGManager 等组件统一用它建连，避免各处手拼、漏改一处就连不上
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @field_validator("deepseek_api_key", "zhipu_api_key")
    @classmethod
    def _check_key_not_empty(cls, v: str, info) -> str:
        # 【什么】校验两个模型密钥是否为空
        # 【作用】 为空只打告警不抛错，保证本机没配 Key 也能跑 ingest/stats 这类命令
        if not v:
            import logging
            logging.warning(f"配置项 {info.field_name} 为空，请检查 .env 文件")
        return v


# 【什么】全局配置单例（lru_cache 只缓存 1 份）
# 【作用】 .env 只解析一次，任何地方 get_settings() 拿到的都是同一个对象
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例（带缓存，避免重复解析 .env）。"""
    return Settings()


# 【什么】开启 LangSmith 追踪（往 os.environ 写 8 个变量）
# 【作用】 让 LangChain/LangGraph 自动上报调用链；没配 Key 时直接跳过，不影响主流程
def setup_langsmith():
    """启用 LangSmith 全链路追踪；未配置 Key 时静默跳过，不影响主流程。

    背景知识：
        LangSmith 是 LangChain 官方的可观测性平台（类似前端的 Sentry/APM）。
        LangChain/LangGraph 运行时只要检测到下面几个环境变量，就会把每一次
        LLM 调用、每个图节点的输入输出、耗时、token 用量自动上报到 LangSmith
        网页后台，形成一条可展开的调用链，方便排查「Agent 为什么调了这个工具」。

    为什么用「写环境变量」而不是传参：
        LangChain SDK 在底层直接读 os.environ 里的 LANGCHAIN_* 变量，
        没有提供显式的 init 函数，所以必须在程序最早期（main() 第一行）
        把 .env 里的配置转写成它认识的环境变量。

    【两套变量名：为什么写 8 个而不是 4 个】
        LangSmith 从 LangChain 里独立出去后，环境变量前缀换成了 LANGSMITH_，
        官方文档现在给的是新名：LANGSMITH_TRACING / LANGSMITH_ENDPOINT /
        LANGSMITH_API_KEY / LANGSMITH_PROJECT；项目早期用的是旧名 LANGCHAIN_*
        （其中追踪开关旧名是 LANGCHAIN_TRACING_V2，带 V2 后缀）。
        读变量的函数 get_env_var() 查找顺序是 namespaces=("LANGSMITH", "LANGCHAIN")，
        两套都认、新名优先（实测 langsmith 0.11.0）。全写是为了跨版本都能生效：
            版本偏新、只认 LANGSMITH_*  → 新名兜住
            版本偏旧、只认 LANGCHAIN_*  → 旧名兜住
        不双写的后果：只写一套时，换台机器/换个版本可能静默失效 ——
        不报错、不上报，本地日志也看不出来，要等去网页后台才发现没有 trace。

    例子 1 —— .env 里配了 Key（追踪开启）：
        # .env 文件内容：
        # LANGSMITH_API_KEY=lsv2_pt_abc123...
        # LANGSMITH_PROJECT=ai-agent-prod
        setup_langsmith()
        # 执行后 os.environ 中多出 8 个变量：
        #   LANGSMITH_TRACING     = "true"
        #   LANGSMITH_ENDPOINT    = "https://api.smith.langchain.com"
        #   LANGSMITH_API_KEY     = "lsv2_pt_abc123..."
        #   LANGSMITH_PROJECT     = "ai-agent-prod"
        #   LANGCHAIN_TRACING_V2  = "true"
        #   LANGCHAIN_ENDPOINT    = "https://api.smith.langchain.com"
        #   LANGCHAIN_API_KEY     = "lsv2_pt_abc123..."
        #   LANGCHAIN_PROJECT     = "ai-agent-prod"
        # 之后每次 graph.invoke() / llm.create() 都会上报到该项目，
        # 登录 smith.langchain.com 即可看到 prepare→reason→act 的完整链路树。

    例子 2 —— .env 里没配 Key（本地开发，追踪关闭）：
        # .env 中 LANGSMITH_API_KEY 为空字符串
        setup_langsmith()
        # if 判断为 False，8 个环境变量一个都不写，函数直接结束；
        # LangChain 检测不到开关，按普通模式运行，不发起任何网络上报。

    【依赖警告】与 langsmith 包读变量的顺序硬绑定：
        get_env_var() 的 namespaces 是 ("LANGSMITH", "LANGCHAIN")，新名优先。
        若日后想精简成一套名字，必须先确认目标版本仍认另一套，否则会静默失效。

    返回值：无（仅通过修改 os.environ 产生副作用）。
    """
    s = get_settings()
    if not s.langsmith_api_key:
        return
    # ---- 新名（官方文档现行写法，langsmith SDK 优先读这一套）----
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
    # ---- 旧名（LANGCHAIN_ 前缀，老版本 SDK 仍读这一套，追踪开关带 V2 后缀）----
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = s.langsmith_project
