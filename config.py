"""
统一配置加载模块
================
从 .env 读取全局配置，提供参数校验与单例访问。
所有密钥、数据库地址、模型参数集中管理，禁止硬编码。
"""
import os
from pathlib import Path
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


_PROJECT_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _PROJECT_DIR.parent

if (_PROJECT_DIR / ".env").exists():
    load_dotenv(_PROJECT_DIR / ".env")
elif (_PARENT_DIR / ".env").exists():
    load_dotenv(_PARENT_DIR / ".env")
else:
    load_dotenv()


class Settings(BaseSettings):
    """全局配置单例，集中管理所有运行参数。

    继承 BaseSettings：实例化时自动从 os.environ 按 alias 读取配置；
    模块顶部 load_dotenv 已将 .env 内容注入 os.environ。
    """

    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    zhipu_api_key: str = Field(default="", alias="ZHIPU_API_KEY")
    zhipu_base_url: str = Field(default="https://open.bigmodel.cn/api/paas/v4", alias="ZHIPU_BASE_URL")
    embedding_model: str = Field(default="embedding-2", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")

    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    pg_host: str = Field(default="localhost", alias="PG_HOST")
    pg_port: int = Field(default=5432, alias="PG_PORT")
    pg_user: str = Field(default="postgres", alias="PG_USER")
    pg_password: str = Field(default="postgres", alias="PG_PASSWORD")
    pg_database: str = Field(default="ai_agent", alias="PG_DATABASE")

    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_collection: str = Field(default="knowledge_base", alias="MILVUS_COLLECTION")

    mysql_host: str = Field(default="localhost", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str = Field(default="root", alias="MYSQL_USER")
    mysql_password: str = Field(default="", alias="MYSQL_PASSWORD")
    mysql_database: str = Field(default="ai_agent_enterprise", alias="MYSQL_DATABASE")
    rbac_enabled: bool = Field(default=True, alias="RBAC_ENABLED")
    rbac_cache_ttl: int = Field(default=60, alias="RBAC_CACHE_TTL")
    kb_acl_enabled: bool = Field(default=True, alias="KB_ACL_ENABLED")
    mask_enabled: bool = Field(default=True, alias="MASK_ENABLED")

    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="ai-agent-prod", alias="LANGSMITH_PROJECT")
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")

    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_attempts: int = Field(default=3, alias="LLM_MAX_ATTEMPTS")
    fallback_model: str = Field(default="", alias="FALLBACK_MODEL")
    tool_timeout_seconds: float = Field(default=30.0, alias="TOOL_TIMEOUT_SECONDS")

    rag_keyword_fallback: bool = Field(default=True, alias="RAG_KEYWORD_FALLBACK")
    rag_recency_boost: bool = Field(default=True, alias="RAG_RECENCY_BOOST")
    memory_keyword_check: bool = Field(default=True, alias="MEMORY_KEYWORD_CHECK")
    review_auto_enabled: bool = Field(default=True, alias="REVIEW_AUTO_ENABLED")
    checkpoint_enabled: bool = Field(default=True, alias="CHECKPOINT_ENABLED")

    max_react_iterations: int = Field(default=10, alias="MAX_REACT_ITERATIONS")
    max_history_tokens: int = Field(default=4000, alias="MAX_HISTORY_TOKENS")
    short_memory_rounds: int = Field(default=3, alias="SHORT_MEMORY_ROUNDS")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rerank_top_k: int = Field(default=3, alias="RERANK_TOP_K")
    similarity_threshold: float = Field(default=0.65, alias="SIMILARITY_THRESHOLD")
    session_expire_hours: int = Field(default=72, alias="SESSION_EXPIRE_HOURS")

    skill_high_threshold: float = Field(default=0.6, alias="SKILL_HIGH_THRESHOLD")
    skill_low_threshold: float = Field(default=0.1, alias="SKILL_LOW_THRESHOLD")
    skill_vector_threshold: float = Field(default=0.75, alias="SKILL_VECTOR_THRESHOLD")
    skill_vector_top_n: int = Field(default=3, alias="SKILL_VECTOR_TOP_N")
    skill_llm_route_enabled: bool = Field(default=True, alias="SKILL_LLM_ROUTE_ENABLED")
    skill_llm_top_n: int = Field(default=3, alias="SKILL_LLM_TOP_N")
    skill_llm_confidence: float = Field(default=0.5, alias="SKILL_LLM_CONFIDENCE")

    mysql_readonly_user: str = Field(default="", alias="MYSQL_READONLY_USER")
    mysql_readonly_password: str = Field(default="", alias="MYSQL_READONLY_PASSWORD")
    rbac_fail_closed: bool = Field(default=True, alias="RBAC_FAIL_CLOSED")
    reconcile_interval_hours: int = Field(default=24, alias="RECONCILE_INTERVAL_HOURS")
    session_hot_days: int = Field(default=90, alias="SESSION_HOT_DAYS")
    archive_enabled: bool = Field(default=True, alias="ARCHIVE_ENABLED")
    archive_dir: str = Field(default="archives", alias="ARCHIVE_DIR")
    data_retention_days: int = Field(default=180, alias="DATA_RETENTION_DAYS")
    offboard_archive_enabled: bool = Field(default=True, alias="OFFBOARD_ARCHIVE_ENABLED")

    # ---- 多模态（Vision 视觉模型）----
    vision_enabled: bool = Field(default=False, alias="VISION_ENABLED")
    vision_base_url: str = Field(default="https://api.openai.com/v1", alias="VISION_BASE_URL")
    vision_api_key: str = Field(default="", alias="VISION_API_KEY")
    vision_model: str = Field(default="gpt-4o-mini", alias="VISION_MODEL")
    vision_temperature: float = Field(default=0.2, alias="VISION_TEMPERATURE")
    vision_max_tokens: int = Field(default=2048, alias="VISION_MAX_TOKENS")
    vision_timeout_seconds: float = Field(default=60.0, alias="VISION_TIMEOUT_SECONDS")
    vision_max_image_mb: int = Field(default=5, alias="VISION_MAX_IMAGE_MB")
    vision_max_images: int = Field(default=4, alias="VISION_MAX_IMAGES")
    vision_max_file_chars: int = Field(default=8000, alias="VISION_MAX_FILE_CHARS")

    # ---- 多 Agent 团队 ----
    multiagent_enabled: bool = Field(default=True, alias="MULTIAGENT_ENABLED")
    multiagent_max_workers: int = Field(default=4, alias="MULTIAGENT_MAX_WORKERS")
    multiagent_max_steps: int = Field(default=3, alias="MULTIAGENT_MAX_STEPS")
    multiagent_max_total_steps: int = Field(default=30, alias="MULTIAGENT_MAX_TOTAL_STEPS")
    multiagent_timeout_seconds: float = Field(default=120.0, alias="MULTIAGENT_TIMEOUT_SECONDS")
    multiagent_review_enabled: bool = Field(default=True, alias="MULTIAGENT_REVIEW_ENABLED")

    # ---- 可选框架桥接（未安装也照常跑，装了按开关切换）----
    # 两个后端都不设为默认：llama-index 用独立索引（切过去旧知识库就空了），
    # crewai 走自己的 litellm 模型配置（与 DEEPSEEK_API_KEY 不是一套）。
    # 后端切换属于数据/模型迁移级别的操作，必须显式配置，不能靠默认值悄悄发生。
    rag_backend: str = Field(default="native", alias="RAG_BACKEND")                 # native|llamaindex
    multiagent_backend: str = Field(default="native", alias="MULTIAGENT_BACKEND")   # native|crewai
    llamaindex_persist_dir: str = Field(default="storage/llamaindex", alias="LLAMAINDEX_PERSIST_DIR")

    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(extra="ignore")

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @field_validator("deepseek_api_key", "zhipu_api_key")
    @classmethod
    def _check_key_not_empty(cls, v: str, info) -> str:
        if not v:
            import logging
            logging.warning(f"配置项 {info.field_name} 为空，请检查 .env 文件")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例（带缓存，避免重复解析 .env）。"""
    return Settings()


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

    例子 1 —— .env 里配了 Key（追踪开启）：
        # .env 文件内容：
        # LANGSMITH_API_KEY=lsv2_pt_abc123...
        # LANGSMITH_PROJECT=ai-agent-prod
        setup_langsmith()
        # 执行后 os.environ 中多出 4 个变量：
        #   LANGCHAIN_API_KEY      = "lsv2_pt_abc123..."
        #   LANGCHAIN_PROJECT      = "ai-agent-prod"
        #   LANGCHAIN_TRACING_V2   = "true"
        #   LANGCHAIN_ENDPOINT     = "https://api.smith.langchain.com"
        # 之后每次 graph.invoke() / llm.create() 都会上报到该项目，
        # 登录 smith.langchain.com 即可看到 prepare→reason→act 的完整链路树。

    例子 2 —— .env 里没配 Key（本地开发，追踪关闭）：
        # .env 中 LANGSMITH_API_KEY 为空字符串
        setup_langsmith()
        # if 判断为 False，4 个环境变量一个都不写，函数直接结束；
        # LangChain 检测不到开关，按普通模式运行，不发起任何网络上报。

    返回值：无（仅通过修改 os.environ 产生副作用）。
    """
    s = get_settings()
    if s.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = s.langsmith_project
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
