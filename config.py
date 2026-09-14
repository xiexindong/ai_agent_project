"""
统一配置加载模块
================
从 .env 读取全局配置，提供参数校验与单例访问。
所有密钥、数据库地址、模型参数集中管理，禁止硬编码。
"""
import os # 用于获取环境变量
from pathlib import Path
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


# ---- 定位 .env 文件：优先项目目录，其次上级目录 ----
_PROJECT_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _PROJECT_DIR.parent

if (_PROJECT_DIR / ".env").exists():
    load_dotenv(_PROJECT_DIR / ".env")
elif (_PARENT_DIR / ".env").exists():
    load_dotenv(_PARENT_DIR / ".env")
else:
    load_dotenv()  # 兜底：从默认路径加载


class Settings(BaseSettings):
    """全局配置单例，集中管理所有运行参数。

    继承 BaseSettings：实例化时自动从 os.environ 按 alias 读取配置；
    模块顶部 load_dotenv 已将 .env 内容注入 os.environ。
    """

    # ---- DeepSeek 对话 LLM ----
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    # ---- 智谱 Embedding 模型 ----
    zhipu_api_key: str = Field(default="", alias="ZHIPU_API_KEY")
    zhipu_base_url: str = Field(default="https://open.bigmodel.cn/api/paas/v4", alias="ZHIPU_BASE_URL")
    embedding_model: str = Field(default="embedding-2", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")

    # ---- Tavily 搜索 ----
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # ---- PostgreSQL + pgvector ----
    pg_host: str = Field(default="localhost", alias="PG_HOST")
    pg_port: int = Field(default=5432, alias="PG_PORT")
    pg_user: str = Field(default="postgres", alias="PG_USER")
    pg_password: str = Field(default="postgres", alias="PG_PASSWORD")
    pg_database: str = Field(default="ai_agent", alias="PG_DATABASE")

    # ---- Milvus ----
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_collection: str = Field(default="knowledge_base", alias="MILVUS_COLLECTION")

    # ---- LangSmith 追踪 ----
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="ai-agent-prod", alias="LANGSMITH_PROJECT")
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")

    # ---- Agent 运行参数 ----
    max_react_iterations: int = Field(default=10, alias="MAX_REACT_ITERATIONS")
    max_history_tokens: int = Field(default=4000, alias="MAX_HISTORY_TOKENS")
    short_memory_rounds: int = Field(default=3, alias="SHORT_MEMORY_ROUNDS")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rerank_top_k: int = Field(default=3, alias="RERANK_TOP_K")
    similarity_threshold: float = Field(default=0.65, alias="SIMILARITY_THRESHOLD")
    session_expire_hours: int = Field(default=72, alias="SESSION_EXPIRE_HOURS")

    # ---- 服务参数 ----
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # extra="ignore"：.env 里多写的变量（如 OPENAI_API_KEY1）不报错
    # 不设 env_file：统一由模块顶部 load_dotenv 注入 os.environ 后按 alias 读取
    model_config = SettingsConfigDict(extra="ignore")

    # ---- 派生属性：PG 连接串 ----
    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    # ---- 校验：关键密钥不能为空 ----
    @field_validator("deepseek_api_key", "zhipu_api_key")
    @classmethod
    def _check_key_not_empty(cls, v: str, info) -> str:
        # 允许测试环境为空，运行时由启动入口二次校验
        if not v:
            import logging
            logging.warning(f"配置项 {info.field_name} 为空，请检查 .env 文件")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例（带缓存，避免重复解析 .env）。"""
    return Settings()


# ---- LangSmith 环境变量注入（全链路追踪开关）----
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
    # 只有配了 Key 才开启：避免本地/测试环境无 Key 时报错或发出无效请求
    if s.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key        # 上报鉴权凭证
        os.environ["LANGCHAIN_PROJECT"] = s.langsmith_project        # 后台项目名（按项目归类链路）
        os.environ["LANGCHAIN_TRACING_V2"] = "true"                  # 总开关："true" 才启用追踪
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"  # 上报地址
