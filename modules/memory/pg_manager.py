"""
PostgreSQL + pgvector 管理模块
===============================
业务存储 + 对话记忆向量存储一体化管理。
- 会话主表 chat_session：会话信息、Agent结构化任务状态
- 对话记忆表 chat_history：原文、摘要、摘要向量、工具日志
覆盖边界4（上下文压缩）、Agent状态断点续跑。
"""
import json
from typing import List, Optional
from datetime import datetime
from modules.common.logger import get_logger
from modules.common.exceptions import MemoryException, handle_exception
from config import get_settings

_logger = get_logger("pg")


class PGManager:
    """PostgreSQL 连接与表结构管理（带pgvector扩展）。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_connected"):
            self._connect()
            self._connected = True

    def _connect(self):
        """连接 PG 并注册 pgvector 扩展。"""
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            s = get_settings()
            self._conn = psycopg2.connect(
                host=s.pg_host,
                port=s.pg_port,
                user=s.pg_user,
                password=s.pg_password,
                dbname=s.pg_database,
            )
            self._conn.autocommit = False
            self._cur = self._conn.cursor()
            # 注册 pgvector 类型
            self._cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self._conn.commit()
            register_vector(self._conn)
            self._init_tables()
            _logger.info(f"PG连接成功: {s.pg_host}:{s.pg_port}/{s.pg_database}")
        except Exception as e:
            _logger.error(f"PG连接失败: {e}")
            raise MemoryException("PG连接失败", str(e))

    def _init_tables(self):
        """创建生产级表结构（幂等）。"""
        s = get_settings()
        dim = s.embedding_dimension
        ddl = [
            # 会话主表
            f"""
            CREATE TABLE IF NOT EXISTS chat_session (
                session_id   VARCHAR(64) PRIMARY KEY,
                user_id      VARCHAR(64) NOT NULL,
                title        VARCHAR(256) DEFAULT '',
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agent_state  JSONB DEFAULT '{{}}'::jsonb
            );
            """,
            # 对话记忆表（核心向量记忆表）
            f"""
            CREATE TABLE IF NOT EXISTS chat_history (
                id                 BIGSERIAL PRIMARY KEY,
                session_id         VARCHAR(64) NOT NULL REFERENCES chat_session(session_id) ON DELETE CASCADE,
                user_id            VARCHAR(64) NOT NULL,
                user_query         TEXT NOT NULL,
                llm_full_answer    TEXT,
                llm_summary        TEXT,
                summary_embedding  vector({dim}),
                tool_call_log      JSONB DEFAULT '[]'::jsonb,
                rag_sources        JSONB DEFAULT '[]'::jsonb,
                created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # 摘要向量索引（ivfflat 加速语义检索）
            f"""
            CREATE INDEX IF NOT EXISTS idx_history_summary_emb
            ON chat_history USING ivfflat (summary_embedding vector_cosine_ops)
            WITH (lists = 100);
            """,
            # 会话+用户联合索引（多用户隔离查询加速）
            f"""
            CREATE INDEX IF NOT EXISTS idx_history_session_user
            ON chat_history (session_id, user_id, created_at DESC);
            """,
            # 工具调用日志表
            f"""
            CREATE TABLE IF NOT EXISTS tool_call_log (
                id           BIGSERIAL PRIMARY KEY,
                session_id   VARCHAR(64),
                user_id      VARCHAR(64),
                tool_name    VARCHAR(128) NOT NULL,
                tool_type    VARCHAR(32) DEFAULT 'local',
                tool_input   JSONB,
                tool_output  TEXT,
                status       VARCHAR(16) DEFAULT 'success',
                duration_ms  INTEGER DEFAULT 0,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # 人工审核标记表
            f"""
            CREATE TABLE IF NOT EXISTS review_queue (
                id           BIGSERIAL PRIMARY KEY,
                session_id   VARCHAR(64),
                history_id   BIGINT,
                user_id      VARCHAR(64),
                content      TEXT,
                review_type  VARCHAR(32),
                status       VARCHAR(16) DEFAULT 'pending',
                reviewer     VARCHAR(64),
                review_note  TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at  TIMESTAMP
            );
            """,
        ]
        for sql in ddl:
            self._cur.execute(sql)
        self._conn.commit()
        _logger.info("PG表结构初始化完成（chat_session, chat_history, tool_call_log, review_queue）")

    def _reconnect_if_needed(self):
        """连接断开时自动重连（心跳探活 + 失败自愈）。"""
        try:
            # 心跳探测：发一条最轻量的 SQL（SELECT 1 不查表、无业务开销），
            # 成功 = TCP 连接还活着，直接返回继续后续 SQL；
            # 失败 = 连接已失效（PG 重启 / 空闲超时被断开 / 网络抖动），进入 except
            self._cur.execute("SELECT 1")
        except Exception:
            # 捕获到任意数据库异常（如 OperationalError 连接已关闭）：
            # 不把错误抛给上层业务，而是就地自愈——重置标记并调用 _connect() 重建连接池/游标，
            # 让本次请求随后的 SQL 能在新连接上继续执行（后续每个方法入口都会先调本方法）。
            # 类比 JS：axios 拦截器里发现网络断了，先 await reconnect() 再重放原请求
            _logger.warning("PG连接断开，尝试重连...")
            self._connected = False
            self._connect()

    # ==================== 会话管理 ====================

    @handle_exception(default=None, log_msg="创建会话异常", reraise=True)
    def create_session(self, session_id: str, user_id: str, title: str = "") -> None:
        """创建新会话。"""
        self._reconnect_if_needed()
        self._cur.execute(
            "INSERT INTO chat_session (session_id, user_id, title) VALUES (%s, %s, %s) "
            "ON CONFLICT (session_id) DO NOTHING",
            (session_id, user_id, title),
        )
        self._conn.commit()
        _logger.info(f"会话创建: session={session_id}, user={user_id}")

    @handle_exception(default=None, log_msg="更新会话状态异常", reraise=True)
    def update_session_state(self, session_id: str, agent_state: dict) -> None:
        """更新 Agent 结构化任务状态（支持断点续跑）。"""
        self._reconnect_if_needed()
        self._cur.execute(
            "UPDATE chat_session SET agent_state = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE session_id = %s",
            (json.dumps(agent_state, ensure_ascii=False), session_id),
        )
        self._conn.commit()

    @handle_exception(default={}, log_msg="读取会话状态异常")
    def get_session_state(self, session_id: str, user_id: str) -> dict:
        """读取 Agent 任务状态（断点续跑用）。"""
        self._reconnect_if_needed()
        self._cur.execute(
            "SELECT agent_state FROM chat_session WHERE session_id = %s AND user_id = %s",
            (session_id, user_id),
        )
        row = self._cur.fetchone()
        return row[0] if row and row[0] else {}

    # ==================== 对话记忆 ====================

    @handle_exception(default=None, log_msg="保存对话记忆异常", reraise=True)
    def save_history(
        self,
        session_id: str,
        user_id: str,
        user_query: str,
        llm_full_answer: str,
        llm_summary: str,
        summary_embedding: list,
        tool_call_log: list = None,
        rag_sources: list = None,
    ) -> int:
        """
        保存一轮对话到 chat_history 表（三数据落库）。

        【为什么是"三数据"？】一次对话要存 3 类数据，各有用途：
            llm_full_answer    : 完整原文 → 短期记忆用（最近 N 轮塞回 prompt）
            llm_summary        : 一句话摘要 → 长期记忆向量化的源文本
            summary_embedding  : 摘要向量 → 供 search_memory_by_vector 语义召回
            缺任一项后续记忆能力都会残废：缺原文没法看最近聊了啥，
            缺向量没法语义召回历史对话。

        【落库前后的数据变化】
            输入参数：
                session_id="abc123", user_id="cli-user",
                user_query="3*7 等于多少",
                llm_full_answer="3*7 等于 21...",   （长文本）
                llm_summary="数学计算:3*7=21",       （一句话）
                summary_embedding=[0.12, -0.34, ...]（1024 维向量）
                tool_call_log=[{"name":"calculator","args":...,"result":"21"}]

            数据转换：
                summary_embedding → np.array(dtype=float32)
                                  （pgvector 要求 numpy 数组，不能是 list）
                tool_call_log → JSON 字符串（JSONB 列存）
                rag_sources  → JSON 字符串

            落库后（chat_history 表新增 1 行）：
                id=42, session_id="abc123", ... created_at=NOW
            返回: 42（新行的主键 ID，用于后续 add_to_review_queue 关联）
        """
        self._reconnect_if_needed()
        import numpy as np
        emb = np.array(summary_embedding, dtype=np.float32) if summary_embedding else None
        self._cur.execute(
            """INSERT INTO chat_history
               (session_id, user_id, user_query, llm_full_answer, llm_summary,
                summary_embedding, tool_call_log, rag_sources)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (
                session_id, user_id, user_query,
                llm_full_answer, llm_summary, emb,
                json.dumps(tool_call_log or [], ensure_ascii=False),
                json.dumps(rag_sources or [], ensure_ascii=False),
            ),
        )
        row = self._cur.fetchone()
        self._conn.commit()
        history_id = row[0] if row else None
        _logger.info(f"对话记忆保存: history_id={history_id}, session={session_id}")
        return history_id

    @handle_exception(default=[], log_msg="读取短期记忆异常")
    def get_short_term_history(self, session_id: str, user_id: str, rounds: int = 3) -> list:
        """读取最近 N 轮完整对话（短期记忆）。"""
        self._reconnect_if_needed()
        self._cur.execute(
            """SELECT user_query, llm_full_answer, llm_summary, created_at
               FROM chat_history
               WHERE session_id = %s AND user_id = %s
               ORDER BY created_at DESC LIMIT %s""",
            (session_id, user_id, rounds),
        )
        rows = self._cur.fetchall()
        # 时间正序（旧→新）
        rows.reverse()
        return [
            {"query": r[0], "answer": r[1], "summary": r[2], "time": r[3]}
            for r in rows
        ]

    @handle_exception(default=[], log_msg="向量召回历史记忆异常")
    def search_memory_by_vector(self, query_embedding: list, user_id: str, top_k: int = 3) -> list:
        """
        基于摘要向量语义召回历史对话（长期记忆）。

        【SQL 关键算子 <=> 是什么？】pgvector 的"余弦距离"算子，返回 0~2：
            0   = 两向量完全同向（语义最相似）
            2   = 完全反向
            1-cosine 距离 = 相似度分（0~1，越高越像）
            SQL: ORDER BY summary_embedding <=> query_emb LIMIT top_k

        【多用户隔离】WHERE user_id = %s —— 用户 A 的记忆对用户 B 不可见，
            避免跨用户串记忆（企业级安全要求）。

        【输入输出对比】
            输入：
                query_embedding=[0.12,-0.34,...]（当前问题的 1024 维向量）
                user_id="cli-user"
                top_k=3
            输出（按相似度倒序）：
                [
                    {"query":"3*5 等于多少", "summary":"数学计算:3*5=15",
                     "answer":"3*5 等于 15", "similarity":0.91, "time":"2026-09-14 12:00"},
                    {"query":"加法怎么算", "summary":"用户问加法",
                     "answer":"...", "similarity":0.73, "time":"2026-09-13 18:00"},
                ]
                → 这些拼成 memory_context 塞回 prompt，让 LLM 知道"以前聊过类似话题"
                → 若用户问的是数学题，召回历史也涉及数学，模型回答更连贯
        """
        self._reconnect_if_needed()
        import numpy as np
        emb = np.array(query_embedding, dtype=np.float32)
        self._cur.execute(
            """SELECT user_query, llm_summary, llm_full_answer,
                      1 - (summary_embedding <=> %s) AS similarity, created_at
               FROM chat_history
               WHERE user_id = %s AND summary_embedding IS NOT NULL
               ORDER BY summary_embedding <=> %s
               LIMIT %s""",
            (emb, user_id, emb, top_k),
        )
        rows = self._cur.fetchall()
        return [
            {
                "query": r[0], "summary": r[1], "answer": r[2],
                "similarity": round(float(r[3]), 4), "time": r[4],
            }
            for r in rows
        ]

    # ==================== 工具调用日志 ====================

    @handle_exception(default=None, log_msg="保存工具日志异常")
    def save_tool_log(
        self, session_id: str, user_id: str,
        tool_name: str, tool_type: str,
        tool_input: dict, tool_output: str,
        status: str, duration_ms: int,
    ) -> None:
        """全量记录工具调用入参、出参、耗时、状态（审计依据）。"""
        self._reconnect_if_needed()
        self._cur.execute(
            """INSERT INTO tool_call_log
               (session_id, user_id, tool_name, tool_type, tool_input, tool_output, status, duration_ms)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                session_id, user_id, tool_name, tool_type,
                json.dumps(tool_input, ensure_ascii=False),
                tool_output, status, duration_ms,
            ),
        )
        self._conn.commit()

    # ==================== 人工审核 ====================

    @handle_exception(default=None, log_msg="加入审核队列异常")
    def add_to_review_queue(
        self, session_id: str, user_id: str,
        history_id: int, content: str, review_type: str,
    ) -> None:
        """将高价值/异常对话加入人工审核队列。"""
        self._reconnect_if_needed()
        self._cur.execute(
            """INSERT INTO review_queue (session_id, user_id, history_id, content, review_type)
               VALUES (%s, %s, %s, %s, %s)""",
            (session_id, user_id, history_id, content, review_type),
        )
        self._conn.commit()
        _logger.info(f"加入审核队列: history_id={history_id}, type={review_type}")

    @handle_exception(default=[], log_msg="查询待审核列表异常")
    def get_pending_reviews(self, limit: int = 20) -> list:
        """查询待人工审核的记录列表。"""
        self._reconnect_if_needed()
        self._cur.execute(
            """SELECT id, session_id, user_id, history_id, content, review_type, created_at
               FROM review_queue WHERE status = 'pending'
               ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
        return [
            {"id": r[0], "session_id": r[1], "user_id": r[2], "history_id": r[3],
             "content": r[4], "review_type": r[5], "created_at": r[6]}
            for r in self._cur.fetchall()
        ]

    def close(self):
        """关闭连接。"""
        if hasattr(self, "_cur"):
            self._cur.close()
        if hasattr(self, "_conn"):
            self._conn.close()
        _logger.info("PG连接已关闭")
