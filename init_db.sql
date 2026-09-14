-- ====================================================================
-- 企业级 AI Agent 智能助手 - PostgreSQL + pgvector 数据库初始化
-- 执行方式：psql -U postgres -d ai_agent -f init_db.sql
-- ====================================================================

-- 开启 pgvector 向量扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- ====================================================================
-- 1. 会话主表 chat_session
--    存储会话基础信息、Agent全局结构化任务状态（断点续跑）
-- ====================================================================
CREATE TABLE IF NOT EXISTS chat_session (
    session_id   VARCHAR(64) PRIMARY KEY,
    user_id      VARCHAR(64) NOT NULL,
    title        VARCHAR(256) DEFAULT '',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_state  JSONB DEFAULT '{}'::jsonb  -- Agent任务状态、待办、已完成任务、关键结果
);

CREATE INDEX IF NOT EXISTS idx_session_user ON chat_session (user_id, created_at DESC);

-- ====================================================================
-- 2. 对话记忆表 chat_history（核心向量记忆表）
--    存储每轮对话原文、摘要、摘要向量、工具日志
-- ====================================================================
CREATE TABLE IF NOT EXISTS chat_history (
    id                 BIGSERIAL PRIMARY KEY,
    session_id         VARCHAR(64) NOT NULL REFERENCES chat_session(session_id) ON DELETE CASCADE,
    user_id            VARCHAR(64) NOT NULL,
    user_query         TEXT NOT NULL,
    llm_full_answer    TEXT,                              -- 完整详细答案（溯源）
    llm_summary        TEXT,                               -- 精简摘要（上下文投喂）
    summary_embedding  vector(1024),                      -- 摘要向量（智谱embedding-2，1024维）
    tool_call_log      JSONB DEFAULT '[]'::jsonb,         -- 工具调用日志
    rag_sources        JSONB DEFAULT '[]'::jsonb,          -- RAG召回来源
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 摘要向量索引（ivfflat 加速历史记忆语义检索，生产必备）
CREATE INDEX IF NOT EXISTS idx_history_summary_emb
ON chat_history USING ivfflat (summary_embedding vector_cosine_ops)
WITH (lists = 100);

-- 会话+用户联合索引（多用户隔离查询加速）
CREATE INDEX IF NOT EXISTS idx_history_session_user
ON chat_history (session_id, user_id, created_at DESC);

-- ====================================================================
-- 3. 工具调用日志表 tool_call_log（审计依据）
-- ====================================================================
CREATE TABLE IF NOT EXISTS tool_call_log (
    id           BIGSERIAL PRIMARY KEY,
    session_id   VARCHAR(64),
    user_id      VARCHAR(64),
    tool_name    VARCHAR(128) NOT NULL,
    tool_type    VARCHAR(32) DEFAULT 'local',             -- local / mcp
    tool_input   JSONB,
    tool_output  TEXT,
    status       VARCHAR(16) DEFAULT 'success',
    duration_ms  INTEGER DEFAULT 0,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tool_log_session ON tool_call_log (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tool_log_user ON tool_call_log (user_id, created_at DESC);

-- ====================================================================
-- 4. 人工审核队列表 review_queue（人机协同治理）
-- ====================================================================
CREATE TABLE IF NOT EXISTS review_queue (
    id           BIGSERIAL PRIMARY KEY,
    session_id   VARCHAR(64),
    history_id   BIGINT,
    user_id      VARCHAR(64),
    content      TEXT,
    review_type  VARCHAR(32),                              -- no_rag / tool_fail / low_sim / sensitive
    status       VARCHAR(16) DEFAULT 'pending',           -- pending / approved / rejected
    reviewer     VARCHAR(64),
    review_note  TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at  TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue (status, created_at DESC);

-- ====================================================================
-- 5. 标准答案库 standard_answers（人工审核沉淀，Few-shot样本反哺模型）
-- ====================================================================
CREATE TABLE IF NOT EXISTS standard_answers (
    id           BIGSERIAL PRIMARY KEY,
    question     TEXT NOT NULL,
    answer       TEXT NOT NULL,
    source_review_id BIGINT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_std_answer_question ON standard_answers USING gin (to_tsvector('simple', question));
