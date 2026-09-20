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
    archived     SMALLINT DEFAULT 0,       -- 0正常 1已归档（离职员工数据保留期计时中）
    agent_state  JSONB DEFAULT '{}'::jsonb  -- Agent任务状态、待办、已完成任务、关键结果
);

CREATE INDEX IF NOT EXISTS idx_session_user ON chat_session (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_session_archived ON chat_session (archived, updated_at DESC);

-- 【老库升级】已存在的库不会重跑 CREATE TABLE，需幂等补列
ALTER TABLE chat_session ADD COLUMN IF NOT EXISTS archived SMALLINT DEFAULT 0;

-- ====================================================================
-- 2. 对话记忆表 chat_history（核心向量记忆表）
--    存储每轮对话原文、摘要、摘要向量、工具日志
-- ====================================================================
CREATE TABLE IF NOT EXISTS chat_history (
    id                 BIGSERIAL PRIMARY KEY,
    session_id         VARCHAR(64) NOT NULL REFERENCES chat_session(session_id) ON DELETE CASCADE,
    user_id            VARCHAR(64) NOT NULL,
    task_id            VARCHAR(64) DEFAULT '',                -- 业务流水号（跨库串联 MySQL 审计/用量）
    user_query         TEXT NOT NULL,
    llm_full_answer    TEXT,                              -- 完整详细答案（溯源）
    llm_summary        TEXT,                               -- 精简摘要（上下文投喂）
    summary_embedding  vector(1024),                      -- 摘要向量（智谱embedding-2，1024维）
    query_embedding    vector(1024),                      -- 问句向量（用户原话的向量，长期记忆主检索列）
    tool_call_log      JSONB DEFAULT '[]'::jsonb,         -- 工具调用日志
    rag_sources        JSONB DEFAULT '[]'::jsonb,          -- RAG召回来源
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 【为什么还要多一列 query_embedding？】
-- 当前长期记忆检索用的是 summary_embedding（原因见下），query_embedding 暂不参与检索，
-- 是给将来的「重复提问检测」预留的：那种场景需要问句比问句 + 更高的相似度阈值，
-- 与本表的「主题宽召回」不是一个目标，不应混用同一列。
-- 若确定不做重复检测，这一列可以删掉，不影响主流程。
--
-- 【为什么主检索用摘要向量而不是问句向量？】
-- 摘要格式是「问题主题 + 结论」，比问句原文信息更完整：
--   ① 换说法也能召回：现在问"最近业绩如何？" ↔ 历史摘要"Q3财报:营收增长12%"
--   ② 抗代词漂移：历史问句可能是"那它呢？"，原文脱离上下文语义是飘的，
--      但它的摘要是自包含的，任何时候都搜得到
--   ③ 只存一列，省一半 embedding 调用与存储

-- 问句向量索引（长期记忆主检索列，ivfflat 加速）
CREATE INDEX IF NOT EXISTS idx_history_query_emb
ON chat_history USING ivfflat (query_embedding vector_cosine_ops)
WITH (lists = 100);

-- 摘要向量索引（ivfflat 加速历史记忆语义检索，生产必备）
CREATE INDEX IF NOT EXISTS idx_history_summary_emb
ON chat_history USING ivfflat (summary_embedding vector_cosine_ops)
WITH (lists = 100);

-- 【老库升级】如果 chat_history 表在本次改动前已存在，上面的 CREATE TABLE 不会生效，
-- 需对已有库手工补一列（幂等，可重复执行）：
ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS query_embedding vector(1024);
ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS task_id VARCHAR(64) DEFAULT '';

-- 按 task_id 反查"这一轮对话"：跨库对账与排障的入口索引
CREATE INDEX IF NOT EXISTS idx_history_task ON chat_history (task_id);

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
    task_id      VARCHAR(64) DEFAULT '',                  -- 与 chat_history 同一轮的业务流水号
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

-- 【老库升级】tool_call_log 补 task_id 列（幂等）
ALTER TABLE tool_call_log ADD COLUMN IF NOT EXISTS task_id VARCHAR(64) DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_tool_log_task ON tool_call_log (task_id);

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
