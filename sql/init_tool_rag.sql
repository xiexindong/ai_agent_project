-- ============================================================================
-- Tool-RAG 工具元数据表（pgvector）
-- ============================================================================
-- 【这张表解决什么问题】
--   远程/本地技能数量上百时，不能把全部工具 schema 塞给 LLM。
--   这里把「工具名 + 描述 + 参数名」编码成向量存下来，
--   运行时只按语义召回 Top3 候选，剩下的工具对模型不可见。
--
-- 【怎么用】
--   psql -U postgres -d ai_agent -f sql/init_tool_rag.sql
--   或在 Python 里由 ToolVectorStore.ensure_schema() 自动执行（幂等，重复执行无副作用）
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 【第 1 步】启用 pgvector 扩展
--   vector 不是 PostgreSQL 内置类型，不装的话下面建表会报
--   type "vector" does not exist。IF NOT EXISTS 保证重复执行不报错。
-- ----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

-- ----------------------------------------------------------------------------
-- 【第 2 步】工具元数据表
--   一行 = 一个可被调用的技能（本地函数工具或 MCP 远程工具）
--
--   字段大白话：
--     name            工具唯一名，调工具时用的就是这个（如 get_current_time）
--     description     工具描述，语义召回主要比对这一列（写清楚才召得准）
--     domain_tag      业务域标签（finance / hr / it / general …），用于按域过滤
--     source_type     local=本地函数工具；mcp_remote=MCP 远程技能
--     server_name     MCP 场景下该工具属于哪个 server，按需加载时用它连进程
--     parameters_json 参数名与说明（只存名字，不存完整 schema：召回阶段不需要）
--     enabled         False 的工具不参与召回（下线一个工具不用删数据）
--     embedding       description 的 1024 维向量，检索时比对的列
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tool_catalog (
    id              BIGSERIAL PRIMARY KEY,
    name            VARCHAR(128) NOT NULL UNIQUE,
    description     TEXT         NOT NULL DEFAULT '',
    domain_tag      VARCHAR(64)  NOT NULL DEFAULT 'general',
    source_type     VARCHAR(16)  NOT NULL DEFAULT 'local',
    server_name     VARCHAR(64)  NOT NULL DEFAULT '',
    parameters_json JSONB        NOT NULL DEFAULT '{}'::jsonb,
    enabled         BOOLEAN      NOT NULL DEFAULT TRUE,
    embedding       vector(1024),
    updated_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- 【第 3 步】索引
--   ① 向量索引：ivfflat + vector_cosine_ops（与检索 SQL 里的 <=> 余弦距离配套）
--      lists = 100 是经验值：适合 1 万~10 万行量级；
--      行数少于 lists*1000 时 PostgreSQL 可能仍走顺序扫描，结果一样正确、只是慢一点。
--      建完后记得 ANALYZE tool_catalog; 让优化器知道数据分布。
--   ② domain_tag 普通索引：按业务域过滤是高频条件
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_tool_catalog_embedding
    ON tool_catalog USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_tool_catalog_domain
    ON tool_catalog (domain_tag);

-- ----------------------------------------------------------------------------
-- 【依赖警告】向量维度 1024 必须与 config.py 的 EMBEDDING_DIMENSION 一致
--   本文件写死 1024；Python 侧 ToolVectorStore 建表时用 settings.embedding_dimension 拼 DDL。
--   两边不一致的后果：insert 时报 expected 1024 dimensions, not 768，
--   且报错发生在入库阶段，不在启动阶段 —— 属于改了配置才暴露的静默错位。
-- ----------------------------------------------------------------------------

-- 【手工验证用：看召回是否正确】（取消注释即可执行）
-- SELECT name, 1 - (embedding <=> '[0.01,0.02,...]'::vector) AS similarity
-- FROM tool_catalog
-- WHERE enabled = TRUE AND domain_tag = 'finance' AND NOT (name = ANY(ARRAY['submit_invoice']))
-- ORDER BY embedding <=> '[0.01,0.02,...]'::vector
-- LIMIT 3;
