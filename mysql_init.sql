-- ==========================================================================
-- 企业级 AI Agent —— MySQL 身份与权限库（RBAC + 数据域 + 知识库授权 + 审计）
-- 执行方式：mysql -u root -p < mysql_init.sql
-- 依赖：MySQL 5.7+ / 8.0（用到 JSON 列与 SHA2 函数）
--
-- 【为什么身份权限放 MySQL，而不是 PG / Milvus】
--   MySQL  ：强事务、强一致，存"谁是谁、能干什么"，改权限即刻生效 —— 身份的真相源
--   PG     ：pgvector 存对话记忆向量（MySQL 没有等价的向量索引能力），继续保留
--   Milvus ：存知识库切片，按 kb_code 做逻辑隔离，检索时用 expr 过滤
-- 三者职责不重叠：MySQL=权限真相源，PG=记忆，Milvus=知识。
--
-- 【权限码规范】type:resource:action，判定支持末段 `*`
--   tool:calculator          可用计算器工具
--   tool:web_search          可用联网搜索（MCP）
--   tool:mcp:*               可用全部 MCP 远程工具
--   kb:finance               可读财务知识库
--   data:orders:dept         订单数据可见范围为"本部门"
--   api:admin:*              可访问管理端接口
--   model:deepseek-chat      可使用该模型（贵模型按角色授权）
-- ==========================================================================

CREATE DATABASE IF NOT EXISTS ai_agent_enterprise
  DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE ai_agent_enterprise;

-- 1. 租户（多租户 SaaS 预留；单公司场景也是 1 条记录）
CREATE TABLE IF NOT EXISTS sys_tenant (
    id           BIGINT       PRIMARY KEY AUTO_INCREMENT,
    tenant_code  VARCHAR(64)  NOT NULL UNIQUE COMMENT '租户编码',
    tenant_name  VARCHAR(128) NOT NULL COMMENT '租户名称',
    status       TINYINT      NOT NULL DEFAULT 1 COMMENT '1启用 0停用',
    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB COMMENT='租户表';

-- 2. 部门（parent_id 自引用形成树，用于数据域 dept/all 的上下级计算）
CREATE TABLE IF NOT EXISTS sys_dept (
    id         BIGINT       PRIMARY KEY AUTO_INCREMENT,
    tenant_id  BIGINT       NOT NULL,
    dept_code  VARCHAR(64)  NOT NULL COMMENT '部门编码',
    dept_name  VARCHAR(128) NOT NULL,
    parent_id  BIGINT       NULL COMMENT '上级部门，顶层为 NULL',
    status     TINYINT      NOT NULL DEFAULT 1,
    UNIQUE KEY uk_dept (tenant_id, dept_code)
) ENGINE=InnoDB COMMENT='部门表';

-- 3. 用户
CREATE TABLE IF NOT EXISTS sys_user (
    id         BIGINT       PRIMARY KEY AUTO_INCREMENT,
    tenant_id  BIGINT       NOT NULL,
    dept_id    BIGINT       NOT NULL,
    user_code  VARCHAR(64)  NOT NULL COMMENT '工号，Agent 侧的 user_id',
    real_name  VARCHAR(64)  NOT NULL,
    job_title  VARCHAR(128) DEFAULT '' COMMENT '岗位',
    user_level VARCHAR(16)  NOT NULL DEFAULT 'normal' COMMENT 'normal/advanced/admin，兼容旧的一维分级',
    status     TINYINT      NOT NULL DEFAULT 1 COMMENT '1在职 0离职（离职即全面拒绝）',
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_user (tenant_id, user_code),
    KEY idx_dept (dept_id)
) ENGINE=InnoDB COMMENT='用户表';

-- 4. API Key（一个用户多把 Key，支持轮换与吊销；只存哈希，不存明文）
CREATE TABLE IF NOT EXISTS sys_api_key (
    id           BIGINT      PRIMARY KEY AUTO_INCREMENT,
    user_id      BIGINT      NOT NULL,
    key_hash     CHAR(64)    NOT NULL UNIQUE COMMENT 'SHA2(明文,256)',
    key_name     VARCHAR(64) DEFAULT '' COMMENT '用途备注，如 cli / web',
    status       TINYINT     NOT NULL DEFAULT 1 COMMENT '1有效 0吊销',
    expires_at   DATETIME    NULL COMMENT 'NULL=长期有效',
    last_used_at DATETIME    NULL,
    created_at   DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_user (user_id)
) ENGINE=InnoDB COMMENT='API Key 表';

-- 5. 角色（data_scope 决定业务数据的可见范围，是 RBAC 之外的"数据域"维度）
CREATE TABLE IF NOT EXISTS sys_role (
    id          BIGINT      PRIMARY KEY AUTO_INCREMENT,
    tenant_id   BIGINT      NOT NULL,
    role_code   VARCHAR(64) NOT NULL COMMENT '角色编码',
    role_name   VARCHAR(128) NOT NULL,
    data_scope  VARCHAR(16) NOT NULL DEFAULT 'self' COMMENT 'self 仅本人 / dept 本部门 / all 全公司',
    mask_level  TINYINT     NOT NULL DEFAULT 0 COMMENT '脱敏强度 0不脱敏 1部分脱敏 2强脱敏',
    builtin     TINYINT     NOT NULL DEFAULT 0 COMMENT '1内置角色，不允许删除',
    UNIQUE KEY uk_role (tenant_id, role_code)
) ENGINE=InnoDB COMMENT='角色表';

CREATE TABLE IF NOT EXISTS sys_user_role (
    user_id BIGINT NOT NULL,
    role_id BIGINT NOT NULL,
    PRIMARY KEY (user_id, role_id)
) ENGINE=InnoDB COMMENT='用户-角色关联';

-- 6. 权限点
CREATE TABLE IF NOT EXISTS sys_permission (
    id         BIGINT      PRIMARY KEY AUTO_INCREMENT,
    perm_code  VARCHAR(128) NOT NULL UNIQUE COMMENT '权限码 tool:xxx / kb:xxx / data:xxx / api:xxx / model:xxx',
    perm_type  VARCHAR(16) NOT NULL COMMENT 'tool/kb/data/api/model',
    remark     VARCHAR(256) DEFAULT ''
) ENGINE=InnoDB COMMENT='权限点表';

CREATE TABLE IF NOT EXISTS sys_role_permission (
    role_id BIGINT NOT NULL,
    perm_id BIGINT NOT NULL,
    PRIMARY KEY (role_id, perm_id)
) ENGINE=InnoDB COMMENT='角色-权限关联';

-- 7. 知识库（逻辑库，对应 Milvus 单集合内的 kb_code 取值）
CREATE TABLE IF NOT EXISTS sys_kb (
    id              BIGINT      PRIMARY KEY AUTO_INCREMENT,
    tenant_id       BIGINT      NOT NULL,
    kb_code         VARCHAR(64) NOT NULL COMMENT 'Milvus 里 kb_code 字段的取值',
    kb_name         VARCHAR(128) NOT NULL,
    owner_dept_id   BIGINT      NULL COMMENT '归口部门',
    visibility      VARCHAR(16) NOT NULL DEFAULT 'private' COMMENT 'public 全员可见 / private 需授权',
    UNIQUE KEY uk_kb (tenant_id, kb_code)
) ENGINE=InnoDB COMMENT='知识库表';

-- 8. 知识库授权（可对部门/角色/个人授权，实现"财务只能看财务库"）
CREATE TABLE IF NOT EXISTS sys_kb_grant (
    id          BIGINT      PRIMARY KEY AUTO_INCREMENT,
    kb_id       BIGINT      NOT NULL,
    grant_type  VARCHAR(16) NOT NULL COMMENT 'dept / role / user',
    target_code VARCHAR(64) NOT NULL COMMENT '部门码/角色码/工号',
    can_read    TINYINT     NOT NULL DEFAULT 1,
    UNIQUE KEY uk_grant (kb_id, grant_type, target_code)
) ENGINE=InnoDB COMMENT='知识库授权表';

-- 9. 审计日志（谁、在什么时候、对什么资源、做了什么、结果如何）
CREATE TABLE IF NOT EXISTS sys_audit_log (
    id         BIGINT       PRIMARY KEY AUTO_INCREMENT,
    tenant_id  BIGINT       NULL,
    user_code  VARCHAR(64)  NOT NULL,
    dept_code  VARCHAR(64)  DEFAULT '',
    action     VARCHAR(64)  NOT NULL COMMENT 'chat / tool_call / kb_search / auth / admin',
    resource   VARCHAR(128) DEFAULT '' COMMENT 'tool:data_query / kb:finance ...',
    result     VARCHAR(16)  NOT NULL COMMENT 'allow / deny / error',
    detail     JSON         NULL COMMENT '入参、命中的权限、拒绝原因等',
    ip         VARCHAR(64)  DEFAULT '',
    session_id VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '会话ID，跨库串联 PG 的 chat_history/tool_call_log',
    task_id    VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '业务流水号，一轮对话的唯一号，跨库对账主键',
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_user_time (user_code, created_at),
    KEY idx_action (action, created_at),
    KEY idx_session (session_id),
    KEY idx_task (task_id)
) ENGINE=InnoDB COMMENT='审计日志表';

-- 10. LLM 用量（成本按人/部门归集，预算控制的数据源）
CREATE TABLE IF NOT EXISTS sys_llm_usage (
    id                BIGINT      PRIMARY KEY AUTO_INCREMENT,
    user_code         VARCHAR(64) NOT NULL,
    dept_code         VARCHAR(64) DEFAULT '',
    session_id        VARCHAR(64) DEFAULT '',
    task_id           VARCHAR(64) DEFAULT '' COMMENT '业务流水号，与审计表/PG同轮对话同一个号',
    model             VARCHAR(64) NOT NULL,
    prompt_tokens     INT         NOT NULL DEFAULT 0,
    completion_tokens INT         NOT NULL DEFAULT 0,
    total_tokens      INT         NOT NULL DEFAULT 0,
    latency_ms        INT         NOT NULL DEFAULT 0,
    created_at        DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_dept_time (dept_code, created_at),
    KEY idx_user_time (user_code, created_at)
) ENGINE=InnoDB COMMENT='LLM 用量表';

-- ==========================================================================
-- 老库升级：已存在的库不会重跑 CREATE TABLE，用 information_schema 判断后
-- 动态补列/补索引（MySQL 没有 ADD COLUMN IF NOT EXISTS，只能走 PREPARE）
-- 可重复执行
-- ==========================================================================
SET @col_exist := (
    SELECT COUNT(*) FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sys_audit_log' AND COLUMN_NAME = 'session_id'
);
SET @sql := IF(@col_exist = 0,
    'ALTER TABLE sys_audit_log ADD COLUMN session_id VARCHAR(64) NOT NULL DEFAULT '''' AFTER ip',
    'DO 0');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @idx_exist := (
    SELECT COUNT(*) FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sys_audit_log' AND INDEX_NAME = 'idx_session'
);
SET @sql2 := IF(@idx_exist = 0,
    'CREATE INDEX idx_session ON sys_audit_log (session_id)',
    'DO 0');
PREPARE stmt2 FROM @sql2;
EXECUTE stmt2;
DEALLOCATE PREPARE stmt2;

-- sys_audit_log 补 task_id 列 + 索引
SET @col_exist := (
    SELECT COUNT(*) FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sys_audit_log' AND COLUMN_NAME = 'task_id'
);
SET @sql := IF(@col_exist = 0,
    'ALTER TABLE sys_audit_log ADD COLUMN task_id VARCHAR(64) NOT NULL DEFAULT '''' AFTER session_id',
    'DO 0');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @idx_exist := (
    SELECT COUNT(*) FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sys_audit_log' AND INDEX_NAME = 'idx_task'
);
SET @sql2 := IF(@idx_exist = 0,
    'CREATE INDEX idx_task ON sys_audit_log (task_id)',
    'DO 0');
PREPARE stmt2 FROM @sql2;
EXECUTE stmt2;
DEALLOCATE PREPARE stmt2;

-- sys_llm_usage 补 task_id 列
SET @col_exist := (
    SELECT COUNT(*) FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sys_llm_usage' AND COLUMN_NAME = 'task_id'
);
SET @sql := IF(@col_exist = 0,
    'ALTER TABLE sys_llm_usage ADD COLUMN task_id VARCHAR(64) NOT NULL DEFAULT '''' AFTER session_id',
    'DO 0');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- ==========================================================================
-- 演示数据：1 个租户 / 4 个部门 / 8 个用户 / 6 个角色
-- 明文 API Key 只写在注释里，库里只落 SHA2 哈希
-- ==========================================================================
INSERT IGNORE INTO sys_tenant (id, tenant_code, tenant_name) VALUES
 (1, 'acme', 'ACME 科技');

INSERT IGNORE INTO sys_dept (id, tenant_id, dept_code, dept_name, parent_id) VALUES
 (1, 1, 'D_TECH',    '技术部',     NULL),
 (2, 1, 'D_FINANCE', '财务部',     NULL),
 (3, 1, 'D_HR',      '人力资源部', NULL),
 (4, 1, 'D_SALES',   '销售部',     NULL),
 (5, 1, 'D_GM',      '总经办',     NULL);

INSERT IGNORE INTO sys_user (id, tenant_id, dept_id, user_code, real_name, job_title, user_level) VALUES
 (1, 1, 1, 'U001', '张三', '后端工程师',   'normal'),
 (2, 1, 2, 'U002', '李四', '财务专员',     'advanced'),
 (3, 1, 2, 'U003', '王五', '财务经理',     'advanced'),
 (4, 1, 3, 'U004', '赵六', 'HRBP',         'advanced'),
 (5, 1, 4, 'U005', '孙七', '销售代表',     'normal'),
 (6, 1, 1, 'U006', '周八', '技术总监',     'advanced'),
 (7, 1, 5, 'U007', '管理员', 'IT管理员',   'admin'),
 (8, 1, 5, 'U008', '审计员', '内审',       'admin');

INSERT IGNORE INTO sys_role (id, tenant_id, role_code, role_name, data_scope, mask_level, builtin) VALUES
 (1, 1, 'R_EMPLOYEE',  '普通员工', 'self', 1, 1),
 (2, 1, 'R_FINANCE',   '财务专员', 'dept', 0, 1),
 (3, 1, 'R_HR',        '人力资源', 'dept', 0, 1),
 (4, 1, 'R_MANAGER',   '部门经理', 'dept', 0, 1),
 (5, 1, 'R_ADMIN',     '系统管理员', 'all', 0, 1),
 (6, 1, 'R_AUDITOR',   '审计员',   'all',  0, 1);

INSERT IGNORE INTO sys_user_role (user_id, role_id) VALUES
 (1, 1),                -- 张三：普通员工
 (2, 2),                -- 李四：财务专员
 (3, 2), (3, 4),        -- 王五：财务专员 + 部门经理
 (4, 3),                -- 赵六：HR
 (5, 1),                -- 孙七：普通员工
 (6, 1), (6, 4),        -- 周八：员工 + 部门经理
 (7, 5),                -- 管理员
 (8, 6);                -- 审计员

INSERT IGNORE INTO sys_permission (perm_code, perm_type, remark) VALUES
 -- 工具
 ('tool:calculator',        'tool', '本地计算器'),
 ('tool:data_query',        'tool', '业务数据查询'),
 ('tool:web_search',        'tool', 'MCP 联网搜索'),
 ('tool:get_current_time',  'tool', 'MCP 当前时间'),
 ('tool:knowledge_stats',   'tool', 'MCP 知识库统计'),
 ('tool:mcp:*',             'tool', '全部 MCP 远程工具'),
 -- 知识库
 ('kb:public',   'kb', '全员公开制度'),
 ('kb:tech',     'kb', '技术文档库'),
 ('kb:finance',  'kb', '财务知识库'),
 ('kb:hr',       'kb', '人力知识库'),
 ('kb:sales',    'kb', '销售知识库'),
 -- 数据域（与角色 data_scope 叠加取最大）
 ('data:users:self',      'data', '仅本人档案'),
 ('data:users:dept',      'data', '本部门人员档案'),
 ('data:orders:dept',     'data', '本部门订单'),
 ('data:orders:all',      'data', '全公司订单'),
 ('data:inventory:dept',  'data', '本部门库存'),
 ('data:inventory:all',   'data', '全公司库存'),
 -- 接口
 ('api:admin:*',  'api', '管理端接口'),
 ('api:audit:*',  'api', '审计接口'),
 -- 模型
 ('model:deepseek-chat', 'model', '标准模型'),
 ('model:deepseek-reasoner', 'model', '高成本推理模型');

-- 角色权限装配
-- 普通员工：本地工具 + 公开库 + 本人数据
INSERT IGNORE INTO sys_role_permission (role_id, perm_id) SELECT 1, id FROM sys_permission WHERE perm_code IN
 ('tool:calculator','kb:public','data:users:self','model:deepseek-chat');
-- 财务专员：本地工具 + 财务库 + 订单/库存本部门
INSERT IGNORE INTO sys_role_permission (role_id, perm_id) SELECT 2, id FROM sys_permission WHERE perm_code IN
 ('tool:calculator','tool:data_query','kb:public','kb:finance',
  'data:users:dept','data:orders:dept','data:inventory:dept','model:deepseek-chat');
-- HR：人力库 + 本部门人员档案
INSERT IGNORE INTO sys_role_permission (role_id, perm_id) SELECT 3, id FROM sys_permission WHERE perm_code IN
 ('tool:calculator','tool:data_query','kb:public','kb:hr','data:users:dept','model:deepseek-chat');
-- 部门经理：+ 全部 MCP 工具
INSERT IGNORE INTO sys_role_permission (role_id, perm_id) SELECT 4, id FROM sys_permission WHERE perm_code IN
 ('tool:calculator','tool:data_query','tool:mcp:*','kb:public','kb:tech','kb:finance','kb:hr','kb:sales',
  'data:users:dept','data:orders:all','data:inventory:all','model:deepseek-chat');
-- 管理员：全量
INSERT IGNORE INTO sys_role_permission (role_id, perm_id) SELECT 5, id FROM sys_permission;
-- 审计员：只读审计相关
INSERT IGNORE INTO sys_role_permission (role_id, perm_id) SELECT 6, id FROM sys_permission WHERE perm_code IN
 ('tool:calculator','tool:data_query','api:audit:*','kb:public','data:users:dept','data:orders:all','model:deepseek-chat');

INSERT IGNORE INTO sys_kb (id, tenant_id, kb_code, kb_name, owner_dept_id, visibility) VALUES
 (1, 1, 'public',  '公司通用制度', 5, 'public'),
 (2, 1, 'tech',    '技术文档库',   1, 'private'),
 (3, 1, 'finance', '财务知识库',   2, 'private'),
 (4, 1, 'hr',      '人力知识库',   3, 'private'),
 (5, 1, 'sales',   '销售知识库',   4, 'private');

-- 私有库按部门授权（public 库因 visibility=public 自动对全员开放）
INSERT IGNORE INTO sys_kb_grant (kb_id, grant_type, target_code, can_read) VALUES
 (2, 'dept', 'D_TECH',    1),
 (3, 'dept', 'D_FINANCE', 1),
 (3, 'role', 'R_ADMIN',   1),
 (4, 'dept', 'D_HR',      1),
 (5, 'dept', 'D_SALES',   1),
 (5, 'dept', 'D_GM',      1);

-- API Key（明文见注释，库中只存哈希）
INSERT IGNORE INTO sys_api_key (user_id, key_hash, key_name, status, expires_at) VALUES
 (1, SHA2('sk-acme-U001-tech-001',256),        '张三-开发机',   1, NULL),
 (2, SHA2('sk-acme-U002-finance-002',256),     '李四-财务',     1, NULL),
 (3, SHA2('sk-acme-U003-manager-003',256),     '王五-经理',     1, NULL),
 (4, SHA2('sk-acme-U004-hr-004',256),          '赵六-HR',       1, NULL),
 (5, SHA2('sk-acme-U005-sales-005',256),       '孙七-销售',     1, DATE_ADD(NOW(), INTERVAL -1 DAY)), -- 已过期，演示拒绝
 (6, SHA2('sk-acme-U006-director-006',256),    '周八-总监',     1, NULL),
 (7, SHA2('sk-acme-U007-admin-007',256),       '管理员',        1, NULL),
 (8, SHA2('sk-acme-U008-auditor-008',256),     '审计员',        1, NULL);

-- ==========================================================================
-- 常用运维 SQL
--   查某人权限：SELECT p.perm_code FROM sys_user u
--               JOIN sys_user_role ur ON ur.user_id=u.id
--               JOIN sys_role_permission rp ON rp.role_id=ur.role_id
--               JOIN sys_permission p ON p.id=rp.perm_id WHERE u.user_code='U002';
--   吊销 Key ：UPDATE sys_api_key SET status=0 WHERE key_hash=SHA2('sk-xxx',256);
--   部门账单：SELECT dept_code, SUM(total_tokens) FROM sys_llm_usage GROUP BY dept_code;
--   越权审计：SELECT * FROM sys_audit_log WHERE result='deny' ORDER BY id DESC LIMIT 50;
-- ==========================================================================

-- ==========================================================================
-- AI 只读账号（大厂做法：AI 服务对业务库只有 SELECT，没有写权限）
--   代码侧已用 MySQLClient(read_only=True) 拦了一道（只放行 SELECT/SHOW/EXPLAIN），
--   这里是数据库侧的最后一道防线，两层都要有：
--     只靠 GRANT  → 本地/测试环境常图省事用同一个高权限账号，等于没拦
--     只靠代码   → 换个人改代码、换个 SDK 就绕过去了
--   按需取消注释执行，密码按实际环境改；审计与用量表要写，所以不给 ai_ro，
--   那两处走主账号（MYSQL_USER）。
-- ==========================================================================
-- CREATE USER 'ai_ro'@'%' IDENTIFIED BY '换成强密码';
-- GRANT SELECT ON ai_agent_enterprise.sys_tenant  TO 'ai_ro'@'%';
-- GRANT SELECT ON ai_agent_enterprise.sys_dept    TO 'ai_ro'@'%';
-- GRANT SELECT ON ai_agent_enterprise.sys_user    TO 'ai_ro'@'%';
-- GRANT SELECT ON ai_agent_enterprise.sys_api_key TO 'ai_ro'@'%';
-- GRANT SELECT ON ai_agent_enterprise.sys_role    TO 'ai_ro'@'%';
-- GRANT SELECT ON ai_agent_enterprise.sys_user_role       TO 'ai_ro'@'%';
-- GRANT SELECT ON ai_agent_enterprise.sys_permission      TO 'ai_ro'@'%';
-- GRANT SELECT ON ai_agent_enterprise.sys_role_permission TO 'ai_ro'@'%';
-- GRANT SELECT ON ai_agent_enterprise.sys_kb             TO 'ai_ro'@'%';
-- GRANT SELECT ON ai_agent_enterprise.sys_kb_grant       TO 'ai_ro'@'%';
-- FLUSH PRIVILEGES;
