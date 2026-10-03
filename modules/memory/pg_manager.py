"""
PostgreSQL + pgvector 管理模块
===============================
业务存储 + 对话记忆向量存储一体化管理。
- 会话主表 chat_session：会话信息、Agent结构化任务状态
- 对话记忆表 chat_history：原文、摘要、摘要向量、工具日志
覆盖边界4（上下文压缩）、Agent状态断点续跑。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【一句话看懂这个文件在干嘛】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
它就是一个「仓库管理员」，负责往 PostgreSQL 里存、取 Agent 的对话记录，
一共管 4 张表（4 个货架）：

    ┌─────────────────────────────────────────────────────────┐
    │ chat_session      会话表    = 一个「笔记本」的封面        │
    │                            记：谁在聊(session_id/user_id)│
    │                                话题名、Agent当前进展     │
    ├─────────────────────────────────────────────────────────┤
    │ chat_history      记忆表    = 笔记本里「每一页」          │
    │                            每问一句对话 = 新写 1 页       │
    │                            含：原文+摘要+摘要向量+工具日志│
    ├─────────────────────────────────────────────────────────┤
    │ tool_call_log     工具表    = 「工具使用流水账」          │
    │                            记：调了啥工具、传了啥、耗时   │
    ├─────────────────────────────────────────────────────────┤
    │ review_queue      审核表    = 「待复核清单」              │
    │                            可疑/高风险对话挑出来给人看    │
    └─────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【通俗例子：一次对话，这 4 张表怎么配合】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
假设你敲了 `python main.py chat`，问了三句话，全过程是这样的：

  【第 0 步：开新笔记本】
    程序启动 → create_session("abc123", "cli-user", "CLI对话")
    chat_session 表新增 1 行：
        session_id="abc123", user_id="cli-user", title="CLI对话", agent_state={}

  【第 1 步：你问「3*7 等于多少」】
    ① Agent 先「回忆」:
        get_short_term_history(...)      → 翻最近 3 页，看刚才聊了啥
        search_memory_by_vector(...)     → 翻历史所有页，找意思相近的
    ② Agent 回答并把这轮对话写进新一页:
        save_history(...)                → chat_history 表新增 1 行
            user_query      = "3*7 等于多少"
            llm_full_answer = "3*7 等于 21..."        ← 完整原文，给短期记忆用
            llm_summary     = "数学计算:3*7=21"       ← 一句话，给上下文投喂用
            summary_embedding = [0.12,-0.34,...]      ← 摘要的向量（长期记忆检索真正比对的列）
            query_embedding   = [0.09,-0.28,...]      ← 问句的向量（预留，当前不参与检索）
            tool_call_log   = [{"name":"calculator",...}]
    ③ 工具调用另记一笔流水:
        save_tool_log(...)               → tool_call_log 表新增 1 行
            tool_name="calculator", duration_ms=3, status="success"
    ④ 把当前进展写在封面:
        update_session_state("abc123", {...})  → agent_state 字段更新

  【第 2 步：你问「那 3*8 呢」】
    Agent 再次「回忆」，这次 search_memory_by_vector 会命中第 1 步那一页
    （因为「3*8」和「3*7」的向量很接近），于是模型知道你还在聊数学，回答更连贯。

  【第 3 步：程序 Ctrl+C 退出】
    close()  → 关闭游标和连接，相当于合上笔记本、锁门。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【summary_embedding 那个 1024 维向量到底是什么？】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
把「一句话」翻译成「一串数字（坐标）」，意思相近的话坐标挨得近：

    "3*7 等于多少"  → [0.12, -0.34, 0.56, ...] 共 1024 个数  ┐ 两个点距离很近
    "数学计算题"    → [0.11, -0.31, 0.52, ...]              ┘ （相似度 0.91）

    "今天午饭吃啥"  → [0.87, 0.42, -0.19, ...]               ← 这个点离得很远
                                                              （相似度 0.13）

好处：不用像 SQL 的 LIKE '%数学%' 那样死板地匹配关键词，
而是能识别「意思像不像」——这就是 search_memory_by_vector 的语义召回能力。
这也是为什么建表时要装 pgvector 扩展：普通 PostgreSQL 不认识 vector 这种列类型。
"""
# 【什么】标准库与项目内依赖
# 【作用】json 转 JSONB 列、MemoryException 抛记忆错误、handle_exception 统一兜异常、get_settings 读连接参数
import json
from typing import List, Optional
from datetime import datetime
from modules.common.logger import get_logger
from modules.common.exceptions import MemoryException, handle_exception
from config import get_settings

_logger = get_logger("pg")                          # 模块级日志器，前缀 pg


# 【什么】PG 连接与 4 张表的读写入口（单例）
# 【作用】全项目唯一的记忆持久化层：会话、对话页、工具流水、审核队列都走它
class PGManager:
    """PostgreSQL 连接与表结构管理（带pgvector扩展）。"""

    _instance = None                                # 类级单例对象；None 表示还没创建过

    # 【什么】控制实例只创建一次
    # 【作用】反复 PGManager() 拿到同一对象，连接与游标全进程共用一份
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # 【什么】首次实例化时才真正建连
    # 【作用】_connected 是"已连过"标记，单例下避免每次 PGManager() 都重连一次
    def __init__(self):
        if not hasattr(self, "_connected"):
            self._connect()
            self._connected = True                  # 连成功才打标记；失败时 _connect 已抛异常

    # 【什么】建立连接、装 pgvector 扩展、建表
    # 【作用】一次性把"能存向量"的三件事做齐；失败直接抛 MemoryException，不做降级
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
            self._conn.autocommit = False            # 关自动提交：每条写操作后由代码显式 commit

            self._cur = self._conn.cursor()          # 全类共用这一个游标（单线程模型）

            # ------------------------------------------------------------------
            # 【这一行在做什么】注册 pgvector 扩展，让 PostgreSQL 认识 vector 类型
            #   CREATE EXTENSION 是 PostgreSQL 的「装插件」语句。vector 不是 PG 内置
            #   类型，不装的话下面 _init_tables() 里建表时的
            #       summary_embedding vector(1024)
            #   会直接报 type "vector" does not exist，程序连启动都过不去。
            #
            # 【为什么必须带 IF NOT EXISTS】
            #   扩展是「数据库级别」的，装一次就永久存在。不带这个条件的话，
            #   第二次启动程序会报 extension "vector" already exists 直接崩；
            #   带上就是幂等的：装过就跳过，和 _init_tables() 里的建表一个套路。
            #
            # 【依赖警告】与数据库镜像硬绑定
            #   这行要求镜像里已经编译了 pgvector（本项目用 pgvector/pgvector:pg16）。
            #   若换成官方 postgres:16 镜像，会报 could not open extension control
            #   file，属于「换个镜像就炸」的隐性依赖。
            # ------------------------------------------------------------------
            self._cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # ------------------------------------------------------------------
            # 【为什么要单独 commit】因为上面关掉了自动提交
            #   self._conn.autocommit = False 意味着执行过的 SQL 只是留在
            #   「未提交事务」里，不 commit 就等于没真正落库。
            #
            # 【为什么在这里提交，而不是攒着一起提交】
            #   目的是把「装扩展」和「建表」拆成两个独立事务，缩小失败的影响范围。
            #     不提交（反例）：扩展创建会一直悬在同一个事务里，后面
            #       _init_tables() 若某条建表失败并回滚，扩展创建会被一起带走，
            #       等于白装；而且未结束的事务一直持有锁，第二个进程启动时执行
            #       同样的 CREATE EXTENSION 只能阻塞等待。
            #     提交（正例）：扩展一旦 commit 就永久生效，之后建表失败也不会
            #       把它连带回滚，重跑只需重建表，不用重装扩展。
            #
            # 【为什么必须在 register_vector 之前】
            #   register_vector() 要先去 pg_type 查出 vector 类型的 OID 才能注册
            #   适配器，所以顺序只能是「先让扩展生效，再注册」，两句不能调换。
            # ------------------------------------------------------------------
            self._conn.commit()

            # ------------------------------------------------------------------
            # 【上面这句 + 这两行，三层合起来在干什么】让「存向量」从不可能变成可能
            #
            #   第 1 层（上面那句 execute）—— 数据库服务端视角
            #       CREATE EXTENSION vector：让 PostgreSQL 自己认识 vector 类型。
            #       不装的话，建表时写 vector(1024) 会报 type "vector" does not exist。
            #
            #   第 2 层（本行）—— Python 客户端视角
            #       register_vector() 告诉 psycopg2「vector 类型在这个进程里该怎么收发」。
            #       它内部先查 pg_type 拿到 vector 类型的 OID，再注册两个适配器：
            #           写入方向：Python 的 list / numpy array  转成  数据库的 vector
            #           读取方向：数据库的 vector              转成  Python 的 numpy array
            #       不注册的后果（反例）：save_history() 传 summary_embedding=[0.12,-0.34,...]
            #       时 psycopg2 不认识 list 该转成什么，直接抛 can't adapt type 'list'，
            #       向量根本写不进去 —— 而这恰恰是长期记忆语义检索的核心字段。
            #
            #   第 3 层（本行）—— 业务视角
            #       _init_tables() 把 4 张表真正建出来，其中 chat_history 表的
            #       summary_embedding 列就声明为 vector(1024)。
            #       它内部跑完 DDL 后自己会 commit 一次（见该函数末尾），
            #       所以这里不需要再补 commit。
            #
            # 【为什么第 1 层必须先 commit，才能做第 2 层】
            #   register_vector 要查 pg_type 找 OID，而扩展必须已经生效可见；
            #   若扩展还悬在未提交事务里，这一步就可能查不到类型导致注册失败。
            #
            # 【一句话串起来】
            #   数据库认识 vector（服务端） + Python 会收发 vector（客户端）
            #   + 表里真有 vector 列（业务）= 能写入 1024 维摘要向量并按语义检索。
            # ------------------------------------------------------------------
            register_vector(self._conn)
            self._init_tables()
            _logger.info(f"PG连接成功: {s.pg_host}:{s.pg_port}/{s.pg_database}")
        except Exception as e:
            _logger.error(f"PG连接失败: {e}")
            raise MemoryException("PG连接失败", str(e))

    # 【什么】建表与建索引（幂等，可反复执行）
    # 【作用】把 5 张表 + 索引建齐；每条 DDL 带 IF NOT EXISTS，重复启动不报错
    def _init_tables(self):
        """创建生产级表结构（幂等）。

        【幂等是什么意思？】每条 SQL 都带 IF NOT EXISTS = "没有才创建，有了就跳过"，
        所以这个函数每次启动程序都会跑一遍，但永远不会报错、不会重复建表。
        类比前端的 `if (!el) createEl()`，或者直接理解为：可以放心重复执行。

        【本次要建的 4 张表 + 2 个索引，用大白话说】

            ① chat_session    会话表 —— 「笔记本封面」
                一个 session_id 就是一次完整对话（从你打开聊天窗口到关闭）
                字段大白话：
                    session_id   = 笔记本编号（主键，唯一）
                    user_id      = 这个笔记本属于谁
                    title        = 封面上写的标题
                    agent_state  = Agent 的「当前进展快照」（JSON），用于断点续跑

            ② chat_history    记忆表 —— 「笔记本里的每一页」（核心表）
                你每问一句，这里就多一行
                字段大白话：
                    user_query        = 你问了啥
                    llm_full_answer   = AI 完整回答（很长）
                    llm_summary       = 一句话摘要（很短）
                    summary_embedding = 摘要的向量（1024 个小数）← 长期记忆语义检索的主列
                    query_embedding   = 问句的向量（1024 个小数）← 预留，当前不参与检索
                    tool_call_log     = 这轮用了哪些工具
                    rag_sources       = 参考了知识库哪些文档
                特殊点：
                    REFERENCES chat_session(session_id) ON DELETE CASCADE
                    = 外键约束：删掉笔记本封面时，里面的页自动一起删，不会留下孤儿数据

            ③ tool_call_log   工具日志表 —— 「工具流水账」
                审计用：谁在什么时候调了什么工具、传了什么参数、耗时多少、成功与否

            ④ review_queue    审核队列表 —— 「待复核清单」
                可疑/高风险对话挑出来，status='pending' 等人工处理

            ⑤ idx_history_summary_emb  向量索引 —— 「给语义搜索加速」
                没有它：每次召回都要把表里每一行都算一遍距离（全表扫描，慢）
                有了它：ivfflat 把向量分成 100 个簇，先定位到相近的簇再细找（快很多）
                类比：书没有目录只能一页页翻；有了目录直接跳到相关章节

            ⑥ idx_history_session_user  联合索引 —— 「给按用户查历史加速」
                查询条件常是 "某个会话 + 某个用户 + 按时间倒序"，建联合索引一次搞定
                类比：书的目录 + 按作者排序
        """
        s = get_settings()
        dim = s.embedding_dimension                 # 向量维度，默认 1024，决定 vector(N) 里的 N
        # 【什么】DDL 列表：建表 / 加列 / 建索引，按顺序逐条执行
        # 【作用】新增列一律用 ALTER ... IF NOT EXISTS，老库升级不必重建表，数据不丢
        ddl = [
            # ① chat_session 会话表：一本笔记本的封面
            f"""
            CREATE TABLE IF NOT EXISTS chat_session (
                session_id   VARCHAR(64) PRIMARY KEY,
                user_id      VARCHAR(64) NOT NULL,
                title        VARCHAR(256) DEFAULT '',
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                archived     SMALLINT DEFAULT 0,
                agent_state  JSONB DEFAULT '{{}}'::jsonb
            );
            """,
            # 老库补列：archived 离职归档标记（0 未归档 / 1 已归档）
            """
            ALTER TABLE chat_session ADD COLUMN IF NOT EXISTS archived SMALLINT DEFAULT 0;
            """,
            # 索引：匹配「归档标记 + 更新时间倒序」，加速冷热分离与过期清理的取数
            """
            CREATE INDEX IF NOT EXISTS idx_session_archived
            ON chat_session (archived, updated_at DESC);
            """,
            # ② chat_history 对话记忆表：笔记本里的每一页（核心表）
            f"""
            CREATE TABLE IF NOT EXISTS chat_history (
                id                 BIGSERIAL PRIMARY KEY,
                session_id         VARCHAR(64) NOT NULL REFERENCES chat_session(session_id) ON DELETE CASCADE,
                user_id            VARCHAR(64) NOT NULL,
                task_id            VARCHAR(64) DEFAULT '',
                user_query         TEXT NOT NULL,
                llm_full_answer    TEXT,
                llm_summary        TEXT,
                summary_embedding  vector({dim}),
                query_embedding    vector({dim}),
                tool_call_log      JSONB DEFAULT '[]'::jsonb,
                rag_sources        JSONB DEFAULT '[]'::jsonb,
                created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # 老库补列：query_embedding 问句向量（当前不参与检索，预留给重复提问检测）
            f"""
            ALTER TABLE chat_history
            ADD COLUMN IF NOT EXISTS query_embedding vector({dim});
            """,
            # ivfflat 向量索引（问句向量）：lists=100 表示把向量分成 100 个簇
            f"""
            CREATE INDEX IF NOT EXISTS idx_history_query_emb
            ON chat_history USING ivfflat (query_embedding vector_cosine_ops)
            WITH (lists = 100);
            """,
            # ivfflat 向量索引（摘要向量）：长期记忆语义召回走的正是它
            f"""
            CREATE INDEX IF NOT EXISTS idx_history_summary_emb
            ON chat_history USING ivfflat (summary_embedding vector_cosine_ops)
            WITH (lists = 100);
            """,
            # 联合索引：匹配「某会话 + 某用户 + 时间倒序」这一常用取数条件
            f"""
            CREATE INDEX IF NOT EXISTS idx_history_session_user
            ON chat_history (session_id, user_id, created_at DESC);
            """,
            # 老库补列：task_id 跨库对账流水号
            """
            ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS task_id VARCHAR(64) DEFAULT '';
            """,
            # ------------------------------------------------------------------
            # 【记忆治理两列】memory_status / superseded_by
            #   memory_status : active(正常可召回) / superseded(已被新记忆取代，不再召回)
            #   superseded_by : 被哪一条新记忆取代（chat_history.id），冲突可追溯
            #
            # 【解决什么】同一主题的新旧记忆并存（如"我住在上海"→"我搬到北京"），
            #   不加状态列时两条都会被召回，模型两头都说、互相矛盾。
            #   冲突处理：相似度 ≥ MEMORY_CONFLICT_THRESHOLD(0.90) 判为同一主题，
            #   旧的那条置 superseded 并记下新条 id，召回 SQL 只取 active。
            # ------------------------------------------------------------------
            """
            ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS memory_status VARCHAR(16) DEFAULT 'active';
            """,
            """
            ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS superseded_by BIGINT;
            """,
            # 索引：召回 SQL 的过滤条件是 (user_id, memory_status)，建联合索引加速
            """
            CREATE INDEX IF NOT EXISTS idx_history_user_status
            ON chat_history (user_id, memory_status, created_at DESC);
            """,
            # 索引：按 task_id 反查某一笔业务流水涉及哪些对话页
            """
            CREATE INDEX IF NOT EXISTS idx_history_task ON chat_history (task_id);
            """,
            # ③ tool_call_log 工具流水表：记调用入参、出参、耗时、状态
            f"""
            CREATE TABLE IF NOT EXISTS tool_call_log (
                id           BIGSERIAL PRIMARY KEY,
                session_id   VARCHAR(64),
                user_id      VARCHAR(64),
                task_id      VARCHAR(64) DEFAULT '',
                tool_name    VARCHAR(128) NOT NULL,
                tool_type    VARCHAR(32) DEFAULT 'local',
                tool_input   JSONB,
                tool_output  TEXT,
                status       VARCHAR(16) DEFAULT 'success',
                duration_ms  INTEGER DEFAULT 0,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # 老库补列：task_id 与 chat_history 对齐，供跨库对账
            """
            ALTER TABLE tool_call_log ADD COLUMN IF NOT EXISTS task_id VARCHAR(64) DEFAULT '';
            """,
            # 索引：按 task_id 反查某一笔业务流水调用了哪些工具
            """
            CREATE INDEX IF NOT EXISTS idx_tool_log_task ON tool_call_log (task_id);
            """,
            # ④ review_queue 审核队列表：pending / approved / rejected 三态
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
            # ⑤ standard_answers 标准答案表：审核沉淀下来反哺 Few-shot
            """
            CREATE TABLE IF NOT EXISTS standard_answers (
                id               BIGSERIAL PRIMARY KEY,
                question         TEXT NOT NULL,
                answer           TEXT NOT NULL,
                source_review_id BIGINT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            # GIN 全文索引：simple 配置不分词，中文场景实际靠 ILIKE 匹配（见 match_standard_answers）
            """
            CREATE INDEX IF NOT EXISTS idx_std_answer_question
            ON standard_answers USING gin (to_tsvector('simple', question));
            """,
        ]
        # 【执行与提交】逐条执行后统一 commit 一次
        #   任一条 DDL 失败会走 except 抛 MemoryException，已执行的因未提交而回滚
        for sql in ddl:
            self._cur.execute(sql)
        self._conn.commit()
        _logger.info("PG表结构初始化完成（chat_session, chat_history, tool_call_log, review_queue）")

    # 【什么】写操作前的心跳探活，断了就重连
    # 【作用】每个对外方法开头都调一次，避免"连接被 PG 掐断后第一次操作直接报错"
    def _reconnect_if_needed(self):
        """连接断开时自动重连（心跳探活 + 失败自愈）。"""
        try:
            self._cur.execute("SELECT 1")           # 最轻的探活语句：能执行就说明连接还活着
        except Exception:
            _logger.warning("PG连接断开，尝试重连...")
            self._connected = False
            self._connect()


    # 【什么】建一个新会话（给新笔记本贴封面）
    # 【作用】每轮对话落库前先确保会话存在；异常向外抛，由上层决定是否降级
    @handle_exception(default=None, log_msg="创建会话异常", reraise=True)
    def create_session(self, session_id: str, user_id: str, title: str = "") -> None:
        """创建新会话（相当于给新笔记本贴封面）。

        【调用例子】
            pg.create_session("abc123", "cli-user", "CLI对话")

            chat_session 表新增一行：
                session_id | user_id  | title   | agent_state
                abc123     | cli-user | CLI对话 | {}

        【ON CONFLICT (session_id) DO NOTHING 是什么意思？】
            如果这个编号已经存在了，就什么都不做，也不报错。
            不写这句 → 重复调用会直接崩："duplicate key value violates unique constraint"
            类比 JS：Map.set 之前先 has() 检查一遍；或理解为 upsert 的"只插不覆盖"版本。
        """
        self._reconnect_if_needed()
        self._cur.execute(
            "INSERT INTO chat_session (session_id, user_id, title) VALUES (%s, %s, %s) "
            "ON CONFLICT (session_id) DO NOTHING",
            (session_id, user_id, title),   # 依次填 SQL 里 3 个 %s：会话号 / 用户号 / 标题
        )
        self._conn.commit()
        _logger.info(f"会话创建: session={session_id}, user={user_id}")

    # 【什么】判断会话是否存在且属于该用户
    # 【作用】server.py 的归属校验专用：只查记录存在性，与 agent_state 内容无关
    @handle_exception(default=False, log_msg="校验会话归属异常")
    def session_belongs_to(self, session_id: str, user_id: str) -> bool:
        """会话存在且归属该用户 → True；不存在/不属于/异常 → False。

        【为什么不复用 get_session_state？】它返回的是 agent_state 内容，
            新会话 agent_state 为 NULL 时会被判成"不存在"，续聊被误拦（403）。
            归属校验只关心"有没有这条记录"，不看状态字段。
        """
        self._reconnect_if_needed()
        self._cur.execute(
            "SELECT 1 FROM chat_session WHERE session_id = %s AND user_id = %s",
            (session_id, user_id),
        )
        return self._cur.fetchone() is not None

    # 【什么】列出某用户的会话（侧边栏"会话列表"的取数口）
    # 【作用】只取本人、未归档的会话，按最近活跃倒序；异常返回 []，列表页宁可空也不报错
    @handle_exception(default=[], log_msg="读取会话列表异常")
    def list_user_sessions(self, user_id: str, limit: int = 50) -> list:
        """读取当前用户的会话列表（会话隔离：WHERE 必须带 user_id）。

        【返回】[{session_id, title, created_at, updated_at}, ...]，按 updated_at 倒序
        【为什么只查封面表 chat_session 不连 chat_history】
            列表只需要"有哪些会话、叫什么、什么时候聊过"，
            连历史表会把同一会话放大成 N 行还要 DISTINCT，纯浪费；
            点开某一条时前端才调 /history/{session_id} 懒加载明细。
        【archived=0 过滤】离职归档/过期清理会话不进用户列表（冷热分离）。
        """
        self._reconnect_if_needed()
        self._cur.execute(
            """SELECT session_id, title, created_at, updated_at
               FROM chat_session
               WHERE user_id = %s AND archived = 0
               ORDER BY updated_at DESC
               LIMIT %s""",
            (user_id, limit),
        )
        cols = ("session_id", "title", "created_at", "updated_at")
        return [dict(zip(cols, row)) for row in self._cur.fetchall()]

    # 【什么】把 Agent 当前进展快照写进会话封面
    # 【作用】断点续跑的落盘端；异常向外抛，让上层知道状态没存成功
    @handle_exception(default=None, log_msg="更新会话状态异常", reraise=True)
    def update_session_state(self, session_id: str, agent_state: dict) -> None:
        """更新 Agent 结构化任务状态（支持断点续跑）。

        【通俗理解】把 Agent 此刻的「脑子」拍张快照存起来——
        推理到第几轮、问了什么、调过哪些工具、拿到什么结果，
        全都记进笔记本封面的 agent_state 字段。
        程序崩了重启后，用 get_session_state 把快照读回来就能接着算，
        而不是从头再来一次（这就是文档里说的「断点续跑」）。

        【调用例子】
            pg.update_session_state("abc123", {
                "user_query": "3*7 等于多少",
                "iteration": 2,
                "tool_call_logs": [{"name": "calculator", "result": "21"}],
                "finished": False,
            })
            → chat_session 表 abc123 那行的 agent_state 被整段替换成上面这个 JSON
            ⚠ 注意是「整段覆盖」不是「合并」，所以调用方必须传完整状态，传缺字段会丢数据

        【为什么 agent_state 要 json.dumps？】
            Python 里是 dict，PG 里是 JSONB 列，中间必须转成 JSON 字符串才能传，
            %s 占位符收到的就是这个字符串。
            ensure_ascii=False → 中文保持原样（否则 "你好" 会被存成 "\\u4f60\\u597d"）
        """
        self._reconnect_if_needed()
        self._cur.execute(
            "UPDATE chat_session SET agent_state = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE session_id = %s",
            (json.dumps(agent_state, ensure_ascii=False), session_id),  # dict 转 JSON 字符串才能进 JSONB 列
        )
        self._conn.commit()

    # 【什么】读取 Agent 任务状态（断点续跑的读回端）
    # 【作用】同时充当越权校验：WHERE 带 user_id，不属于你的会话返回空 dict
    @handle_exception(default={}, log_msg="读取会话状态异常")
    def get_session_state(self, session_id: str, user_id: str) -> dict:
        """读取 Agent 任务状态（断点续跑用）。

        【调用例子】
            state = pg.get_session_state("abc123", "cli-user")
            命中  → {"iteration": 2, "tool_call_logs": [...], "finished": False}
            没命中 → {}

        【为什么 WHERE 里要带上 user_id？】
            安全隔离：防止用户 A 拿着用户 B 的 session_id 把别人的对话状态读走。
            正因如此，server.py 的 /chat 可以直接拿它当"越权校验"用：
                if not pg.get_session_state(session_id, user_id):
                    raise HTTPException(403, "无权访问该会话")
            返回 {}（假值）就意味着「要么没这个会话，要么不属于你」。

        【返回值为什么写成 row[0] if row and row[0] else {}？】
            两种情况都算"没查到"，统一返回空 dict 兜底，调用方不用判空：
                row is None         → 压根没这条会话记录
                row[0] 是空的 {}    → 有记录但状态是空的（{} 是假值）
        """
        self._reconnect_if_needed()
        self._cur.execute(
            "SELECT agent_state FROM chat_session WHERE session_id = %s AND user_id = %s",
            (session_id, user_id),
        )
        row = self._cur.fetchone()                  # 取一行；没有匹配行时返回 None

        return row[0] if row and row[0] else {}     # 没记录或状态为空都兜底成 {}，调用方不用判空


    # 【什么】把一轮对话写进 chat_history（四数据落库）
    # 【作用】记忆的唯一写入口；异常向外抛，上层据此降级，不能静默丢记忆
    @handle_exception(default=None, log_msg="保存对话记忆异常", reraise=True)
    def save_history(
        self,
        session_id: str,                        # 会话号，外键指向 chat_session
        user_id: str,                           # 用户号，多用户隔离依据
        user_query: str,                        # 用户原话（NOT NULL 列）
        llm_full_answer: str,                   # 完整原文，短期记忆读它
        llm_summary: str,                       # 一句话摘要，上下文投喂读它
        summary_embedding: list,                # 摘要向量，长期记忆主检索列
        tool_call_log: list | None = None,      # 本轮工具流水，None 存成 []
        rag_sources: list | None = None,        # 本轮知识库来源，None 存成 []
        query_embedding: list | None = None,    # 问句向量，预留给重复提问检测
        task_id: str = "",                      # 跨库对账流水号，留空则从请求上下文取
    ) -> int | None:
        """
        保存一轮对话到 chat_history 表（四数据落库）。

        【为什么是"四数据"？】一次对话要存 4 类数据，各有用途：
            llm_full_answer    : 完整原文 → 短期记忆用（最近 N 轮塞回 prompt）
            llm_summary        : 一句话摘要 → 上下文投喂 LLM（省 token）
            summary_embedding  : 摘要向量 → 长期记忆【主检索列】（主题宽召回）
            query_embedding    : 问句向量 → 当前不参与检索，预留给「重复提问检测」
            缺任一项后续记忆能力都会残废：缺原文没法看最近聊了啥，
            缺向量没法语义召回历史对话。

        【落库前后的数据变化】
            输入参数：
                session_id="abc123", user_id="cli-user",
                task_id="T-20260920-3f9c1a2b",   （跨库业务流水号，留空则自动取请求上下文）
                user_query="3*7 等于多少",
                llm_full_answer="3*7 等于 21...",   （长文本）
                llm_summary="数学计算:3*7=21",       （一句话）
                summary_embedding=[0.12, -0.34, ...]（1024 维向量，摘要的）
                query_embedding=[0.09, -0.28, ...]  （1024 维向量，问句的）
                tool_call_log=[{"name":"calculator","args":...,"result":"21"}]

            数据转换：
                summary_embedding → np.array(dtype=float32)
                query_embedding   → np.array(dtype=float32)
                                  （pgvector 要求 numpy 数组，不能是 list）
                tool_call_log → JSON 字符串（JSONB 列存）
                rag_sources  → JSON 字符串

            落库后（chat_history 表新增 1 行）：
                id=42, session_id="abc123", ... created_at=NOW
            返回: 42（新行的主键 ID，用于后续 add_to_review_queue 关联）
        """
        self._reconnect_if_needed()
        import numpy as np
        emb = np.array(summary_embedding, dtype=np.float32) if summary_embedding else None    # pgvector 只认 numpy 数组
        q_emb = np.array(query_embedding, dtype=np.float32) if query_embedding else None      # 问句向量同理，可为空
        # ------------------------------------------------------------------
        # 【task_id 从哪来】显式传了就用显式值，没传就从请求上下文取。
        #   它是跨库对账的主键：MySQL 的 sys_audit_log / sys_llm_usage 与
        #   PG 的 chat_history 必须落同一个号，否则定时对账会判定为"单边缺失"。
        #   取不到就是空串——宁可空着，也不要造一个假号（假号比空号更难查）。
        # ------------------------------------------------------------------
        from modules.common import context as req_ctx
        task_id = task_id or req_ctx.get_task_id()
        self._cur.execute(
            """INSERT INTO chat_history
               (session_id, user_id, task_id, user_query, llm_full_answer, llm_summary,
                summary_embedding, query_embedding, tool_call_log, rag_sources)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (
                session_id, user_id, task_id, user_query,     # 对应 SQL 前 4 个 %s
                llm_full_answer, llm_summary, emb, q_emb,     # 对应后 4 个 %s，两个向量可为 None
                json.dumps(tool_call_log or [], ensure_ascii=False),    # 转 JSON 字符串存 JSONB 列
                json.dumps(rag_sources or [], ensure_ascii=False),      # 同上；ensure_ascii=False 保中文原样
            ),
        )
        row = self._cur.fetchone()                  # RETURNING id 的返回行，拿新主键
        # ------------------------------------------------------------------
        # 【同事务刷新会话封面的 updated_at】
        #   不刷新的话，多轮续聊的会话在"会话列表"里永远按创建时间排队，
        #   最近在聊的沉底、早就废弃的反而排在前面。
        #   放在 commit 之前：与本轮历史同生共死，不会出现"列表时间新了但内容没存上"。
        # ------------------------------------------------------------------
        self._cur.execute(
            "UPDATE chat_session SET updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
            (session_id,),
        )
        self._conn.commit()
        history_id = row[0] if row else None        # 新对话页的 id，供审核队列回连
        _logger.info(f"对话记忆保存: history_id={history_id}, session={session_id}")
        return history_id

    # 【什么】取当前会话最近 N 轮完整对话（短期记忆）
    # 【作用】让模型接得上话；异常返回 []，宁可没有上下文也不中断对话
    @handle_exception(default=[], log_msg="读取短期记忆异常")
    def get_short_term_history(self, session_id: str, user_id: str, rounds: int = 3) -> list:
        """读取最近 N 轮完整对话（短期记忆）。

        【通俗理解：短期记忆 vs 长期记忆，两者分工完全不同】
            短期记忆（本方法）  = 「刚才咱聊到哪了」
                只看当前这一个会话的最近 3 页，原样搬全文
                作用：让模型知道上一句在说什么，能接得上话（比如你追问"那它呢？"）

            长期记忆（search_memory_by_vector）= 「很久以前聊过类似的吗」
                跨所有会话，按语义相似度找（哪怕措辞完全不一样）
                作用：让模型记住你的偏好和习惯（比如上次说过你是做财务的）

            Agent 每次回答前，这两份记忆会拼在一起塞进 prompt。

        【调用例子】
            pg.get_short_term_history("abc123", "cli-user", rounds=3)

            数据库里实际存了 5 页：
                第1页 "你好"          (12:00)
                第2页 "3*7等于多少"    (12:01)
                第3页 "那3*8呢"        (12:02)
                第4页 "推荐个火锅"      (12:03)
                第5页 "公司报销流程"    (12:04)  ← 最新

            SQL 的 ORDER BY created_at DESC LIMIT 3 先取出最新的 3 条：
                [第5页, 第4页, 第3页]         （倒序：新→旧）
            再 rows.reverse() 翻转成正序：
                [第3页, 第4页, 第5页]         （正序：旧→新）← 这才是给模型看的自然语序

            返回：
                [
                    {"query":"那3*8呢",   "answer":"3*8 等于 24", "summary":"数学计算:3*8=24", "time":...},
                    {"query":"推荐个火锅", "answer":"海底捞...",   "summary":"餐饮推荐",        "time":...},
                    {"query":"公司报销流程","answer":"需提交发票...","summary":"报销流程咨询",   "time":...},
                ]

        【为什么要 reverse 翻转一次？】
            SQL 里必须 DESC 排序才能 LIMIT 到"最新 3 条"（ASC 会取到最老的），
            但拿到的顺序是反的，模型读着别扭，所以翻转成正序再返回。
        """
        self._reconnect_if_needed()
        self._cur.execute(
            """SELECT user_query, llm_full_answer, llm_summary, created_at
               FROM chat_history
               WHERE session_id = %s AND user_id = %s
               ORDER BY created_at DESC LIMIT %s""",
            (session_id, user_id, rounds),
        )
        rows = self._cur.fetchall()
        rows.reverse()                              # SQL 为取"最新 N 条"用 DESC，这里翻回旧→新的自然语序
        return [
            {"query": r[0], "answer": r[1], "summary": r[2], "time": r[3]}   # 顺序对应 SELECT 的 4 列
            for r in rows
        ]

    # 【什么】按摘要向量语义召回历史对话（长期记忆）
    # 【作用】跨会话找语义相近的旧记忆；异常返回 []，不打断本轮对话
    @handle_exception(default=[], log_msg="向量召回历史记忆异常")
    def search_memory_by_vector(self, query_embedding: list, user_id: str, top_k: int = 3) -> list:
        """
        基于摘要向量语义召回历史对话（长期记忆）。

        【比的是哪一列？—— summary_embedding（摘要向量）】
            传入的 query_embedding 是「当前问句」的向量（embed(user_query) 得来），
            但被比对的是历史【摘要】的向量。这是有意选择，不是笔误：

                当前问句  "帮我算一下12*8"      → 向量 A
                历史摘要  "数学计算:12*8=96"     → 向量 B

            为什么不用问句比问句？因为摘要更「完整」，它同时含【问题主题 + 结论】：
                ① 换说法也能召回：
                   现在问"最近业绩如何？" ↔ 历史摘要"Q3财报:营收增长12%"  ✅
                   （"业绩"和"营收"语义近，但历史问句原文若写的是"财报怎么样"就没这么贴）
                ② 抗代词漂移（关键）：
                   历史问句可能是"那它呢？""再算一个"——脱离上下文向量是飘的；
                   但它的摘要"数学计算:12*9=108"是自包含的，任何时候都搜得到。
                ③ 只存一列，省一半 embedding 调用与存储。
            代价：属于「非对称检索」（短查询 vs 结论句），相似度会被稀释一些。
            缓解手段是阈值别设太高（当前 0.65，见 config.py similarity_threshold）。

        【query_embedding 列是干什么的？】
            表里另有一列 query_embedding（历史问句的向量），本方法【不使用】它。
            它留给将来的「重复提问检测」用：那种场景需要问句比问句、且阈值要更高
            （0.9+），跟本方法的「主题宽召回」是两个目标，不能混在一条 SQL 里。

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
        emb = np.array(query_embedding, dtype=np.float32)   # 问句向量，与历史摘要向量做余弦距离比较
        # ------------------------------------------------------------------
        # 【记忆状态过滤】COALESCE(memory_status,'active')='active'
        #   只召回"活着"的记忆；已被新记忆取代（superseded）的不进结果。
        #   COALESCE 兜底：万一有行还没补上该列值（NULL），按 active 对待，
        #   与 ALTER TABLE ... DEFAULT 'active' 的语义一致。
        #
        # 【不过滤会怎样】用户先说"我住在上海"再说"我搬到北京"，两条都在库里；
        #   问"我住哪"时两条都被召回，模型两头都说——这就是"记忆互相矛盾"的根源。
        # ------------------------------------------------------------------
        self._cur.execute(
            """SELECT user_query, llm_summary, llm_full_answer,
                      1 - (summary_embedding <=> %s) AS similarity, created_at
               FROM chat_history
               WHERE user_id = %s AND summary_embedding IS NOT NULL
                 AND COALESCE(memory_status, 'active') = 'active'
               ORDER BY summary_embedding <=> %s
               LIMIT %s""",
            (emb, user_id, emb, top_k),     # %s 顺序：相似度里的向量 → user_id → 排序用的向量 → 条数上限

        )
        # ------------------------------------------------------------------
        # execute 只负责「执行」，返回 None；结果行留在游标的结果集里，要靠
        # fetch 系列搬回 Python。官方对 fetchall 的说明是 "Return all the
        # REMAINING rows" —— remaining 说明取过的不会再给。
        #
        # 实测（本项目真实库）：
        #   不 fetch 拿不到数据，重复 fetchall 得 []；
        #   不 fetch 也能拿到的只有 description（列名）和 rowcount（行数）。
        #   ⚠ INSERT 之后 fetchall 是直接抛 ProgrammingError，不是返回 []。
        #   ⚠ 没取就执行下一条 SQL，旧结果集被静默丢弃，fetchall 拿到新结果。
        #
        # fetchone() / fetchmany(n) / fetchall() 取的是同一份结果集，区别只在内存；
        # 本方法有 LIMIT 兜底，用 fetchall 划算。返回值是 tuple，故用 r[0]、r[1] 取。
        #
        # 两个隐患：
        #   1) 全类共用 self._cur（第 121 行）：单线程无碍，多线程下方法 B 的
        #      execute 会顶掉方法 A 的结果集，A 的 fetchall 拿到错数据且不报错。
        #   2) 客户端游标：execute 时结果已全读进本机内存，fetchall 省不了内存，
        #      要流式取数需用命名游标 conn.cursor(name=...)。批量任务留意这点。
        # ------------------------------------------------------------------
        rows = self._cur.fetchall()
        return [
            {
                "query": r[0], "summary": r[1], "answer": r[2],      # 原问句 / 摘要 / 完整原文
                "similarity": round(float(r[3]), 4), "time": r[4],   # 相似度保留 4 位小数；r[4] 是写入时间
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 【记忆冲突处理】入库前的两步：查冲突 → 旧的置 superseded
    #
    # 【为什么需要】同一个主题（如"用户的住址"）先后存了两条记忆时，
    #   向量召回会把两条都带回来塞进 prompt，模型输出自相矛盾。
    #   处理策略：新记忆入库前先查同用户、相似度 ≥ 阈值(0.90)的旧记忆，
    #   有 → 旧的置 superseded（不物理删，留审计痕迹），新条照常入库。
    #
    # 【为什么不物理删除】"被取代"和"记错要删"是两回事：
    #   置状态可追溯（superseded_by 指向新条 id），真删了就查无对证。
    # ------------------------------------------------------------------
    # 【什么】查同用户中与新记忆高度相似的旧记忆（冲突检测）
    # 【作用】相似度 ≥ threshold 的第一条旧记忆即判定为冲突；无冲突返回 None
    @handle_exception(default=None, log_msg="记忆冲突检测异常")
    def find_conflict_memory(
        self, summary_embedding: list, user_id: str,
        threshold: float = 0.90, exclude_id: int | None = None,
    ) -> dict | None:
        """
        冲突检测：找同用户里与新摘要向量最相近的一条 active 记忆。

        【输入输出对照】
            输入: summary_embedding=[新"我搬到北京"的向量], user_id="cli-user", threshold=0.90
            库里旧记忆: "用户住址:上海"(active), similarity=0.93
            输出: {"id": 42, "summary": "用户住址:上海", "similarity": 0.93}
            若最相近的只有 0.71（不同主题）→ 输出 None（不算冲突，照常插入）

        【为什么阈值 0.90 比召回阈值 0.65 高】
            召回答的是"像不像"（宽进），冲突答的是"是不是同一件事"（严判）。
            0.65 会让"我住上海"和"我去上海出差"被判成冲突，误杀正常记忆。
        """
        self._reconnect_if_needed()
        import numpy as np
        emb = np.array(summary_embedding, dtype=np.float32)
        # 相似度 ≥ threshold 用 HAVING 式过滤在 SQL 里做（1 - 距离 >= 阈值），
        # 只取第一条：同主题旧记忆通常也只有一条是 active（前面的已依次取代）
        cond = " AND id <> %s" if exclude_id else ""
        params = [emb, user_id, emb, 1 - float(threshold)]
        if exclude_id:
            params.append(exclude_id)
        self._cur.execute(
            f"""SELECT id, llm_summary, 1 - (summary_embedding <=> %s) AS similarity
               FROM chat_history
               WHERE user_id = %s
                 AND summary_embedding IS NOT NULL
                 AND COALESCE(memory_status, 'active') = 'active'
                 AND 1 - (summary_embedding <=> %s) >= %s{cond}
               ORDER BY summary_embedding <=> %s
               LIMIT 1""",
            (*params, emb),
        )
        row = self._cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "summary": row[1], "similarity": round(float(row[2]), 4)}

    # 【什么】把一条旧记忆标记为"已被新记忆取代"
    # 【作用】UPDATE 状态列 + 记录取代者的 id；返回是否真的更新了行
    @handle_exception(default=False, log_msg="记忆取代标记异常")
    def supersede_memory(self, memory_id: int, superseded_by: int) -> bool:
        """
        旧记忆置 superseded：memory_status='superseded'，superseded_by=新条 id。

        【输入输出对照】
            输入: memory_id=42, superseded_by=57（新插入那条的主键）
            执行: UPDATE chat_history SET memory_status='superseded',
                       superseded_by=57 WHERE id=42 AND memory_status='active'
            输出: True（rowcount=1）；若 42 已是 superseded（重复调用）→ False
        """
        self._reconnect_if_needed()
        self._cur.execute(
            """UPDATE chat_history
               SET memory_status = 'superseded', superseded_by = %s
               WHERE id = %s AND COALESCE(memory_status, 'active') = 'active'""",
            (superseded_by, memory_id),
        )
        updated = self._cur.rowcount > 0
        self._conn.commit()
        if updated:
            _logger.info(f"记忆冲突处理: 旧记忆 id={memory_id} 已被 id={superseded_by} 取代")
        return updated

    # 【什么】历史数据回填：把老记录的 user_query 向量写进 query_embedding 列
    # 【作用】为将来的「重复提问检测」备数据；当前检索不用它，属于提前准备
    @handle_exception(default=0, log_msg="回填问句向量异常")
    def backfill_query_embeddings(self, user_id: str | None = None, batch_size: int = 64) -> int:
        """
        历史数据回填：把老记录的 user_query 向量化，写进 query_embedding 列。

        【现在要不要跑】当前长期记忆检索用的是 summary_embedding（见 search_memory_by_vector），
            本方法【不影响现有检索功能】，属于"提前把数据备好"。
            什么时候它才真正派上用场：将来做「重复提问检测」时（问句比问句 + 高阈值），
            那时老数据如果没回填，就检测不到改造前的历史提问。
            不想预留这个能力的话，本方法连同 query_embedding 列一起删掉也不影响主流程。

        【幂等性】只处理 query_embedding IS NULL 的行，重复执行安全，已处理过的不会重算。
        【手工调用】启动时不自动执行（避免拖慢启动 / 产生额外 embedding 费用）：
            from modules.memory.pg_manager import PGManager
            PGManager().backfill_query_embeddings()          # 全量
            PGManager().backfill_query_embeddings("U003")    # 只补某个用户

        返回：实际回填的行数。
        """
        self._reconnect_if_needed()
        import numpy as np
        from modules.memory.embedding_service import get_embedding_service

        svc = get_embedding_service()               # 复用向量服务单例，内部自动按 64 条切批
        # 【筛选意图】只捞问句向量为空、且问句非空的行：已回填过的不会重算，故可重复执行
        base_sql = (
            "SELECT id, user_query FROM chat_history "
            "WHERE query_embedding IS NULL AND user_query IS NOT NULL AND user_query <> ''"
        )
        total = 0                                   # 累计回填行数，最终作为返回值
        while True:
            params = []
            sql = base_sql
            if user_id:                             # 传了就只补这一个用户，不传全量补
                sql += " AND user_id = %s"
                params.append(user_id)
            sql += " ORDER BY id LIMIT %s"          # 按主键顺序取一批，保证不重不漏
            params.append(batch_size)               # 每批 batch_size 条（默认 64）

            self._cur.execute(sql, params)
            rows = self._cur.fetchall()
            if not rows:
                break

            ids = [r[0] for r in rows]                      # 本批主键，UPDATE 时按 id 定位
            vectors = svc.embed_batch([r[1] for r in rows]) # 本批问句向量，与 ids 一一对应
            self._cur.executemany(
                "UPDATE chat_history SET query_embedding = %s WHERE id = %s",
                [(np.array(v, dtype=np.float32), _id) for _id, v in zip(ids, vectors)],
            )
            self._conn.commit()
            total += len(ids)
            _logger.info(f"回填问句向量: 本批 {len(ids)} 行，累计 {total} 行")
            if len(rows) < batch_size:              # 取到的不足一批 = 已到末尾，结束循环
                break
        _logger.info(f"回填问句向量完成: 共 {total} 行")
        return total


    # 【什么】记录一次工具调用的全量信息
    # 【作用】排障、性能分析、安全审计的依据；异常只记日志，不阻断对话
    @handle_exception(default=None, log_msg="保存工具日志异常")
    def save_tool_log(
        self, session_id: str, user_id: str,          # 哪个会话、哪个用户触发的
        tool_name: str, tool_type: str,               # 工具名与类型（local / mcp）
        tool_input: dict, tool_output: str,           # 入参会 json.dumps 存 JSONB，出参存文本
        status: str, duration_ms: int,                # 状态默认 'success'；耗时单位毫秒
        task_id: str = "",                            # 跨库对账流水号，留空则取请求上下文
    ) -> None:
        """全量记录工具调用入参、出参、耗时、状态（审计依据）。

        【通俗理解】这是本笔记本之外的第 2 个账本「工具流水账」。
        chat_history 记的是「你和 AI 聊了什么」，
        tool_call_log 记的是「AI 偷偷做过什么动作」。

        【调用例子】用户问「3*7 等于多少」，AI 决定调计算器：
            pg.save_tool_log(
                session_id="abc123", user_id="cli-user",
                tool_name="calculator", tool_type="local",
                tool_input={"expression": "3*7"},   # dict → 会被 json.dumps 转成字符串
                tool_output="21",
                status="success", duration_ms=3,
            )

            tool_call_log 表新增一行：
                id | tool_name  | tool_type | tool_input            | tool_output | status  | duration_ms
                1  | calculator | local     | {"expression":"3*7"}  | 21          | success | 3

        【记这些有什么用？】
            ① 排障：答案算错了，一查就知道是模型传错参数还是工具本身有 bug
            ② 性能优化：duration_ms 能看出哪个工具慢，拖慢整体响应
            ③ 安全审计：谁在什么时间调用了什么，事后可追溯（企业级合规要求）
        """
        from modules.common import context as req_ctx
        task_id = task_id or req_ctx.get_task_id()
        self._reconnect_if_needed()
        self._cur.execute(
            """INSERT INTO tool_call_log
               (session_id, user_id, task_id, tool_name, tool_type, tool_input, tool_output, status, duration_ms)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                session_id, user_id, task_id, tool_name, tool_type,
                json.dumps(tool_input, ensure_ascii=False),  # dict 转 JSON 字符串才能进 JSONB 列
                tool_output, status, duration_ms,
            ),
        )
        self._conn.commit()


    # 【什么】把一轮对话挂进人工审核队列
    # 【作用】高风险内容先不直接采信，插旗等人复核；history_id 让审核员能回看原文
    @handle_exception(default=None, log_msg="加入审核队列异常")
    def add_to_review_queue(
        self, session_id: str, user_id: str,              # 定位会话与用户
        history_id: int, content: str, review_type: str,  # 对话页 id / 触发内容 / 类型（sensitive_query 等）
    ) -> None:
        """将高价值/异常对话加入人工审核队列。

        【通俗理解】AI 自己拿不准或风险较高时，这一页不直接算「已归档」，
        而是插一面小红旗放到 review_queue 清单里，等人来看一眼。

        【调用例子】某轮对话命中了敏感操作，把它挂起来：
            history_id = pg.save_history(...)          # 先存，拿到这一页的页码 42
            pg.add_to_review_queue(
                session_id="abc123", user_id="cli-user",
                history_id=42,                          # 关联回那一页，方便复核时查看原文
                content="帮我查一下张三的工资流水",        # 触发审核的内容
                review_type="sensitive_query",           # 审核类型：sensitive_query / tool_error / low_confidence
            )

            review_queue 表新增一行：
                id | history_id | content              | review_type     | status
                1  | 42         | 帮我查一下张三的工资流水 | sensitive_query | pending  ← 等人处理

        【history_id 为什么要传？】
            它指向 chat_history 的那一行，审核员点开就能看到完整对话上下文，
            而不是只看被挑出来的那一句话。
        """
        self._reconnect_if_needed()
        self._cur.execute(
            """INSERT INTO review_queue (session_id, user_id, history_id, content, review_type)
               VALUES (%s, %s, %s, %s, %s)""",
            (session_id, user_id, history_id, content, review_type),
        )
        self._conn.commit()
        _logger.info(f"加入审核队列: history_id={history_id}, type={review_type}")

    # 【什么】查询待人工审核的记录列表
    # 【作用】供管理页展示待办清单；异常返回 []，页面不至于白屏
    @handle_exception(default=[], log_msg="查询待审核列表异常")
    def get_pending_reviews(self, limit: int = 20) -> list:
        """查询待人工审核的记录列表（也就是所有插着小红旗、还没处理的页）。

        【调用例子】
            pg.get_pending_reviews(limit=20)

            返回：
                [
                    {"id": 1, "session_id": "abc123", "user_id": "cli-user", "history_id": 42,
                     "content": "帮我查一下张三的工资流水", "review_type": "sensitive_query",
                     "created_at": datetime(2026, 9, 14, 12, 5)},
                    {"id": 2, ...},
                ]
            一条都没有 → 返回 []（@handle_exception 里配的 default=[] 兜底异常，
                            正常无数据时 self._cur.fetchall() 本身就是 []）

        【WHERE status = 'pending' 的意义】
            只捞"还没处理"的；审核完的行会被改成 approved / rejected，
            下次再查就不会重复出现，相当于一个待办清单。
            API 层对应路由 GET /reviews/pending，前端管理页直接展示这个列表。
        """
        self._reconnect_if_needed()
        # 【筛选意图】status='pending' 只捞未处理的；审核后变 approved/rejected，不会再出现在清单里
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

    # 【什么】审核处置：pending 改为 approved / rejected
    # 【作用】记录审核人与意见，返回是否真的改到了行；异常返回 False
    @handle_exception(default=False, log_msg="审核处置异常")
    def update_review_status(
        self, review_id: int, status: str, reviewer: str, note: str = ""
    ) -> bool:
        """人工审核处置：把 pending 置为 approved / rejected，并记录审核人与意见。"""
        self._reconnect_if_needed()
        # 【筛选意图】WHERE 带 status='pending'：已处置过的行不会被重复改，审核结论不会被覆盖
        self._cur.execute(
            """UPDATE review_queue
               SET status=%s, reviewer=%s, review_note=%s, reviewed_at=CURRENT_TIMESTAMP
               WHERE id=%s AND status='pending'""",
            (status, reviewer, note, review_id),
        )
        self._conn.commit()
        return self._cur.rowcount > 0           # 0 行受影响 = id 不存在或已处置过，返回 False

    # 【什么】把人工修正后的标准答案写入标准答案库
    # 【作用】审核结论反哺 Few-shot，下次同类问题可直接给标准答案
    @handle_exception(default=None, log_msg="标准答案入库异常")
    def add_standard_answer(
        self, question: str, answer: str, source_review_id: int | None = None   # 溯源：来自哪条审核记录
    ) -> int | None:
        """审核沉淀：把人工修正后的标准答案写入标准答案库，反哺 Few-shot。"""
        self._reconnect_if_needed()
        self._cur.execute(
            """INSERT INTO standard_answers (question, answer, source_review_id)
               VALUES (%s, %s, %s) RETURNING id""",
            (question, answer, source_review_id),
        )
        new_id = self._cur.fetchone()[0]
        self._conn.commit()
        _logger.info(f"标准答案入库: id={new_id} source_review={source_review_id}")
        return new_id

    # 【什么】关键词粗匹配标准答案库
    # 【作用】命中后作为 Few-shot 投喂模型；异常返回 []，走正常回答流程
    @handle_exception(default=[], log_msg="标准答案召回异常")
    def match_standard_answers(self, query: str, limit: int = 2) -> list:
        """关键词匹配标准答案库，命中后作为 Few-shot 投喂模型（人工审核沉淀反哺）。

        PG 的 simple 全文配置对中文不分词，故这里用 ILIKE 做关键词粗匹配。
        """
        import re
        kws = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", query)   # 中文取 2 字以上，英数取 2 字符以上
        kws.sort(key=len, reverse=True)                                   # 长词优先：越长的词区分度越高
        kws = kws[:3]                                                     # 最多取 3 个关键词，控制 SQL 条件数
        if not kws:
            return []                                                     # 一个关键词都抽不出就直接放弃匹配
        self._reconnect_if_needed()
        where = " OR ".join(["question ILIKE %s"] * len(kws))   # 任一关键词命中即算匹配，ILIKE 忽略大小写
        self._cur.execute(
            f"SELECT id, question, answer FROM standard_answers WHERE {where} LIMIT %s",
            (*[f"%{k}%" for k in kws], limit),
        )
        return [
            {"id": r[0], "question": r[1], "answer": r[2]}
            for r in self._cur.fetchall()
        ]

    # 【什么】统计指定用户未归档的会话数
    # 【作用】离职对账用，只读不写；异常返回 {}，对账方按 0 处理
    @handle_exception(default={}, log_msg="活跃会话统计异常")
    def count_active_sessions(self, user_codes: list) -> dict:
        """统计指定用户未归档的会话数（离职对账用，只读不写）。

        输入 ["U001","U005"] → {"U001": 3, "U005": 0}
        """
        if not user_codes:
            return {}
        self._reconnect_if_needed()
        # 【筛选意图】ANY(%s) 一次匹配整批用户；archived = 0 只数未归档（还活着）的会话
        self._cur.execute(
            """SELECT user_id, COUNT(*) FROM chat_session
               WHERE user_id = ANY(%s) AND archived = 0 GROUP BY user_id""",
            (list(user_codes),),
        )
        return {r[0]: r[1] for r in self._cur.fetchall()}

    # 【什么】把指定用户的会话标记为已归档（离职处置第一步）
    # 【作用】只打标不删数据；异常返回 0，调用方据此知道没标成
    @handle_exception(default=0, log_msg="会话归档异常")
    def archive_user_sessions(self, user_codes: list) -> int:
        """把指定用户的会话标记为归档（离职处置第一步，只打标不删数据）。

        输入 user_codes=["U001","U005"] → 返回被标记的会话行数（已归档的不重复计）。
        """
        if not user_codes:
            return 0
        self._reconnect_if_needed()
        # 【筛选意图】archived = 0 保证已归档的不重复计；updated_at 同步刷新，作为保留期起算点
        self._cur.execute(
            """UPDATE chat_session
               SET archived = 1, updated_at = CURRENT_TIMESTAMP
               WHERE user_id = ANY(%s) AND archived = 0""",
            (list(user_codes),),
        )
        n = self._cur.rowcount
        self._conn.commit()
        if n:
            _logger.info(f"会话归档: users={len(user_codes)} sessions={n}")
        return n

    # 【什么】列出已归档用户及其会话数
    # 【作用】保留期清理前的对账视图；异常返回 []，不阻断清理流程
    @handle_exception(default=[], log_msg="归档用户查询异常")
    def list_archived_users(self, limit: int = 100) -> list:
        """列出已被归档的用户及其会话数（保留期清理前的对账视图）。"""
        self._reconnect_if_needed()
        # 【筛选意图】archived = 1 只统计已归档用户；按最后更新时间倒序，最近处置的排前面
        self._cur.execute(
            """SELECT user_id, COUNT(*) AS sessions, MAX(updated_at) AS archived_at
               FROM chat_session WHERE archived = 1
               GROUP BY user_id ORDER BY archived_at DESC LIMIT %s""",
            (limit,),
        )
        rows = self._cur.fetchall()
        return [{"user_id": r[0], "sessions": r[1], "archived_at": str(r[2])} for r in rows]

    # 【什么】删除已归档且超保留期的会话数据（离职处置第二步）
    # 【作用】真正落删；先删工具流水再删会话，异常返回 None 由调用方判断
    @handle_exception(default=None, log_msg="过期数据清理异常")
    def purge_archived_data(self, retention_days: int) -> dict:
        """删除归档且超过保留期的会话数据（离职处置第二步，真正落删）。

        chat_history 外键带 ON DELETE CASCADE，删 session 会连带删除对话页；
        tool_call_log 无外键，需单独删。review_queue 属治理证据，保留不删。
        """
        self._reconnect_if_needed()
        # 【筛选意图】只删"已归档 且 更新时间早于 now - 保留天数"的会话；tool_call_log 无外键，先单独删
        self._cur.execute(
            "DELETE FROM tool_call_log WHERE session_id IN "
            "(SELECT session_id FROM chat_session WHERE archived = 1 "
            " AND updated_at < CURRENT_TIMESTAMP - make_interval(days => %s))",
            (int(retention_days),),     # make_interval 的 days 参数，单位天
        )
        tool_rows = self._cur.rowcount                 # 本次删掉的工具流水行数
        # 删会话：chat_history 靠外键 ON DELETE CASCADE 连带删除；review_queue 作为治理证据保留
        self._cur.execute(
            "DELETE FROM chat_session WHERE archived = 1 "
            " AND updated_at < CURRENT_TIMESTAMP - make_interval(days => %s)",
            (int(retention_days),),
        )
        session_rows = self._cur.rowcount                 # 本次删掉的会话行数（不含级联的对话页）
        self._conn.commit()
        if session_rows:
            _logger.info(f"过期数据清理: sessions={session_rows} tool_logs={tool_rows}")
        return {"retention_days": retention_days, "sessions_deleted": session_rows,
                "tool_logs_deleted": tool_rows}           # 两个计数均取自 rowcount

    # 【什么】取最近 N 小时写入的 task_id（两张表的并集）
    # 【作用】跨库对账用：拿它与 MySQL 审计流水比对找单边缺失；异常返回 []
    @handle_exception(default=[], log_msg="task_id 列表查询异常")
    def list_task_ids_since(self, hours: int = 24, limit: int = 5000) -> list:
        """取最近 N 小时写入的 task_id（chat_history ∪ tool_call_log），供跨库对账。

        输入 hours=24 → ["T-20260920-3f9c1a2b", "T-20260920-91ad44e0"]
        空 task_id 的行不参与（对账只比有号的，老数据没有号是已知的债）。
        """
        self._reconnect_if_needed()
        # 【筛选意图】task_id <> '' 排掉老数据（无号是已知债务）；UNION 自动去重，两表同一号只算一次
        self._cur.execute(
            "SELECT task_id FROM chat_history "
            "WHERE task_id <> '' AND created_at >= CURRENT_TIMESTAMP - make_interval(hours => %s) "
            "UNION "
            "SELECT task_id FROM tool_call_log "
            "WHERE task_id <> '' AND created_at >= CURRENT_TIMESTAMP - make_interval(hours => %s) "
            "LIMIT %s",
            (int(hours), int(hours), int(limit)),
        )
        return [r[0] for r in self._cur.fetchall() if r[0]]

    @handle_exception(default=[], log_msg="待归档会话查询异常")
    def list_archived_session_ids(self, retention_days: int, limit: int = 500) -> list:
        """列出已归档且超保留期的 session_id（冷热分离/清理前的目标清单）。"""
        self._reconnect_if_needed()
        self._cur.execute(
            "SELECT session_id FROM chat_session WHERE archived = 1 "
            "AND updated_at < CURRENT_TIMESTAMP - make_interval(days => %s) LIMIT %s",
            (int(retention_days), int(limit)),
        )
        return [r[0] for r in self._cur.fetchall()]

    @handle_exception(default=[], log_msg="过期会话查询异常")
    def list_expired_session_ids(self, retention_days: int, limit: int = 500) -> list:
        """列出超过热数据保留期的 session_id（冷热分离用，不看 archived 标记）。

        输入 retention_days=90 → ["s-1", "s-2"]（90 天内有更新的会话不在此列）
        """
        self._reconnect_if_needed()
        self._cur.execute(
            "SELECT session_id FROM chat_session "
            "WHERE updated_at < CURRENT_TIMESTAMP - make_interval(days => %s) LIMIT %s",
            (int(retention_days), int(limit)),
        )
        return [r[0] for r in self._cur.fetchall()]

    @handle_exception(default=[], log_msg="会话历史导出异常")
    def fetch_history_by_session(self, session_id: str) -> list:
        """导出一个会话的全部对话页（归档用，不删数据）。"""
        self._reconnect_if_needed()
        cols = ["id", "session_id", "user_id", "task_id", "user_query", "llm_full_answer",
                "llm_summary", "tool_call_log", "rag_sources", "created_at"]
        self._cur.execute(
            f"SELECT {', '.join(cols)} FROM chat_history WHERE session_id = %s ORDER BY id",
            (session_id,),
        )
        return [dict(zip(cols, r)) for r in self._cur.fetchall()]

    @handle_exception(default=[], log_msg="工具日志导出异常")
    def fetch_tool_logs_by_session(self, session_id: str) -> list:
        """导出一个会话的全部工具流水（归档用）。"""
        self._reconnect_if_needed()
        cols = ["id", "session_id", "user_id", "task_id", "tool_name", "tool_type",
                "tool_input", "tool_output", "status", "duration_ms", "created_at"]
        self._cur.execute(
            f"SELECT {', '.join(cols)} FROM tool_call_log WHERE session_id = %s ORDER BY id",
            (session_id,),
        )
        return [dict(zip(cols, r)) for r in self._cur.fetchall()]

    @handle_exception(default=0, log_msg="会话删除异常")
    def delete_sessions(self, session_ids: list) -> int:
        """按 session_id 批量删除（工具流水 + 会话，对话页靠外键级联删除）。"""
        if not session_ids:
            return 0
        self._reconnect_if_needed()
        self._cur.execute(
            "DELETE FROM tool_call_log WHERE session_id = ANY(%s)", (list(session_ids),)
        )
        self._cur.execute(
            "DELETE FROM chat_session WHERE session_id = ANY(%s)", (list(session_ids),)
        )
        n = self._cur.rowcount
        self._conn.commit()
        return n

    def close(self):
        """关闭连接（合上笔记本、锁门）。

        【调用例子】
            pg = PGManager()
            ...各种操作...
            pg.close()          # main.py 的 cmd_chat 在退出 while 循环后就会调

        【为什么要 hasattr 检查？】
            如果 _connect 一开始就失败（比如 PG 没启动），对象上根本没有 _cur / _conn，
            直接 self._cur.close() 会抛 AttributeError，把真正的报错（PG 连接失败）盖掉。
            加这层判断，保证「连没连上都敢调 close()」。
            类比 JS：if (this.db) this.db.end()
        """
        if hasattr(self, "_cur"):
            self._cur.close()
        if hasattr(self, "_conn"):
            self._conn.close()
        _logger.info("PG连接已关闭")
