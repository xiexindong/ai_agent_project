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
        dim = s.embedding_dimension
        ddl = [
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
            """
            ALTER TABLE chat_session ADD COLUMN IF NOT EXISTS archived SMALLINT DEFAULT 0;
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_session_archived
            ON chat_session (archived, updated_at DESC);
            """,
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
            f"""
            ALTER TABLE chat_history
            ADD COLUMN IF NOT EXISTS query_embedding vector({dim});
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_history_query_emb
            ON chat_history USING ivfflat (query_embedding vector_cosine_ops)
            WITH (lists = 100);
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_history_summary_emb
            ON chat_history USING ivfflat (summary_embedding vector_cosine_ops)
            WITH (lists = 100);
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_history_session_user
            ON chat_history (session_id, user_id, created_at DESC);
            """,
            """
            ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS task_id VARCHAR(64) DEFAULT '';
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_history_task ON chat_history (task_id);
            """,
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
            """
            ALTER TABLE tool_call_log ADD COLUMN IF NOT EXISTS task_id VARCHAR(64) DEFAULT '';
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_tool_log_task ON tool_call_log (task_id);
            """,
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
            """
            CREATE TABLE IF NOT EXISTS standard_answers (
                id               BIGSERIAL PRIMARY KEY,
                question         TEXT NOT NULL,
                answer           TEXT NOT NULL,
                source_review_id BIGINT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_std_answer_question
            ON standard_answers USING gin (to_tsvector('simple', question));
            """,
        ]
        for sql in ddl:
            self._cur.execute(sql)
        self._conn.commit()
        _logger.info("PG表结构初始化完成（chat_session, chat_history, tool_call_log, review_queue）")

    def _reconnect_if_needed(self):
        """连接断开时自动重连（心跳探活 + 失败自愈）。"""
        try:
            self._cur.execute("SELECT 1")
        except Exception:
            _logger.warning("PG连接断开，尝试重连...")
            self._connected = False
            self._connect()


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
            (session_id, user_id, title),
        )
        self._conn.commit()
        _logger.info(f"会话创建: session={session_id}, user={user_id}")

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
            (json.dumps(agent_state, ensure_ascii=False), session_id),
        )
        self._conn.commit()

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
        row = self._cur.fetchone()

        return row[0] if row and row[0] else {}


    @handle_exception(default=None, log_msg="保存对话记忆异常", reraise=True)
    def save_history(
        self,
        session_id: str,
        user_id: str,
        user_query: str,
        llm_full_answer: str,
        llm_summary: str,
        summary_embedding: list,
        tool_call_log: list | None = None,
        rag_sources: list | None = None,
        query_embedding: list | None = None,
        task_id: str = "",
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
        emb = np.array(summary_embedding, dtype=np.float32) if summary_embedding else None
        q_emb = np.array(query_embedding, dtype=np.float32) if query_embedding else None
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
                session_id, user_id, task_id, user_query,
                llm_full_answer, llm_summary, emb, q_emb,
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
        rows.reverse()
        return [
            {"query": r[0], "answer": r[1], "summary": r[2], "time": r[3]}
            for r in rows
        ]

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
                "query": r[0], "summary": r[1], "answer": r[2],
                "similarity": round(float(r[3]), 4), "time": r[4],
            }
            for r in rows
        ]

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

        svc = get_embedding_service()
        base_sql = (
            "SELECT id, user_query FROM chat_history "
            "WHERE query_embedding IS NULL AND user_query IS NOT NULL AND user_query <> ''"
        )
        total = 0
        while True:
            params = []
            sql = base_sql
            if user_id:
                sql += " AND user_id = %s"
                params.append(user_id)
            sql += " ORDER BY id LIMIT %s"
            params.append(batch_size)

            self._cur.execute(sql, params)
            rows = self._cur.fetchall()
            if not rows:
                break

            ids = [r[0] for r in rows]
            vectors = svc.embed_batch([r[1] for r in rows])
            self._cur.executemany(
                "UPDATE chat_history SET query_embedding = %s WHERE id = %s",
                [(np.array(v, dtype=np.float32), _id) for _id, v in zip(ids, vectors)],
            )
            self._conn.commit()
            total += len(ids)
            _logger.info(f"回填问句向量: 本批 {len(ids)} 行，累计 {total} 行")
            if len(rows) < batch_size:
                break
        _logger.info(f"回填问句向量完成: 共 {total} 行")
        return total


    @handle_exception(default=None, log_msg="保存工具日志异常")
    def save_tool_log(
        self, session_id: str, user_id: str,
        tool_name: str, tool_type: str,
        tool_input: dict, tool_output: str,
        status: str, duration_ms: int,
        task_id: str = "",
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
                json.dumps(tool_input, ensure_ascii=False),
                tool_output, status, duration_ms,
            ),
        )
        self._conn.commit()


    @handle_exception(default=None, log_msg="加入审核队列异常")
    def add_to_review_queue(
        self, session_id: str, user_id: str,
        history_id: int, content: str, review_type: str,
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

    @handle_exception(default=False, log_msg="审核处置异常")
    def update_review_status(
        self, review_id: int, status: str, reviewer: str, note: str = ""
    ) -> bool:
        """人工审核处置：把 pending 置为 approved / rejected，并记录审核人与意见。"""
        self._reconnect_if_needed()
        self._cur.execute(
            """UPDATE review_queue
               SET status=%s, reviewer=%s, review_note=%s, reviewed_at=CURRENT_TIMESTAMP
               WHERE id=%s AND status='pending'""",
            (status, reviewer, note, review_id),
        )
        self._conn.commit()
        return self._cur.rowcount > 0

    @handle_exception(default=None, log_msg="标准答案入库异常")
    def add_standard_answer(
        self, question: str, answer: str, source_review_id: int | None = None
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

    @handle_exception(default=[], log_msg="标准答案召回异常")
    def match_standard_answers(self, query: str, limit: int = 2) -> list:
        """关键词匹配标准答案库，命中后作为 Few-shot 投喂模型（人工审核沉淀反哺）。

        PG 的 simple 全文配置对中文不分词，故这里用 ILIKE 做关键词粗匹配。
        """
        import re
        kws = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", query)
        kws.sort(key=len, reverse=True)
        kws = kws[:3]
        if not kws:
            return []
        self._reconnect_if_needed()
        where = " OR ".join(["question ILIKE %s"] * len(kws))
        self._cur.execute(
            f"SELECT id, question, answer FROM standard_answers WHERE {where} LIMIT %s",
            (*[f"%{k}%" for k in kws], limit),
        )
        return [
            {"id": r[0], "question": r[1], "answer": r[2]}
            for r in self._cur.fetchall()
        ]

    @handle_exception(default={}, log_msg="活跃会话统计异常")
    def count_active_sessions(self, user_codes: list) -> dict:
        """统计指定用户未归档的会话数（离职对账用，只读不写）。

        输入 ["U001","U005"] → {"U001": 3, "U005": 0}
        """
        if not user_codes:
            return {}
        self._reconnect_if_needed()
        self._cur.execute(
            """SELECT user_id, COUNT(*) FROM chat_session
               WHERE user_id = ANY(%s) AND archived = 0 GROUP BY user_id""",
            (list(user_codes),),
        )
        return {r[0]: r[1] for r in self._cur.fetchall()}

    @handle_exception(default=0, log_msg="会话归档异常")
    def archive_user_sessions(self, user_codes: list) -> int:
        """把指定用户的会话标记为归档（离职处置第一步，只打标不删数据）。

        输入 user_codes=["U001","U005"] → 返回被标记的会话行数（已归档的不重复计）。
        """
        if not user_codes:
            return 0
        self._reconnect_if_needed()
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

    @handle_exception(default=[], log_msg="归档用户查询异常")
    def list_archived_users(self, limit: int = 100) -> list:
        """列出已被归档的用户及其会话数（保留期清理前的对账视图）。"""
        self._reconnect_if_needed()
        self._cur.execute(
            """SELECT user_id, COUNT(*) AS sessions, MAX(updated_at) AS archived_at
               FROM chat_session WHERE archived = 1
               GROUP BY user_id ORDER BY archived_at DESC LIMIT %s""",
            (limit,),
        )
        rows = self._cur.fetchall()
        return [{"user_id": r[0], "sessions": r[1], "archived_at": str(r[2])} for r in rows]

    @handle_exception(default=None, log_msg="过期数据清理异常")
    def purge_archived_data(self, retention_days: int) -> dict:
        """删除归档且超过保留期的会话数据（离职处置第二步，真正落删）。

        chat_history 外键带 ON DELETE CASCADE，删 session 会连带删除对话页；
        tool_call_log 无外键，需单独删。review_queue 属治理证据，保留不删。
        """
        self._reconnect_if_needed()
        self._cur.execute(
            "DELETE FROM tool_call_log WHERE session_id IN "
            "(SELECT session_id FROM chat_session WHERE archived = 1 "
            " AND updated_at < CURRENT_TIMESTAMP - make_interval(days => %s))",
            (int(retention_days),),
        )
        tool_rows = self._cur.rowcount
        self._cur.execute(
            "DELETE FROM chat_session WHERE archived = 1 "
            " AND updated_at < CURRENT_TIMESTAMP - make_interval(days => %s)",
            (int(retention_days),),
        )
        session_rows = self._cur.rowcount
        self._conn.commit()
        if session_rows:
            _logger.info(f"过期数据清理: sessions={session_rows} tool_logs={tool_rows}")
        return {"retention_days": retention_days, "sessions_deleted": session_rows,
                "tool_logs_deleted": tool_rows}

    @handle_exception(default=[], log_msg="task_id 列表查询异常")
    def list_task_ids_since(self, hours: int = 24, limit: int = 5000) -> list:
        """取最近 N 小时写入的 task_id（chat_history ∪ tool_call_log），供跨库对账。

        输入 hours=24 → ["T-20260920-3f9c1a2b", "T-20260920-91ad44e0"]
        空 task_id 的行不参与（对账只比有号的，老数据没有号是已知的债）。
        """
        self._reconnect_if_needed()
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
