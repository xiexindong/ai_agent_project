"""
工具元数据的 pgvector 存取
==========================
Tool-RAG 的存储层：一张表装下所有技能的「名 + 描述 + 参数名 + 向量」，
运行时按语义召回，而不是把全部工具塞给模型。

【表结构】见 sql/init_tool_rag.sql（本模块的 DDL 与之保持一致）：
    name / description / domain_tag / source_type / server_name /
    parameters_json / enabled / embedding vector(1024) / updated_at

【三个动作】
    ensure_schema()   建表 + 建索引（幂等，可重复执行）
    upsert_tools()    工具向量入库（同名覆盖，描述变了只更新那一行）
    search()          按 query 向量召回候选（支持按业务域过滤 + 黑名单排除）

【连接不可用时的行为】
    PG 连不上时 _ok=False，所有方法返回空结果而不是抛异常 ——
    存储层不可用只意味着「召不回候选」，不该让整条请求流水线挂掉。
"""
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from modules.common.logger import get_logger
from config import get_settings

_logger = get_logger("pipeline.tool_store")

# 表名白名单校验：表名要拼进 SQL，不能接受任意字符串
_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# domain_tag 为这些值时不做过滤（表示"全域检索"）
_NO_DOMAIN_FILTER = ("", "general", "all", "none")

# 关键词降级时的相似度上限：关键词匹配不如向量准，分数压低避免与向量结果混淆
_KEYWORD_SCORE_BASE = 0.60


# 【什么】浮点向量 → pgvector 字符串字面量
# 【作用】list 直接传参会变成 ARRAY[...] 而报运算符不存在，拼成 "[...]" 才稳
def to_vector_literal(vector: Sequence[float]) -> str:
    """把 Python 浮点列表转成 pgvector 能直接吃的字符串字面量。

    【为什么不能把 list 直接当参数传】
        psycopg2 遇到 list 会渲染成 SQL 数组字面量 ARRAY[0.1,0.2,...]，
        而 pgvector 只给「vector <=> vector」定义了运算符，没有「vector <=> real[]」。
        直接传 list 会报：
            operator does not exist: vector <=> numeric[]
        所以这里自己拼成 '[0.1,0.2,...]' 字符串，SQL 侧再补 ::vector 显式转换。

    【输入输出对照】
        输入 [0.0059608887, -0.034335047]
            → 输出 "[0.0059608887,-0.034335047]"（字符串，可直接写进 SQL）
    """
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


# 【什么】Tool-RAG 存储层：一张表装下工具元数据与向量
# 【作用】对外只暴露三个动作：建表、入库、检索；PG 不可用时返回空而不抛异常
class ToolVectorStore:
    """工具元数据的向量存取（PostgreSQL + pgvector）。"""

    # 【什么】构造函数：确定表名与维度后立刻建连
    # 【作用】table/dimension 留空即取配置项，运行期不再重复读配置
    def __init__(self, table: str = "", dimension: int = 0):   # table/dimension 为空时取配置值
        s = get_settings()
        self._table = self._safe_table(table or s.tool_vector_table)  # 表名，非法时回退 tool_catalog
        self._dim = dimension or int(s.embedding_dimension)  # 向量维度，如 1024
        self._conn = None   # psycopg2 连接对象
        self._cur = None    # 数据库游标
        self._ok = False    # 存储层是否可用；False 时所有方法返回空结果
        self._connect()

    # 【什么】表名安全校验（静态方法）
    # 【作用】表名是唯一必须拼进 SQL 的标识符，白名单卡死，不合法就回退默认表名
    # ------------------------------------------------------------------
    # 【表名为什么要校验】
    #   表名是唯一必须拼进 SQL 字符串的标识符（psycopg2 的参数占位符只能代替换值，
    #   不能代替表名/列名）。不做校验的话，配置里写个奇怪的表名就会拼出非法 SQL。
    #   这里用白名单正则卡死：只允许字母数字下划线，不合法就退回默认表名。
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_table(table: str) -> str:
        if _TABLE_NAME_RE.match(table or ""):
            return table
        _logger.warning(f"非法表名 {table!r}，回退为 tool_catalog")
        return "tool_catalog"

    # 【什么】存储层可用性标识
    # 【作用】召回层先看它再决定走向量还是关键词/本地兜底
    @property
    def available(self) -> bool:
        """存储层是否可用（PG 连上了才为 True）。"""
        return self._ok

    # 【什么】当前生效的表名
    # 【作用】建表与检索都用它，排障时确认操作的是哪张表
    @property
    def table(self) -> str:
        return self._table

    # 【什么】建立 PG 连接并注册 pgvector 类型
    # 【作用】连接失败不抛异常，只置 _ok=False，把"连不上"变成可降级状态
    def _connect(self) -> None:
        """连接 PG 并注册 pgvector 类型适配器。

        【三步缺一不可，顺序也不能换】
            ① CREATE EXTENSION vector  —— 让数据库认识 vector 类型
            ② commit                   —— 让扩展真正生效（否则下面查类型 OID 会查不到）
            ③ register_vector(conn)    —— 让 psycopg2 会收发 vector 类型

            少了 ①：建表时报 type "vector" does not exist
            少了 ②：register_vector 查 pg_type 找不到类型，注册失败
            少了 ③：插入时报 can't adapt type 'list'，向量根本写不进去
        """
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector

            s = get_settings()
            self._conn = psycopg2.connect(
                host=s.pg_host, port=s.pg_port, user=s.pg_user,
                password=s.pg_password, dbname=s.pg_database,
            )
            self._conn.autocommit = False
            self._cur = self._conn.cursor()
            self._cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self._conn.commit()
            register_vector(self._conn)
            self._ok = True
            _logger.info(f"Tool-RAG 存储已连接: {s.pg_host}:{s.pg_port}/{s.pg_database} 表={self._table}")
        except Exception as e:
            self._ok = False
            _logger.warning(f"Tool-RAG 存储不可用（将走关键词降级）: {e}")

    # 【什么】探活：执行一次 SELECT 1
    # 【作用】PG 重启或网络闪断后能自愈一次，避免整批请求都走降级
    def _is_alive(self) -> bool:
        """检查连接是否还活着（PG 重启/网络闪断后能自愈一次）。"""
        if not self._ok or self._conn is None:
            return False
        try:
            self._cur.execute("SELECT 1")
            return True
        except Exception:
            try:
                self._conn.rollback()
                self._connect()
            except Exception:
                pass
            return self._ok

    # 【什么】建表 + 建索引（幂等，可重复执行）
    # 【作用】IVFFlat 向量索引按余弦距离建，lists=100；域索引加速 domain_tag 过滤
    # ------------------------------------------------------------------
    # 【建表】DDL 与 sql/init_tool_rag.sql 保持一致
    #   幂等：IF NOT EXISTS，重复执行无副作用
    # ------------------------------------------------------------------
    def ensure_schema(self) -> bool:
        """建表 + 建索引。返回是否成功。"""
        if not self._is_alive():
            return False
        try:
            ddl = f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id              BIGSERIAL PRIMARY KEY,
                    name            VARCHAR(128) NOT NULL UNIQUE,
                    description     TEXT         NOT NULL DEFAULT '',
                    domain_tag      VARCHAR(64)  NOT NULL DEFAULT 'general',
                    source_type     VARCHAR(16)  NOT NULL DEFAULT 'local',
                    server_name     VARCHAR(64)  NOT NULL DEFAULT '',
                    parameters_json JSONB        NOT NULL DEFAULT '{{}}'::jsonb,
                    enabled         BOOLEAN      NOT NULL DEFAULT TRUE,
                    embedding       vector({self._dim}),
                    updated_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """
            self._cur.execute(ddl)
            self._cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._table}_embedding "
                f"ON {self._table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
            )
            self._cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._table}_domain ON {self._table} (domain_tag);"
            )
            self._conn.commit()
            _logger.info(f"工具元数据表已就绪: {self._table} (dim={self._dim})")
            return True
        except Exception as e:
            try:
                self._conn.rollback()
            except Exception:
                pass
            _logger.warning(f"建表失败: {e}")
            return False

    # 【什么】拼出用于 embedding 编码的文本
    # 【作用】只放"名 + 描述 + 参数名"，避免完整 schema 里的类型词稀释语义
    @staticmethod
    def routing_text(name: str, description: str, parameters: Any) -> str:
        """拼出用于编码的文本。

        【为什么只放"名 + 描述 + 参数名"，不放完整 schema】
            召回阶段要回答的是"该不该用这个工具"，不是"怎么用"。
            把完整 JSON schema 塞进去，参数类型（"string"/"integer"）这类词会稀释语义，
            让"查询天气"和"创建工单"因为都有 string 参数而变近。

        【输入输出对照】
            输入 name="city_weather", description="查询城市天气", parameters={"city":{...},"date":{...}}
                → 输出 "city_weather\n查询城市天气\n参数: city, date"
        """
        if isinstance(parameters, dict):
            names = [str(k) for k in parameters.get("properties", parameters).keys()]
        elif isinstance(parameters, (list, tuple)):
            names = [str(k) for k in parameters]
        else:
            names = []
        tail = f"\n参数: {', '.join(names)}" if names else ""
        return f"{name}\n{description}{tail}"

    # 【什么】工具向量批量入库（同名覆盖）
    # 【作用】先一次批量编码再逐行 upsert，返回实际写入行数，失败返回 0
    def upsert_tools(self, tools: Sequence[Dict[str, Any]], embed_fn=None) -> int:  # embed_fn=None 时用智谱 embedding
        """工具向量入库（同名覆盖）。

        Args:
            tools: 每个元素至少含 name / description，可选 domain_tag / source_type /
                   server_name / parameters
            embed_fn: 编码函数 (List[str]) -> List[List[float]]；为 None 时用智谱 embedding 服务

        Returns:
            实际写入的行数（失败返回 0）

        【输入输出对照】
            输入 [{"name":"city_weather","description":"查询城市天气",
                   "domain_tag":"data","source_type":"mcp_remote","server_name":"data-mcp"}]
                → 返回 1（表里 name=city_weather 这一行被写入，embedding 为 1024 维向量）
            输入 同一个工具再 upsert 一次（描述改了）
                → 返回 1（更新同一行，不会新增第二行）
        """
        if not tools:
            return 0
        if not self._is_alive() or not self.ensure_schema():
            return 0

        # 归一化字段
        rows = []
        for t in tools:
            name = str(t.get("name", "")).strip()
            if not name:
                continue
            rows.append({
                "name": name,
                "description": str(t.get("description", "") or ""),
                "domain_tag": str(t.get("domain_tag", "") or "general"),
                "source_type": str(t.get("source_type", "") or "local"),
                "server_name": str(t.get("server_name", "") or ""),
                "parameters": t.get("parameters") or t.get("parameters_json") or {},
            })
        if not rows:
            return 0

        # ------------------------------------------------------------------
        # 【为什么要一次批量编码】逐条调 embedding 接口，N 个工具就是 N 次网络往返；
        #   embed_batch 内部按 64 条一批切分，30 个工具只要 1 次调用。
        # ------------------------------------------------------------------
        texts = [self.routing_text(r["name"], r["description"], r["parameters"]) for r in rows]
        try:
            vectors = embed_fn(texts) if embed_fn else get_embedding_service().embed_batch(texts)
        except Exception as e:
            _logger.warning(f"工具向量编码失败，未写入: {e}")
            return 0

        sql = f"""
            INSERT INTO {self._table}
                (name, description, domain_tag, source_type, server_name, parameters_json, enabled, embedding, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE, %s::vector, CURRENT_TIMESTAMP)
            ON CONFLICT (name) DO UPDATE SET
                description     = EXCLUDED.description,
                domain_tag      = EXCLUDED.domain_tag,
                source_type     = EXCLUDED.source_type,
                server_name     = EXCLUDED.server_name,
                parameters_json = EXCLUDED.parameters_json,
                embedding       = EXCLUDED.embedding,
                updated_at      = CURRENT_TIMESTAMP
        """

        written = 0
        try:
            for r, vec in zip(rows, vectors):
                # 维度不一致的行直接跳过：PG 会报 expected N dimensions，
                # 一条坏数据会让整批 insert 一起回滚
                if not vec or len(vec) != self._dim:
                    _logger.warning(f"向量维度不符，跳过 {r['name']}: {len(vec) if vec else 0} != {self._dim}")
                    continue
                self._cur.execute(sql, (
                    r["name"], r["description"], r["domain_tag"], r["source_type"],
                    r["server_name"], json.dumps(r["parameters"], ensure_ascii=False),
                    to_vector_literal(vec),   # 字符串 + SQL 侧 ::vector，见函数注释
                ))
                written += 1
            self._conn.commit()
            _logger.info(f"工具向量入库: {written}/{len(rows)} 条 → {self._table}")
            return written
        except Exception as e:
            try:
                self._conn.rollback()
            except Exception:
                pass
            _logger.warning(f"工具向量入库失败: {e}")
            return 0

    # 【什么】按语义召回候选工具（向量检索 + 双重阈值过滤）
    # 【作用】只用 SQL 取回 top_k，再用绝对/相对两道阈值切掉尾部噪声
    def search(
        self,
        query_embedding: Sequence[float],      # 用户提问的向量
        domain_tag: str = "",                  # 业务域过滤；空/general/all/none 表示不过滤
        exclude: Optional[List[str]] = None,   # 黑名单工具名，本轮不许被召回
        top_k: int = 3,                        # 召回数量上限
        min_score: float = 0.25,               # 绝对相似度下限
        relative_ratio: float = 0.60,          # 相对下限系数：需 >= 本轮最高分 * 该系数
    ) -> List[Dict[str, Any]]:
        """按语义召回候选工具。

        Args:
            query_embedding: 用户提问的向量
            domain_tag: 业务域过滤；空 / general / all 表示不过滤
            exclude: 黑名单工具名（本轮不许被召回）
            top_k: 召回数量上限
            min_score: 绝对相似度下限
            relative_ratio: 相对下限系数（候选分数需 ≥ 本轮最高分 * 该系数）

        Returns:
            [{"name","description","domain_tag","source_type","server_name","score"}]

        【过滤为什么写在 SQL 里而不是捞回来再过滤】
            域过滤和黑名单过滤都是"先缩候选集"，放在 SQL 里能少传数据、少占索引扫描；
            只有 min_score 这种要和阈值比较的放在 Python 里做（顺便记录被过滤掉多少）。

        【SQL 里的两个运算符】
            embedding <=> %s        余弦距离（0=完全相同，2=完全相反）
            1 - (...)               转成"相似度"（1=最像，0=最不像）
            ORDER BY 距离 ASC  = ORDER BY 相似度 DESC，取前 top_k 个

        【两道阈值怎么用（实测：智谱 embedding-2，1024 维）】
            绝对：score >= min_score(0.25)
            相对：score >= 本轮最高分 * relative_ratio(0.60)
            两条都要满足。实测数据（12 个工具）：
                "我的报销单到哪一步了" → 最高 0.5517，相对阈值 0.331
                    → 保留 query_reimbursement(0.5517)、submit_invoice(0.3651)，砍掉 0.3134
                "上海明天天气怎么样"   → 最高 0.2871，相对阈值 0.172，绝对阈值生效
                    → 保留 city_weather(0.2871)，砍掉 server_health(0.1412)

        【为什么要相对阈值】
            embedding-2 的相似度整体偏低（正确的中文匹配常在 0.30~0.60），
            纯绝对阈值换个模型就要重调；加一条"跟着第一名走"的相对阈值后，
            只要第一名明显领先，尾部噪声就会被自动切掉。

        【输入输出对照】
            query_embedding=<"报销单进度"的向量>, domain_tag="finance", exclude=["submit_invoice"], top_k=3
                → [{"name":"query_reimbursement","score":0.5517,...},
                   {"name":"logistics_track","score":0.3134,...}]
                （submit_invoice 在黑名单里，即使相似度 0.3651 也不出现）
        """
        if not self._is_alive() or not query_embedding:
            return []

        domain = domain_tag if domain_tag and str(domain_tag).lower() not in _NO_DOMAIN_FILTER else None
        exclude = [str(x) for x in (exclude or []) if str(x).strip()]

        # ------------------------------------------------------------------
        # 【两处 ::vector 不能省】
        #   %s 直接吃 Python list 会被 psycopg2 渲染成 ARRAY[...]，
        #   PG 侧没有 vector <=> numeric[] 运算符，报错：
        #       operator does not exist: vector <=> numeric[]
        #   传字符串字面量 + ::vector 才是稳的写法（不依赖 pgvector 的隐式 cast）。
        # ------------------------------------------------------------------
        sql = f"""
            SELECT name, description, domain_tag, source_type, server_name,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {self._table}
            WHERE enabled = TRUE
              AND embedding IS NOT NULL
              AND (%s IS NULL OR domain_tag = %s)
              AND NOT (name = ANY(%s))
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        vec_literal = to_vector_literal(query_embedding)
        try:
            self._cur.execute(sql, (vec_literal, domain, domain, exclude, vec_literal, top_k))
            rows = self._cur.fetchall()
        except Exception as e:
            try:
                self._conn.rollback()
            except Exception:
                pass
            _logger.warning(f"Tool-RAG 检索失败: {e}")
            return []

        # ------------------------------------------------------------------
        # 【双重阈值】绝对下限 + 相对下限，两条都要满足
        #   threshold = max(min_score, 本轮最高分 * relative_ratio)
        # ------------------------------------------------------------------
        scores = [float(r[5]) for r in rows]
        threshold = max(min_score, max(scores) * relative_ratio) if scores else min_score

        hits = []
        dropped = 0
        for name, description, dtag, stype, server, score in rows:
            score = float(score)
            if score < threshold:
                dropped += 1
                continue
            hits.append({
                "name": name, "description": description, "domain_tag": dtag,
                "source_type": stype, "server_name": server or "", "score": round(score, 4),
            })
        _logger.info(
            f"Tool-RAG 召回: domain={domain or '全部'} exclude={exclude} "
            f"命中 {len(hits)} 个，低于阈值 {threshold:.4f} 丢弃 {dropped} 个"
        )
        return hits

    # 【什么】关键词降级检索
    # 【作用】PG 可用但 embedding 不可用时兜底；分数上限 0.60，与向量结果一眼区分
    def search_keyword(
        self,
        keywords: Sequence[str],               # 关键词列表，来自意图识别的 keyword_list
        domain_tag: str = "",                  # 业务域过滤，规则同 search()
        exclude: Optional[List[str]] = None,   # 黑名单工具名
        top_k: int = 3,                        # 返回数量上限（先捞 200 行再排序截取）
    ) -> List[Dict[str, Any]]:
        """关键词降级检索（PG 可用但 embedding 不可用，或向量召回为空时用）。

        【打分规则】
            命中关键词数 / 关键词总数 * 0.60，上限 0.60 ——
            刻意低于向量召回的典型分数区间（0.6~0.9），
            这样看分数就知道这次结果来自降级路径。

        【输入输出对照】
            keywords=["报销","发票"], top_k=3
                → [{"name":"query_reimbursement","score":0.30}, ...]
                （命中 1/2 个关键词 → 0.5 * 0.60 = 0.30）
        """
        if not self._is_alive() or not keywords:
            return []

        domain = domain_tag if domain_tag and str(domain_tag).lower() not in _NO_DOMAIN_FILTER else None
        exclude = [str(x) for x in (exclude or []) if str(x).strip()]

        sql = f"""
            SELECT name, description, domain_tag, source_type, server_name
            FROM {self._table}
            WHERE enabled = TRUE
              AND (%s IS NULL OR domain_tag = %s)
              AND NOT (name = ANY(%s))
            LIMIT 200
        """
        try:
            self._cur.execute(sql, (domain, domain, exclude))
            rows = self._cur.fetchall()
        except Exception as e:
            _logger.warning(f"关键词检索失败: {e}")
            return []

        lowered = [str(k).lower() for k in keywords if str(k).strip()]
        hits = []
        for name, description, dtag, stype, server in rows:
            blob = f"{name} {description}".lower()
            hit = sum(1 for k in lowered if k in blob)
            if hit == 0:
                continue
            hits.append({
                "name": name, "description": description, "domain_tag": dtag,
                "source_type": stype, "server_name": server or "",
                "score": round(hit / len(lowered) * _KEYWORD_SCORE_BASE, 4),
            })
        hits.sort(key=lambda x: x["score"], reverse=True)
        _logger.info(f"关键词降级召回: 命中 {len(hits)} 个 → 取前 {top_k}")
        return hits[:top_k]

    # 【什么】统计表里工具记录条数
    # 【作用】自检用；查不到返回 -1，区别于"表存在但为空"的 0
    def count(self) -> int:
        """表里有多少条工具记录（-1 表示查不到）。"""
        if not self._is_alive():
            return -1
        try:
            self._cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return int(self._cur.fetchone()[0])
        except Exception:
            return -1

    # 【什么】关闭游标与连接
    # 【作用】进程退出前释放资源；内部吞掉异常，避免退出流程被连接错误打断
    def close(self) -> None:
        """关闭连接（进程退出前调用）。"""
        try:
            if self._cur:
                self._cur.close()
            if self._conn:
                self._conn.close()
        except Exception:
            pass
        self._ok = False


# 【什么】延迟导入并返回 embedding 服务单例
# 【作用】函数内 import，避免本模块在无网络/缺依赖环境下导入即失败
def get_embedding_service():
    """延迟导入 embedding 服务（避免本模块在无网络环境下导入即失败）。"""
    from modules.memory.embedding_service import get_embedding_service as _get
    return _get()
