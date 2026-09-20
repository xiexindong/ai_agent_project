"""
Milvus 向量存储模块
====================
静态知识库向量库：文档向量化、入库、相似度召回。
覆盖边界1（RAG无结果降级）、边界5（知识库过期管理）。
"""
from typing import List
from datetime import datetime
from modules.common.logger import get_logger
from modules.common.exceptions import RAGException, handle_exception
from modules.memory.embedding_service import get_embedding_service
from modules.rag.text_splitter import TextChunk
from config import get_settings

_logger = get_logger("milvus")


class MilvusVectorStore:
    """Milvus 向量数据库管理器。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # s 是全局配置单例（config.py 的 Settings），所有 *_xxx 参数都从它身上取值，
        # 也都可以用 .env 同名环境变量覆盖，所以这里没有一个值是写死的。
        s = get_settings()

        # 【_collection_name】Milvus 里的"集合名"，等价于 MySQL 的"表名"。
        #   默认 knowledge_base（config.py:58 ← .env 的 MILVUS_COLLECTION）。
        #   本项目只有一个集合——"文档知识库 RAG"和"长期记忆"用的是两套不同的库：
        #   这里走 Milvus，长期记忆走 PostgreSQL + pgvector（pg_manager.py），别混。
        self._collection_name = s.milvus_collection

        # 【_dim】向量维度，默认 1024（config.py:43 ← .env 的 EMBEDDING_DIMENSION）。
        #   必须和 embedding 模型的输出维度严格一致：本项目用智谱 embedding-2，固定产出 1024 维。
        #   它被用在两个地方：
        #     ① 建集合时声明字段类型 → _ensure_collection() 里 embedding 字段的 dim=self._dim
        #     ② 间接约束 pg_manager.py:280 建表的 vector(1024)
        #   三者必须同步改，否则入库时报 "dimension mismatch"。
        #
        #   【陷阱】这个值只在"集合不存在、第一次创建"时才生效（has_collection 为 False 才走创建）。
        #   集合一旦建成，再改 EMBEDDING_DIMENSION 不会有任何效果——
        #   Milvus 不会给你改表，必须删集合重来。
        self._dim = s.embedding_dimension

        # 【_top_k】一次粗召回最多返回几条候选，默认 5（config.py:69 ← .env 的 RAG_TOP_K）。
        #   用在下面 search() 的 limit=k 参数上，是"召回上限"，不是"最终条数"：
        #   粗召回 5 条 → 再被 threshold 过滤一遍 → 剩下的才交给 rag_chain 精排。
        #   search(query, top_k=...) 允许调用方临时覆盖，不传就用这里的值
        #   （search() 里 k = top_k or self._top_k）。
        self._top_k = s.rag_top_k

        # 【_threshold】余弦相似度及格线，默认 0.65（config.py:71 ← .env 的 SIMILARITY_THRESHOLD）。
        #   用在下面 search() 的过滤循环里：score < threshold 的结果直接丢弃。
        #   作用是边界1/边界8——防止"用户问天气、召回到股票"的误匹配污染 Prompt。
        #   调小 → 召得宽但噪声多；调大 → 宁缺毋滥，可能一条都不剩。
        #
        #   【注意】这个阈值是"两个模块共用"的：
        #     modules/rag/vector_store.py  ← 文档知识库 RAG（Milvus + COSINE）
        #     modules/memory/memory_manager.py:31  ← 长期记忆召回（pgvector）
        #   两处读的是同一个 config 字段。改 .env 会同时影响两边，别只想着一边。
        self._threshold = s.similarity_threshold

        # 连接 + 确保集合存在 + 建索引 + load 到内存，全在这一步里完成。
        #
        # 【一个容易忽略的点】上面 __new__ 是单例，但单例挡不住 __init__ 重跑：
        #   每写一次 MilvusVectorStore()，__new__ 返回同一个对象，可 __init__ 照常执行一遍，
        #   于是 connections.connect(...) 又被调用一次。
        #   pymilvus 的 connect 是幂等的（同 alias 重复连接覆盖同一条连接，不会泄漏），所以没问题，
        #   但如果哪天在里面塞了有副作用的代码，这里就是个坑。
        self._connect()

    def _connect(self):
        """连接 Milvus 并确保集合存在。"""
        try:
            # pymilvus 放在函数内 import，而不是文件顶部：
            #   这是刻意的——模块 import 失败（pip 没装 pymilvus）不该让整个项目起不来，
            #   要让它在用到时才失败、并被下面 58-60 行的 except 转成 RAGException 交给上层降级。
            #   对比：pg_manager.py 的 psycopg2 也是同样的策略。
            from pymilvus import (
                connections, utility, Collection,
                CollectionSchema, FieldSchema, DataType,
            )
            s = get_settings()

            # connections.connect 只做一件事：建立一个带 alias 的连接。
            #   alias="default" 是这条连接的"名字"——后续 Collection(...)、utility.xxx()
            #   都不传 host/port，靠的就是这个默认 alias 自己去认。类比 psycopg2 的 conn 对象。
            #   host 默认 localhost / port 默认 19530（config.py:56-57 ← MILVUS_HOST / MILVUS_PORT）。
            #   port 要 str()：settings 里是 int，但 pymilvus 这版只吃字符串，传 int 会静默连不上。
            #
            #   【这里没有 timeout 也没有重试】Milvus 没起时不会立刻报错，
            #   而是在这里阻塞到 pymilvus 自己的默认超时才抛异常。
            connections.connect(
                alias="default",
                host=s.milvus_host,
                port=str(s.milvus_port),
            )

            # 下面 5 行：把 import 进来的 pymilvus 符号挂到 self 上，做成"实例持有"。
            #   这么做是为了让 _ensure_collection / insert_chunks 等方法也能用到它们，
            #   而不用每个方法都重复 import 一遍。
            self._connections = connections
            # pymilvus 2.6：has_collection 等集合管理函数在 utility 模块下，不在 connections 上
            self._utility = utility
            self._Collection = Collection
            self._DataType = DataType
            self._FieldSchema = FieldSchema
            self._CollectionSchema = CollectionSchema
            self._ensure_collection()
            _logger.info(f"Milvus连接成功: {s.milvus_host}:{s.milvus_port}")
        except Exception as e:
            _logger.error(f"Milvus连接失败: {e}")
            raise RAGException("Milvus连接失败", str(e))

    def _ensure_collection(self):
        """确保知识库集合存在，不存在则创建。"""
        s = get_settings()
        if not self._utility.has_collection(self._collection_name):
            fields = [
                self._FieldSchema(name="chunk_id", dtype=self._DataType.VARCHAR, max_length=128, is_primary=True),
                self._FieldSchema(name="doc_id", dtype=self._DataType.VARCHAR, max_length=128),
                self._FieldSchema(name="content", dtype=self._DataType.VARCHAR, max_length=65535),
                self._FieldSchema(name="source", dtype=self._DataType.VARCHAR, max_length=512),
                self._FieldSchema(name="file_name", dtype=self._DataType.VARCHAR, max_length=256),
                self._FieldSchema(name="updated_at", dtype=self._DataType.VARCHAR, max_length=32),
                # 知识库隔离字段：kb_code 对应 sys_kb.kb_code，检索时按用户可见范围做 expr 过滤
                # dept_code 冗余一份归口部门，便于按部门维度下线/审计
                self._FieldSchema(name="kb_code", dtype=self._DataType.VARCHAR, max_length=64),
                self._FieldSchema(name="dept_code", dtype=self._DataType.VARCHAR, max_length=64),
                self._FieldSchema(name="embedding", dtype=self._DataType.FLOAT_VECTOR, dim=self._dim),
            ]
            schema = self._CollectionSchema(fields, description="企业知识库RAG向量索引")
            collection = self._Collection(self._collection_name, schema=schema)
            # 创建向量索引
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                },
            )
            _logger.info(f"Milvus集合创建: {self._collection_name}")
        self._collection = self._Collection(self._collection_name)
        self._collection.load()
        # 兼容老集合：升级前建的集合没有 kb_code/dept_code 字段，
        # 插入与检索都要按"有没有这个字段"走不同分支，避免直接报错。
        try:
            self._has_kb_field = any(f.name == "kb_code" for f in self._collection.schema.fields)
        except Exception:
            self._has_kb_field = False
        _logger.info(f"Milvus集合就绪: {self._collection_name} | 知识库隔离字段={self._has_kb_field}")

    def insert_chunks(self, chunks: List[TextChunk]):
        """将文本分片向量化后入库。"""
        if not chunks:
            return 0
        emb_service = get_embedding_service()
        vectors = emb_service.embed_batch([c.content for c in chunks])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = [
            [c.metadata.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)],
            [c.metadata.get("doc_id", "") for c in chunks],
            [c.content for c in chunks],
            [c.metadata.get("source", "") for c in chunks],
            [c.metadata.get("file_name", "") for c in chunks],
            [now] * len(chunks),
        ]
        if getattr(self, "_has_kb_field", False):
            # 未显式标注 kb_code 的一律归入 public（公司通用制度）
            data.append([c.metadata.get("kb_code", "public") for c in chunks])
            data.append([c.metadata.get("dept_code", "") for c in chunks])
        data.append(vectors)
        result = self._collection.insert(data)
        self._collection.flush()
        _logger.info(f"向量入库完成: {len(chunks)} 条分片")
        return len(chunks)

    @handle_exception(default=None, log_msg="Milvus向量召回异常", reraise=False)
    def search(self, query: str, top_k: int = None, expr: str = None) -> list[dict]:
        """
        向量相似度召回，返回 [{content, source, file_name, score, ...}]。
        相似度低于 threshold 的结果会被过滤。

        Args:
            expr: Milvus 标量过滤表达式，用于知识库权限隔离。
                  例：'kb_code in ["public","finance"]' → 只在财务+公开库里搜。
                  None = 不限制（保持旧行为）。老集合无 kb_code 字段时会自动回退。
        """
        emb_service = get_embedding_service()
        query_vec = emb_service.embed(text=query)
        k = top_k or self._top_k

        # ------------------------------------------------------------------
        # 【做什么】向 Milvus 发起一次 ANN（近似最近邻）检索：
        #   拿 1 个问句向量，在整张表里找出和它最相似的 k 条分片。
        #
        # 【为什么下面要写 results[0]】search() 支持「一次传多个 query 批量搜」，
        #   所以它的返回值外层永远是「每个 query 一个结果集」：
        #     results     = [结果集0]                 ← 只传了 1 个 query，长度就是 1
        #     results[0]  = [Hit0, Hit1, ...]         ← 下面 for hit in results[0] 遍历的就是这一层
        #   少写 [0] 会拿着一个列表去当 Hit 用，取 .score 直接 AttributeError。
        #
        # 【输入输出对照】
        #   输入: query_vec = [0.021, -0.117, ...] 共 1024 个 float，k = 5
        #   输出: results[0] 最多 5 个 Hit，每个形如
        #         score = 0.8321
        #         entity = {'content': '年假按工龄计算...',
        #                   'file_name': '员工手册.pdf', 'source': '...', ...}
        #   注意 hit.score 是【相似度】，越大越像（-1~1，文本向量实际落在 0~1），
        #   不是「距离」（越小越像）。下面 if score < self._threshold(0.65) 就丢弃，
        #   用的正是"越大越像"这个方向。
        #
        # 【依赖警告】这里的 metric_type 必须和建索引时（_ensure_collection 里的 create_index）一致：
        #   建索引写的是 IVF_FLAT + COSINE + nlist=128，所以这里也得是 COSINE。
        #   nprobe 的上限就是 nlist=128：填 200 无意义，填 1 只搜 1/128 的表，
        #   召回率明显下降却不报错——这是静默降质，不是崩溃，日志里看不出来。
        # ------------------------------------------------------------------
        base_fields = ["chunk_id", "doc_id", "content", "source", "file_name", "updated_at"]
        extra_fields = ["kb_code", "dept_code"] if getattr(self, "_has_kb_field", False) else []
        try:
            results = self._search_once(query_vec, k, expr, base_fields + extra_fields)
        except Exception as e:
            if not expr:
                raise
            # 老集合缺少 kb_code 字段时 expr 会执行失败：宁可放开检索也不能让对话失败，
            # 但要留 warning —— 这是"权限隔离失效"的信号，必须能被监控发现。
            _logger.warning(f"知识库过滤失败，回退无过滤检索（疑似集合缺 kb_code 字段）: {e}")
            results = self._search_once(query_vec, k, None, base_fields)

        hits = []
        for hit in results[0]:
            score = hit.score
            if score < self._threshold:
                _logger.debug(f"低相关结果过滤: score={score:.4f} < {self._threshold}")
                continue
            hits.append({
                "content": hit.entity.get("content", ""),
                "source": hit.entity.get("source", ""),
                "file_name": hit.entity.get("file_name", ""),
                "doc_id": hit.entity.get("doc_id", ""),
                "updated_at": hit.entity.get("updated_at", ""),
                "score": round(score, 4),
            })
        _logger.info(
            f"向量召回: query='{query[:30]}...' | 命中={len(hits)}/{k} | "
            f"知识库过滤={'有' if expr else '无'}"
        )
        return hits

    def _search_once(self, query_vec: list, k: int, expr: str, output_fields: list):
        """执行一次 ANN 检索；expr 为空时不下过滤条件（兼容老集合）。"""
        kwargs = dict(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=k,
            output_fields=output_fields,
        )
        if expr:
            kwargs["expr"] = expr
        return self._collection.search(**kwargs)

    @handle_exception(default=False, log_msg="Milvus文档下线异常")
    def remove_by_doc_id(self, doc_id: str) -> bool:
        """按文档ID下线过期文档（边界5：知识库过期管理）。"""
        expr = f'doc_id == "{doc_id}"'
        self._collection.delete(expr)
        self._collection.flush()
        _logger.info(f"文档下线完成: doc_id={doc_id}")
        return True

    @handle_exception(default=0, log_msg="Milvus统计异常")
    def count(self) -> int:
        """统计当前知识库向量总数。"""
        self._collection.flush()
        return self._collection.num_entities
