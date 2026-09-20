"""
Milvus 向量存储模块
====================
静态知识库向量库：文档向量化、入库、相似度召回。
覆盖边界1（RAG无结果降级）、边界5（知识库过期管理）。
"""
from typing import List, Optional
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
        s = get_settings()

        self._collection_name = s.milvus_collection

        self._dim = s.embedding_dimension

        self._top_k = s.rag_top_k

        self._threshold = s.similarity_threshold

        self._connect()

    def _connect(self):
        """连接 Milvus 并确保集合存在。"""
        try:
            from pymilvus import (
                connections, utility, Collection,
                CollectionSchema, FieldSchema, DataType,
            )
            s = get_settings()

            connections.connect(
                alias="default",
                host=s.milvus_host,
                port=str(s.milvus_port),
            )

            self._connections = connections
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
                self._FieldSchema(name="kb_code", dtype=self._DataType.VARCHAR, max_length=64),
                self._FieldSchema(name="dept_code", dtype=self._DataType.VARCHAR, max_length=64),
                self._FieldSchema(name="embedding", dtype=self._DataType.FLOAT_VECTOR, dim=self._dim),
            ]
            schema = self._CollectionSchema(fields, description="企业知识库RAG向量索引")
            collection = self._Collection(self._collection_name, schema=schema)
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

        base_fields = ["chunk_id", "doc_id", "content", "source", "file_name", "updated_at"]
        extra_fields = ["kb_code", "dept_code"] if getattr(self, "_has_kb_field", False) else []
        try:
            results = self._search_once(query_vec, k, expr, base_fields + extra_fields)
        except Exception as e:
            if not expr:
                raise
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

    @handle_exception(default=[], log_msg="Milvus关键词降级检索异常")
    def keyword_search(self, query: str, top_k: Optional[int] = None, expr: Optional[str] = None) -> list[dict]:
        """边界1降级通道：向量无结果时，用 Milvus 的 like 表达式做关键词粗召回。

        命中结果的 score 统一置为 threshold，交给下游 reranker 用关键词分重排，
        避免"降级结果分数为 0 又被阈值过滤掉"的二次丢失。
        """
        import re
        keywords = [
            w for w in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", query)
        ]
        if not keywords:
            return []
        keywords.sort(key=len, reverse=True)
        keywords = keywords[:3]

        conds = " or ".join(f'content like "%{kw}%"' for kw in keywords)
        final_expr = f"({conds})" + (f" and {expr}" if expr else "")
        rows = self._collection.query(
            expr=final_expr,
            output_fields=["chunk_id", "doc_id", "content", "source", "file_name", "updated_at"],
            limit=top_k or self._top_k,
        )
        hits = [{
            "content": r.get("content", ""),
            "source": r.get("source", ""),
            "file_name": r.get("file_name", ""),
            "doc_id": r.get("doc_id", ""),
            "updated_at": r.get("updated_at", ""),
            "score": self._threshold,
            "recall_type": "keyword",
        } for r in rows]
        _logger.info(f"关键词降级召回: 关键词={keywords} | 命中={len(hits)}")
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
