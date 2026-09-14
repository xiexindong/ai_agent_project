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
            vectors,
        ]
        result = self._collection.insert(data)
        self._collection.flush()
        _logger.info(f"向量入库完成: {len(chunks)} 条分片")
        return len(chunks)

    @handle_exception(default=None, log_msg="Milvus向量召回异常", reraise=False)
    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        向量相似度召回，返回 [{content, source, file_name, score, ...}]。
        相似度低于 threshold 的结果会被过滤。
        """
        emb_service = get_embedding_service()
        query_vec = emb_service.embed(query)
        k = top_k or self._top_k

        results = self._collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=k,
            output_fields=["chunk_id", "doc_id", "content", "source", "file_name", "updated_at"],
        )

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
        _logger.info(f"向量召回: query='{query[:30]}...' | 命中={len(hits)}/{k}")
        return hits

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
