"""
RAG 链路模块
============
整合文档加载 → 切片 → 向量化 → 入库 → 召回 → 精排 → 上下文拼接的完整 RAG 闭环。
覆盖边界1（RAG无结果/低相关降级处理）。
"""
from typing import List, Optional
from datetime import datetime
from modules.common.logger import get_logger
from modules.common.exceptions import RAGException
from modules.rag.document_loader import DocumentLoader, Document
from modules.rag.text_splitter import TextChunk, TextSplitter
from modules.rag.vector_store import MilvusVectorStore
from modules.rag.reranker import BGEReranker
from config import get_settings

_logger = get_logger("rag_chain")


class RAGChain:
    """RAG 全链路管理器。"""

    def __init__(   
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = MilvusVectorStore()
        self.reranker = BGEReranker()   
        self._rerank_top_k = get_settings().rerank_top_k
        self._top_k = get_settings().rag_top_k
        self._keyword_fallback = get_settings().rag_keyword_fallback
        self._recency_boost = get_settings().rag_recency_boost

    def ingest_document(self, file_path: str) -> int:
        """
        文档入库全流程：加载 → 切片 → 向量化 → 写入Milvus。
        返回入库分片数。
        """
        doc = self.loader.load(file_path)
        chunks: List[TextChunk] = self.splitter.split_document(doc)
        if not chunks:
            _logger.warning(f"文档无有效内容: {file_path}")
            return 0
        self.vector_store.insert_chunks(chunks)
        _logger.info(f"文档入库完成: {file_path} | 分片={len(chunks)}")
        return len(chunks)

    def ingest_directory(self, dir_path: str) -> int:
        """批量入库目录下所有文档。"""
        docs = self.loader.load_batch(dir_path)
        total = 0
        for doc in docs:
            chunks = self.splitter.split_document(doc)
            if chunks:
                self.vector_store.insert_chunks(chunks)
                total += len(chunks)
        _logger.info(f"批量入库完成: 共 {total} 分片")
        return total

    def retrieve(self, query: str, top_k: Optional[int] = None, expr: Optional[str] = None) -> List[dict]:
        """
        知识库检索：Milvus向量召回 → BGE精排过滤。
        返回精排后的 top_k 相关文档列表。

        Args:
            expr: 知识库权限过滤表达式（由 RBACService.kb_expr 生成）。
                  None = 不限制。权限隔离在这里生效：越权的文档根本不会被召回，
                  而不是召回后再剔除——后者等于把机密内容送进了模型上下文。
        """
        k = top_k or self._top_k
        candidates = self.vector_store.search(query, top_k=k, expr=expr)

        if not candidates and self._keyword_fallback:
            candidates = self.vector_store.keyword_search(query, top_k=k, expr=expr)
            if candidates:
                _logger.info(f"边界1降级：向量无结果，关键词检索补充命中 {len(candidates)} 条")

        if not candidates:
            _logger.warning(f"RAG无结果，触发兜底（边界1）: query='{query[:50]}'")
            return []

        if self._recency_boost:
            self._apply_recency_boost(candidates)

        reranked = self.reranker.rerank(query, candidates, top_k=self._rerank_top_k)
        return reranked if reranked else []

    @staticmethod
    def _apply_recency_boost(candidates: list) -> None:
        """边界5：按文档更新时间加权，30天内 +0.05、90天内 +0.02，避免旧版文档压过新版。"""
        now = datetime.now()
        for c in candidates:
            try:
                days = (now - datetime.strptime(c.get("updated_at", ""), "%Y-%m-%d %H:%M:%S")).days
            except Exception:
                continue
            bonus = 0.05 if days <= 30 else (0.02 if days <= 90 else 0.0)
            if bonus:
                c["score"] = round(c.get("score", 0.0) + bonus, 4)

    def build_context(self, query: str, top_k: Optional[int] = None, expr: Optional[str] = None) -> str:
        """
        检索知识库并拼接为上下文文本，供LLM问答使用。
        无结果时返回空字符串，禁止模型幻觉编造。
        """
        docs = self.retrieve(query, top_k=top_k, expr=expr)
        if not docs:
            return ""
        blocks = []
        for i, doc in enumerate(docs, 1):
            score = doc.get("rerank_score", doc.get("score", 0))
            source = doc.get("file_name", doc.get("source", "未知"))
            blocks.append(f"【知识{i}】(相关度:{score:.2f} 来源:{source})\n{doc['content']}")
        return "\n\n".join(blocks)

    def get_knowledge_stats(self) -> dict:
        """获取知识库统计信息。"""
        return {
            "total_chunks": self.vector_store.count(),
            "collection": self.vector_store._collection_name,
        }
