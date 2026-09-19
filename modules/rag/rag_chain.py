"""
RAG 链路模块
============
整合文档加载 → 切片 → 向量化 → 入库 → 召回 → 精排 → 上下文拼接的完整 RAG 闭环。
覆盖边界1（RAG无结果/低相关降级处理）。
"""
from typing import List, Optional
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

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """
        知识库检索：Milvus向量召回 → BGE精排过滤。
        返回精排后的 top_k 相关文档列表。
        """
        k = top_k or self._top_k
        # 第一步：Milvus 向量召回
        candidates = self.vector_store.search(query, top_k=k)
        if not candidates:
            _logger.warning(f"RAG无结果，触发兜底（边界1）: query='{query[:50]}'")
            return []
        # 第二步：BGE 精排
        reranked = self.reranker.rerank(query, candidates, top_k=self._rerank_top_k)
        return reranked if reranked else []

    def build_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        检索知识库并拼接为上下文文本，供LLM问答使用。
        无结果时返回空字符串，禁止模型幻觉编造。
        """
        docs = self.retrieve(query, top_k=top_k)
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
