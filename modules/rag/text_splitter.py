"""
文本切分模块
============
递归字符分割 + 重叠分片，保障上下文完整性。
"""
from typing import List
from dataclasses import dataclass
from modules.rag.document_loader import Document
from modules.common.logger import get_logger

_logger = get_logger("splitter")


@dataclass
class TextChunk:
    """文本分片：内容 + 元数据 + 序号。"""
    content: str
    metadata: dict
    chunk_index: int
    total_chunks: int


class TextSplitter:
    """递归字符分割器，支持重叠分片。"""

    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        separators: list | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split_text(self, text: str) -> List[str]:
        """递归切分文本，保证分片不超过 chunk_size，相邻分片有 overlap 重叠。"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        self._recursive_split(text, chunks)
        # 重叠合并：相邻分片之间保留 chunk_overlap 字符重叠
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        return chunks

    def _recursive_split(self, text: str, chunks: list, depth: int = 0):
        """递归按分隔符切分。"""
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return
        if depth >= len(self.separators):
            # 所有分隔符都切不开，按字符硬切
            for i in range(0, len(text), self.chunk_size):
                piece = text[i:i + self.chunk_size].strip()
                if piece:
                    chunks.append(piece)
            return

        sep = self.separators[depth]
        if sep == "":
            # 硬切兜底
            for i in range(0, len(text), self.chunk_size):
                piece = text[i:i + self.chunk_size].strip()
                if piece:
                    chunks.append(piece)
            return

        parts = text.split(sep)
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # 当前 part 本身超长，递归用下一级分隔符
                if len(part) > self.chunk_size:
                    self._recursive_split(part, chunks, depth + 1)
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current.strip())

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """给相邻分片添加重叠，保留上下文连贯性。"""
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.chunk_overlap:]
            result.append(prev_tail + chunks[i])
        return result

    def split_document(self, doc: Document) -> List[TextChunk]:
        """切分文档，返回带元数据的分片列表。"""
        pieces = self.split_text(doc.content)
        total = len(pieces)
        chunks = [
            TextChunk(
                content=p,
                metadata={**doc.metadata, "chunk_id": f"{doc.metadata.get('doc_id', 'doc')}_{i}"},
                chunk_index=i,
                total_chunks=total,
            )
            for i, p in enumerate(pieces)
        ]
        _logger.info(
            f"文档切分完成: {doc.metadata.get('file_name', 'unknown')} | "
            f"分片数={total} | chunk_size={self.chunk_size} | overlap={self.chunk_overlap}"
        )
        return chunks
