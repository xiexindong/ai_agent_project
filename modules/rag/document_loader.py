"""
文档加载模块
============
支持 TXT / MD / PDF 多格式文档加载，提取纯文本。
"""
from pathlib import Path
from typing import List
from dataclasses import dataclass
from modules.common.logger import get_logger

_logger = get_logger("doc_loader")


@dataclass
class Document:
    """文档数据结构：内容 + 元数据。"""
    content: str
    metadata: dict


class DocumentLoader:
    """多格式文档加载器。"""

    SUPPORTED_TYPES = {".txt", ".md", ".pdf"}

    @classmethod
    def load(cls, file_path: str) -> Document:
        """加载单个文档，自动识别格式。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文档不存在: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in cls.SUPPORTED_TYPES:
            raise ValueError(f"不支持的文档格式: {suffix}，仅支持 {cls.SUPPORTED_TYPES}")

        if suffix == ".pdf":
            content = cls._load_pdf(path)
        elif suffix == ".md":
            content = cls._load_text(path)
        else:
            content = cls._load_text(path)

        doc = Document(
            content=content,
            metadata={
                "source": str(path),
                "file_type": suffix,
                "file_name": path.name,
                "doc_id": path.stem,
            },
        )
        _logger.info(f"文档加载完成: {path.name} | 字符数={len(content)}")
        return doc

    @classmethod
    def load_batch(cls, dir_path: str) -> List[Document]:
        """批量加载目录下所有支持格式的文档。"""
        directory = Path(dir_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"路径不是目录: {dir_path}")

        docs = []
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in cls.SUPPORTED_TYPES:
                try:
                    docs.append(cls.load(str(path)))
                except Exception as e:
                    _logger.error(f"加载失败 {path.name}: {e}")
        _logger.info(f"批量加载完成: 共 {len(docs)} 个文档")
        return docs

    @staticmethod
    def _load_text(path: Path) -> str:
        """加载 TXT / MD 文本文件。"""
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _load_pdf(path: Path) -> str:
        """加载 PDF 文档（基于 pypdf）。"""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("请安装 pypdf: pip install pypdf")

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
