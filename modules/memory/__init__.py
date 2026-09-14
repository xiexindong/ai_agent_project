"""
记忆模块
=========
PG+pgvector 对话记忆系统：原文+摘要双存储、摘要向量召回、上下文动态压缩、Agent状态持久化。
覆盖边界4（长对话上下文溢出）、边界8（记忆召回不准）。
"""
from modules.memory.pg_manager import PGManager
from modules.memory.embedding_service import get_embedding_service, EmbeddingService
from modules.memory.summary_generator import SummaryGenerator
from modules.memory.memory_manager import MemoryManager

__all__ = [
    "PGManager", "get_embedding_service", "EmbeddingService",
    "SummaryGenerator", "MemoryManager",
]
