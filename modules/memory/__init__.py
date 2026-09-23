"""
记忆模块
=========
PG+pgvector 对话记忆系统：原文+摘要双存储、摘要向量召回、上下文动态压缩、Agent状态持久化。
覆盖边界4（长对话上下文溢出）、边界8（记忆召回不准）。
"""
# 【什么】本包对外暴露的 5 个符号：PG 管理、向量服务、摘要器、记忆管理器
# 【作用】统一出口，其它模块只从包这一层导入，无需记住子模块文件名
from modules.memory.pg_manager import PGManager
from modules.memory.embedding_service import get_embedding_service, EmbeddingService
from modules.memory.summary_generator import SummaryGenerator
from modules.memory.memory_manager import MemoryManager

# 【什么】__all__：星号导入时真正带出去的名字清单
# 【作用】限定导出面，未列出的内部符号不会被 from ... import * 带出
__all__ = [
    "PGManager", "get_embedding_service", "EmbeddingService",
    "SummaryGenerator", "MemoryManager",
]
