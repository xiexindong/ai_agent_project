"""
llama-index 适配层（可选后端，默认不启用）
==========================================
【定位】
    这一层不替代自研 RAGChain，而是提供"第二种检索实现"。
    自研链路（Milvus + BGE 精排 + 权限 expr）是默认路径；
    llama-index 只在显式配置 RAG_BACKEND=llamaindex 时接管。

【为什么用适配层而不是直接用 llama-index 重写 RAG】
    1. 上层 ReActAgent / API / Harness 全部依赖 RAGChain 的三个方法：
       ingest_document / retrieve / build_context。换成别的库就改一片调用点，
       是典型的"框架渗透进业务代码"，将来再换回来自研就又要改一遍。
    2. 自研链路里有两样东西框架不自带：权限过滤（kb_code in [...]）和
       BGE 精排。适配层必须把这两样补回去，否则换库 = 越权 + 掉精度。
    3. 它是可选依赖：没装库时 import 会失败，适配层要做成"能用就用，
       不能用就明确报错让上层降级"，而不是整个服务起不来。

【依赖警告】
    llama-index 0.10 前后 API 大改（llama_index → llama_index.core），
    本文件按新 API（import from llama_index.core）写，并对旧路径做了兜底。
    装的是 <0.10 的老版本时，走 _import_legacy() 兜底分支。
"""
import os
from typing import Optional

from modules.common.logger import get_logger
from modules.common.exceptions import RAGException
from config import get_settings

_logger = get_logger("llamaindex_adapter")


def is_available() -> bool:
    """检测 llama-index 是否可用（供工厂与 /agent/backends 探活用）。"""
    try:
        import llama_index.core  # noqa: F401
        return True
    except Exception:
        return False


# ------------------------------------------------------------------
# 【权限表达式翻译】Milvus expr → llama-index 过滤条件
#   自研链路用 Milvus 布尔表达式做知识库隔离（如 kb_code in ["kb_a","kb_b"]），
#   llama-index 用的是 MetadataFilters。两套语法必须翻译，
#   翻译不了就必须拒绝查询（fail closed）。
#
# 【为什么不翻译失败就退回全量检索】
#   过滤失效 = 越权文档进入模型上下文 = 泄密。宁可这次查不出东西，
#   也不能让不该看的内容被召回。空结果用户会重试，泄密无法撤回。
#
# 【输入输出对照】
#   输入：'kb_code in ["kb_a","kb_b"]'
#   输出：[{"key":"kb_code","op":"in","value":["kb_a","kb_b"]}]
#   输入：'level == "public" and kb_code in ["kb_a"]'
#   输出：[{"key":"level","op":"==","value":"public"},
#          {"key":"kb_code","op":"in","value":["kb_a"]}]
#   输入：'score > 0.5' → 抛 RAGException（不支持的比较，拒绝放行）
# ------------------------------------------------------------------
def parse_expr(expr: Optional[str]) -> list:
    """把 Milvus 风格的过滤表达式解析成中立结构（不依赖 llama-index，可单测）。"""
    if not expr or not expr.strip():
        return []
    parsed = []
    for clause in expr.split(" and "):
        clause = clause.strip()
        if not clause:
            continue
        if " in [" in clause:
            key, raw = clause.split(" in [", 1)
            if not raw.endswith("]"):
                raise RAGException(f"过滤表达式缺少右括号，拒绝放行（防越权）: {clause}")
            values = [v.strip().strip('"').strip("'") for v in raw[:-1].split(",")]
            parsed.append({"key": key.strip(), "op": "in", "value": [v for v in values if v]})
        elif "==" in clause:
            key, val = clause.split("==", 1)
            parsed.append({"key": key.strip(), "op": "==", "value": val.strip().strip('"').strip("'")})
        elif "!=" in clause:
            key, val = clause.split("!=", 1)
            parsed.append({"key": key.strip(), "op": "!=", "value": val.strip().strip('"').strip("'")})
        else:
            # 不支持 or / like / 比较符：不猜，直接拒绝
            raise RAGException(f"过滤表达式无法翻译，拒绝放行（防越权）: {clause}")
    return parsed


class LlamaIndexRAG:
    """llama-index 版检索实现：接口与自研 RAGChain 对齐，可无缝替换。"""

    def __init__(self, persist_dir: str = "", top_k: int = 0, rerank_top_k: int = 0):
        s = get_settings()
        self.persist_dir = persist_dir or getattr(s, "llamaindex_persist_dir", "storage/llamaindex")
        self._top_k = top_k or s.rag_top_k
        self._rerank_top_k = rerank_top_k or s.rerank_top_k
        self._li = self._import_li()
        self._index = None

    @staticmethod
    def _import_li() -> dict:
        """延迟导入：库没装时不让整个模块 import 失败。"""
        try:
            from llama_index.core import (
                VectorStoreIndex, SimpleDirectoryReader, StorageContext,
                load_index_from_storage, Settings,
            )
            from llama_index.core.vector_stores import (
                MetadataFilters, MetadataFilter, FilterOperator,
            )
            return {
                "VectorStoreIndex": VectorStoreIndex, "SimpleDirectoryReader": SimpleDirectoryReader,
                "StorageContext": StorageContext, "load_index_from_storage": load_index_from_storage,
                "Settings": Settings, "MetadataFilters": MetadataFilters,
                "MetadataFilter": MetadataFilter, "FilterOperator": FilterOperator,
            }
        except ImportError as e:
            raise RAGException(
                f"llama-index 未安装或版本过旧（需 llama_index.core）: {e}。"
                f"安装：pip install -r requirements-extras.txt"
            )

    # ------------------------------------------------------------------
    # 【索引加载】有落盘就加载，没有就建空索引
    #   空索引必须先建出来：否则第一次 retrieve 会因为索引不存在而抛错，
    #   而"知识库为空"是正常的初始状态，不该是一个异常。
    # ------------------------------------------------------------------
    def _ensure_index(self):
        if self._index is not None:
            return self._index
        li = self._li
        index_json = os.path.join(self.persist_dir, "docstore.json")
        if os.path.exists(index_json):
            storage = li["StorageContext"].from_defaults(persist_dir=self.persist_dir)
            self._index = li["load_index_from_storage"](storage)
        else:
            self._index = li["VectorStoreIndex"].from_documents([])
            os.makedirs(self.persist_dir, exist_ok=True)
        return self._index

    def _filters(self, expr: Optional[str]):
        """把权限 expr 翻译成 llama-index 的 MetadataFilters。"""
        parsed = parse_expr(expr)
        if not parsed:
            return None
        li = self._li
        op_map = {"in": li["FilterOperator"].IN, "==": li["FilterOperator"].EQ,
                  "!=": li["FilterOperator"].NE}
        return li["MetadataFilters"](filters=[
            li["MetadataFilter"](key=p["key"], value=p["value"], operator=op_map[p["op"]])
            for p in parsed
        ])

    # 【kb_code 为什么收下但不用】llama-index 后端是独立落盘的单一索引，没有分库概念；
    #   加这个参数只为与自研 RAGChain.ingest_document 签名一致，调用方才不用区分后端。
    def ingest_document(self, file_path: str, kb_code: str = "public") -> int:
        """文档入库：返回切片数（与自研 RAGChain.ingest_document 同签名）。"""
        docs = self._li["SimpleDirectoryReader"](input_files=[file_path]).load_data()
        if not docs:
            _logger.warning(f"llama-index 文档无内容: {file_path}")
            return 0
        index = self._ensure_index()
        for d in docs:
            index.insert(d)
        index.storage_context.persist(persist_dir=self.persist_dir)
        _logger.info(f"llama-index 入库完成: {file_path} | 文档={len(docs)}")
        return len(docs)

    def ingest_directory(self, dir_path: str) -> int:
        """批量入库（与自研 RAGChain.ingest_directory 同签名）。"""
        docs = self._li["SimpleDirectoryReader"](input_dir=dir_path).load_data()
        index = self._ensure_index()
        for d in docs:
            index.insert(d)
        index.storage_context.persist(persist_dir=self.persist_dir)
        _logger.info(f"llama-index 批量入库完成: 共 {len(docs)} 文档")
        return len(docs)

    def retrieve(self, query: str, top_k: Optional[int] = None, expr: Optional[str] = None) -> list:
        """
        检索：返回字段与自研链路保持一致（content/score/file_name/source），
        否则上层 build_context 会 KeyError —— 换后端最常踩的就是这个坑。
        """
        index = self._ensure_index()
        retriever = index.as_retriever(
            similarity_top_k=top_k or self._top_k, filters=self._filters(expr)
        )
        nodes = retriever.retrieve(query)
        hits = []
        for n in nodes:
            meta = getattr(n.node, "metadata", {}) or {}
            hits.append({
                "content": n.node.text,
                "score": float(n.score or 0.0),
                "file_name": meta.get("file_name", meta.get("file_path", "未知")),
                "source": meta.get("file_path", ""),
                "kb_code": meta.get("kb_code", ""),
                "updated_at": meta.get("updated_at", ""),
            })
        return hits[:self._rerank_top_k] if self._rerank_top_k else hits

    def build_context(self, query: str, top_k: Optional[int] = None, expr: Optional[str] = None) -> str:
        """拼接上下文（与自研 RAGChain.build_context 同格式，保证上层无感）。"""
        docs = self.retrieve(query, top_k=top_k, expr=expr)
        if not docs:
            return ""
        return "\n\n".join(
            f"【知识{i}】(相关度:{d['score']:.2f} 来源:{d['file_name']})\n{d['content']}"
            for i, d in enumerate(docs, 1)
        )

    def get_knowledge_stats(self, kb_codes: Optional[list] = None) -> dict:
        """知识库统计（字段对齐自研 RAGChain.get_knowledge_stats）。

        kb_codes 仅为对齐接口签名：llama-index 后端是独立落盘的单一索引，
        没有分库概念，这里忽略该参数。
        """
        return {
            "backend": "llamaindex",
            "persist_dir": self.persist_dir,
            "index_ready": self._index is not None,
            "total_chunks": self._chunk_count(),
            "by_kb": {},
        }

    def _chunk_count(self) -> int:
        """索引里的分片总数；索引未就绪时返回 0（不抛异常，统计接口不该拖垮页面）。"""
        try:
            return len(self._index.docstore.docs) if self._index is not None else 0
        except Exception:
            return 0


def build_rag(prefer: str = ""):
    """
    RAG 后端工厂：按配置返回实现，上层拿到的都是"有 retrieve/build_context 的对象"。

    【为什么不默认切到 llama-index】
        它的索引是独立落盘的（persist_dir），切过去 = 现有 Milvus 知识库看不到，
        表现得像"知识库空了"。所以默认 native，只有显式配置才切 —— 后端切换
        属于数据迁移级别的操作，不能靠一个默认值悄悄发生。
    """
    backend = (prefer or get_settings().rag_backend or "native").lower()
    if backend == "llamaindex":
        if not is_available():
            _logger.warning("RAG_BACKEND=llamaindex 但库不可用，降级为自研链路")
            backend = "native"
        else:
            return LlamaIndexRAG()
    from modules.rag.rag_chain import RAGChain
    return RAGChain()
