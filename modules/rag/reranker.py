"""
BGE 精排模块
============
对向量召回结果做二次重排，过滤低相关文档，提升问答精度。

【为什么要"召回+精排"两步？】类比搜索引擎：
    召回（Milvus 向量检索）：从 10 万文档里粗筛 50 篇"看起来相关"的，速度快但精度一般
    精排（BGE-Reranker）：对这 50 篇用更强的模型重新打分，留下 top 3~5 篇真正最相关的
    两步走的原因：精排模型贵且慢，不能直接对全库精排；召回模型便宜快，但精度差。
    类比前端：召回像 ElasticSearch 粗匹配，精排像人工打分排序。

【两种精排路径对比】
    ✓ BGE-Reranker 模型可用（FlagEmbedding 已装、模型下载完）：
        用 cross-encoder 模型对 (query, doc) 对打分，精度高
    ✗ 模型不可用（包没装/下载失败/显存不足）：
        降级为"向量相似度 * 0.7 + 关键词命中 * 0.3"加权打分，精度差但不崩
"""
from typing import List
from modules.common.logger import get_logger
from modules.common.exceptions import handle_exception

_logger = get_logger("reranker")


class BGEReranker:
    """
    BGE 重排器：对召回结果二次精排。
    优先使用本地 BGE-Reranker 模型，模型不可用时降级为关键词+相似度加权精排。
    """

    _model = None

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self._model_name = model_name
        self._init_model()

    def _init_model(self):
        """尝试加载本地 BGE-Reranker 模型，失败则降级。"""
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(self._model_name, use_fp16=True)
            _logger.info(f"BGE-Reranker加载成功: {self._model_name}")
        except ImportError:
            _logger.warning("FlagEmbedding 未安装，降级为关键词加权精排（pip install FlagEmbedding 启用BGE精排）")
            self._model = None
        except Exception as e:
            _logger.warning(f"BGE-Reranker加载失败，降级精排: {e}")
            self._model = None

    @handle_exception(default=None, log_msg="BGE精排异常")
    def rerank(self, query: str, documents: List[dict], top_k: int = None) -> List[dict]:
        """
        对召回文档二次精排，返回 top_k 篇最相关文档。

        【精排前后的排序变化对比】
            输入（Milvus 召回 5 篇，按向量相似度排）：
                [
                    {"content":"Python 列表用法", "score":0.85},
                    {"content":"Java 数组操作",    "score":0.78},
                    {"content":"Go 切片原理",     "score":0.72},
                    ...
                ]

            精排后（BGE cross-encoder 重打分，按 rerank_score 排，取 top 3）：
                [
                    {"content":"Python 列表用法", "rerank_score":0.94},  # 留下
                    {"content":"Go 切片原理",     "rerank_score":0.61},  # 留下（顺序变了！）
                    {"content":"Java 数组操作",   "rerank_score":0.42},  # 留下
                ]
                # 原本第3的 Go 升到第2，因 BGE 判定它与 query 更相关；Java 降到第3

        Args:
            query:     用户原问句（用于和每个 doc 配对打分）
            documents: Milvus 召回的文档列表，每项需含 "content" 字段
            top_k:     返回前几篇；None=不截断返回全部

        Returns:
            排好序的文档列表，每项新增 "rerank_score" 字段
        """
        if not documents:
            return []
        k = top_k or len(documents)

        if self._model is not None:
            return self._rerank_with_model(query, documents, k)
        else:
            return self._rerank_fallback(query, documents, k)

    def _rerank_with_model(self, query: str, documents: List[dict], top_k: int) -> List[dict]:
        """使用 BGE-Reranker 模型精排。"""
        pairs = [[query, doc["content"]] for doc in documents]
        scores = self._model.compute_score(pairs, max_length=512)
        if not isinstance(scores, list):
            scores = [scores]
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        result = documents[:top_k]
        _logger.info(f"BGE模型精排: 输入={len(documents)} -> 输出={len(result)}")
        return result

    def _rerank_fallback(self, query: str, documents: List[dict], top_k: int) -> List[dict]:
        """降级精排：向量相似度 + 关键词命中加权。"""
        query_terms = set(query.lower().split())
        for doc in documents:
            content = doc.get("content", "").lower()
            term_hits = sum(1 for t in query_terms if t in content)
            term_score = term_hits / max(len(query_terms), 1)
            vec_score = doc.get("score", 0.5)
            doc["rerank_score"] = round(vec_score * 0.7 + term_score * 0.3, 4)
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        result = documents[:top_k]
        _logger.info(f"关键词降级精排: 输入={len(documents)} -> 输出={len(result)}")
        return result
