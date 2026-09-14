"""
Embedding 向量服务
==================
基于智谱 embedding-2（1024维），统一提供文档向量、摘要向量生成。
RAG 向量入库与 PG 记忆向量共用此服务。

【小白导读】这个模块只负责一件事：把文字变成向量（embedding）。
可以把 embedding 理解成「把一句话翻译成一串坐标」——意思相近的句子，
算出来的坐标也相近；之后程序就能靠「比较坐标距离」做语义搜索，
类似在地图上找离你最近的几家店（这正是 pgvector 向量检索的原理）。
JS 类比：相当于项目里统一封装第三方接口的 axios service 模块。
上下游：上游是 RAG 文档入库和记忆系统（MemoryManager），下游是智谱 embedding-2 API。
"""
# 标准库：time.sleep 用于在批量请求之间停顿，避免把第三方 API 打到限频
import time
# typing.List 是类型标注，JS 里没有：相当于注明「这是数组，元素是字符串/浮点数」
from typing import List
# tenacity：第三方重试库。装饰器一加，函数失败自动重试，类似 JS 里给 fetch 包一层 retry
from tenacity import retry, stop_after_attempt, wait_exponential
from modules.common.logger import get_logger
from modules.common.exceptions import LLMException, handle_exception
from config import get_settings

# 模块级 logger：所有日志统一走它，类似 JS 的 const logger = createLogger('embedding')
_logger = get_logger("embedding")


class EmbeddingService:
    """智谱 embedding-2 向量生成服务（1024维）。"""

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            s = get_settings()
            try:
                from zhipuai import ZhipuAI
                self._client = ZhipuAI(api_key=s.zhipu_api_key)
                self._model = s.embedding_model
                self._dim = s.embedding_dimension
                _logger.info(f"Embedding服务初始化: model={self._model}, dim={self._dim}")
            except Exception as e:
                raise LLMException("Embedding服务初始化失败", str(e))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """调用智谱API批量生成向量（单次最多64条）。"""
        s = get_settings()
        resp = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def embed(self, text: str) -> List[float]:
        """生成单条文本的向量。"""
        if not text or not text.strip():
            return [0.0] * self._dim
        try:
            vectors = self._embed_batch([text.strip()])
            return vectors[0]
        except Exception as e:
            _logger.error(f"向量生成失败: {e}")
            raise LLMException("向量生成失败", str(e))

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成向量，自动分批（智谱单次限制64条）。"""
        results = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                vectors = self._embed_batch(batch)
                results.extend(vectors)
                _logger.debug(f"批量向量生成: batch {i//batch_size+1}, size={len(batch)}")
            except Exception as e:
                _logger.error(f"批次{i//batch_size+1}生成失败: {e}")
                # 兜底：用零向量填充，不中断流程
                results.extend([[0.0] * self._dim] * len(batch))
            # 避免限频
            if i + batch_size < len(texts):
                time.sleep(0.3)
        return results

    @property
    def dimension(self) -> int:
        return self._dim


def get_embedding_service() -> EmbeddingService:
    """获取 Embedding 服务单例。"""
    return EmbeddingService()
