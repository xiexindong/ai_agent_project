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
import time
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from modules.common.logger import get_logger
from modules.common.exceptions import LLMException, handle_exception
from config import get_settings

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

    # ------------------------------------------------------------------
    # 【这个装饰器在干什么】给下面的 API 调用自动套一层「失败重试」
    #   被装饰后，_embed_batch() 调用失败不会立刻抛错，tenacity 会自动再调几次。
    #
    # 【为什么需要重试】
    #   调智谱 API 是网络请求，失败分两类：
    #     瞬时故障（值得重试）：网络抖动、429 限流、502/503 服务端抖动
    #       —— 这类错误隔一两秒再试大概率就成功了
    #     永久故障（重试无用）：API Key 无效、参数非法、单条文本超长
    #       —— 这类重试 100 次也一样失败，只是白等、白耗配额
    #   重试机制只对第一类有效，这是它的能力边界。
    #
    # 【次数是怎么被限制的】靠 stop=stop_after_attempt(max_attempt_number=3)
    #   tenacity 的机制：每次调用失败后，先问一遍所有 stop 条件「要不要停」，
    #   任意一个返回 True 就停止重试并把异常抛出去。
    #   参数名 max_attempt_number 直译就是「最多尝试次数」，注意最大的坑：
    #   它是「总共尝试 3 次」，不是「失败后再重试 3 次」。
    #       max_attempt_number   1         2         3
    #       含义                 首次      重试1     重试2（末次，失败即抛错）
    #   所以实际只会重试 2 次。写成 3 却以为调用 4 次，是常见误解。
    #
    # 【间隔是怎么被限制的】靠 wait=wait_exponential(min=1, max=10) 指数退避
    #   公式：等待秒数 = clamp(multiplier × exp_base^(attempt_number - 1), min, max)
    #   tenacity 里 multiplier 默认 1、exp_base 默认 2，所以本配置的实际时序是：
    #       第 1 次失败后等 1 秒（2^0 = 1）再发起第 2 次
    #       第 2 次失败后等 2 秒（2^1 = 2）再发起第 3 次
    #       第 3 次失败 → 已到次数上限，不再等待，直接抛错
    #   即整个过程最多等待 3 秒。用递增间隔而不是固定间隔，是为了避开对方
    #   服务端的拥塞窗口：你越频繁重试，越容易被继续限流。
    #
    # 【察觉不足 1】max=10 在当前配置下是「死参数」，永远不会生效
    #   触发 max 封顶需要等待秒数 ≥ 10，即 2^(n-1) ≥ 10，n ≥ 5。
    #   而这里总共只尝试 3 次，最大等待只到 2 秒，永远够不到 10 秒封顶线。
    #   若哪天把次数调到 5 次以上，max=10 才开始起作用。
    #
    # 【察觉不足 2】没有限定异常类型，永久故障也会白重试
    #   这里没写 retry= 参数，tenacity 默认对所有 Exception 生效。
    #   于是「API Key 错误」这类永久故障同样会被重试 2 次，白等 3 秒。
    #   更精确的写法是 retry=retry_if_exception_type((TimeoutError, RateLimitError))
    #   之类，把重试范围收窄到瞬时故障。
    #
    # 【察觉不足 3】重试耗尽后抛的是 RetryError，不是原始异常
    #   tenacity 默认 reraise=False，会把最后一次的真实异常包进 RetryError 里抛出。
    #   上层 embed() 里 `except Exception as e: str(e)` 拿到的是
    #   "<RetryError [<Future at 0x... raised Timeout>]>" 这种几乎没信息量的字符串，
    #   真正的报错原因被藏住了。要拿到原始异常需 e.last_attempt.exception()，
    #   或在装饰器上加 reraise=True 直接抛出原始异常。
    # ------------------------------------------------------------------
    @retry(stop=stop_after_attempt(max_attempt_number=3), wait=wait_exponential(min=1, max=10))
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        调用智谱API批量生成向量（单次最多64条）。

        【在整条链上的位置】它是全项目唯一真正发网络请求算向量的地方：
            embed(text)         —— 单条文本，记忆系统存取记忆向量用
            embed_batch(texts)  —— 自动按 64 条切分，RAG 文档入库用
                    这两个上层只负责整理参数，最终都调到这里
                    ↓
            _embed_batch(batch) ← 唯一碰网络的一层，上面那个 @retry 就挂在它身上
                    ↓
                智谱 embedding-2 API
            好处是「单一出口」：重试、日志、异常处理只需在这一处做，改一处全局生效。
        """
        s = get_settings()

        # 【举个最简单的例子，一眼看懂这个函数在干什么】
        #   输入 texts（3 条文本）：
        #       ["今天天气不错", "明天要下雨", "周末去爬山"]
        #   输出（3 条向量，与输入一一对应）：
        #       [
        #         [ 0.023, -0.011,  0.087, ... 共 1024 个数],   ← 第 1 条文本的「坐标」
        #         [ 0.015,  0.042, -0.063, ... 共 1024 个数],   ← 第 2 条文本的「坐标」
        #         [ 0.031,  0.008,  0.052, ... 共 1024 个数],   ← 第 3 条文本的「坐标」
        #       ]
        #   一句话概括：把「文字」批量翻译成「数字坐标」。
        #   为什么坐标有用？因为意思相近的句子，算出来的坐标也相近 —— 之后只要比较
        #   坐标距离，就能找出「意思最接近」的那几条，这正是向量检索（pgvector）的原理。
        #   条数规则（要分清是哪一层）：直接调本函数时，传 3 条 ✓、传 64 条 ✓、
        #   传 65 条 ✗（API 直接报错）；但走上层 embed_batch() 就不用管，
        #   它会自动切成 64 + 1 两批 —— 实测传 65 条会调用 API 2 次、返回 65 条向量。
        #
        # 一次把整批文本发过去，而不是循环发单条：
        # 64 条一次请求 vs 发 64 次请求，网络往返从 64 次降到 1 次。
        # 注意「单次最多 64 条」是智谱 API 的硬限制，本函数只写进了文档、
        # 没有代码校验，所以切分责任实际落在上层 embed_batch() 身上。
        #
     
        resp = self._client.embeddings.create(
            model=self._model,   # 智谱 embedding-2
            input=texts,
        )
        # 拆包：SDK 返回的是一串对象，每项形如
        #   {"object": "embedding", "index": 0, "embedding": [0.023, -0.011, ... 共1024个]}
        # 这里剥成裸数值，让上层完全不依赖 zhipuai 的类型，换供应商只改这个文件。
        #
        # 【察觉不足：两处静默错位风险】
        #   1. 未校验返回条数：API 若只回 3 条而输入 5 条，这里不报错，
        #      上层 results.extend(vectors) 照常执行 —— 向量数少于文本数，
        #      之后全部错位，且全程无异常。
        #   2. 未按 index 排序：每个 item 自带的 index 字段本就是用来对齐顺序的，
        #      当前直接依赖返回顺序，上游一旦打乱即为静默错位。
        #      严谨写法：sorted(resp.data, key=lambda x: x.index)
        # 向量检索里「顺序即语义」，错位等于数据损坏却不抛异常，
        # 属于「静默全失败」的同类问题。
        return [item.embedding for item in resp.data]

    def embed(self, text: str) -> List[float]:
        """生成单条文本的向量。"""
        if not text or not text.strip():
            # 意思是：把 [0.0] 这个列表，重复 5 次
            #result = [0.0] * 5
            # 运行结果：[0.0, 0.0, 0.0, 0.0, 0.0]
            
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
                results.extend([[0.0] * self._dim] * len(batch))
            if i + batch_size < len(texts):
                time.sleep(0.3)
        return results

    @property
    def dimension(self) -> int:
        return self._dim


def get_embedding_service() -> EmbeddingService:
    """获取 Embedding 服务单例。"""
    return EmbeddingService()
