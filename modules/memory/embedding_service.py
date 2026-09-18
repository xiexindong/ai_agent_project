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
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 【是的 —— 这里就是在调用「向量模型」（Embedding 模型）】
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 它【不是】对话模型，不会"回答"任何问题，只做一件事：文字 → 一串数字。
        #
        #   对话模型（本项目是 DeepSeek，见 modules/agent/llm.py）
        #       llm.invoke("12*8等于多少")  →  "12×8 = 96"      （生成文字）
        #
        #   向量模型（本方法，智谱 embedding-2）
        #       embed("12*8等于多少")       →  [0.12, -0.34, ...] （生成 1024 个小数）
        #
        # 【逐行拆解】
        #   self._client  → 第 44 行创建的 ZhipuAI 客户端（注意：不是 DeepSeek 的 OpenAI 客户端）
        #   .embeddings.create()  → 智谱 SDK 的「向量接口」
        #       对照：对话接口叫 chat.completions.create()，这里没有 chat，
        #             因为根本没有"对话"这回事，只是把文本丢进去换算成坐标
        #   model=self._model → 配置里的 EMBEDDING_MODEL（默认 "embedding-2"）
        #       注意和对话模型的 DEEPSEEK_MODEL 是两个不同的字段、两家不同的厂商
        #   input=texts  → 字符串数组（["摘要1", "摘要2", ...]），
        #       不是 messages 那种 [{"role":"user",...}] 结构，因为不存在多轮对话
        #
        # 【输入输出对比】
        #   输入：texts = ["数学计算:12*8=96", "餐饮推荐"]
        #   返回：[[0.012, -0.34, ...共1024个], [0.88, 0.02, ...共1024个]]
        #        ↑ resp.data 的顺序和 input 一一对应，第 i 条文本对应第 i 个向量
        #
        # 【为什么用智谱而不用 DeepSeek？】
        #   DeepSeek 目前只提供对话/推理模型，没有开放 embedding 接口；
        #   所以本项目是「两家混用」：聊天用 DeepSeek，向量用智谱 embedding-2。
        #   这也是为什么数据库里向量维度是 1024（embedding-2 固定 1024 维，
        #   见 config.py 的 embedding_dimension，建表 vector(1024) 必须和它一致）。
        #
        # 【一个反直觉的点】
        #   同样的文本调两次，得到的 1024 个数字【完全一样】。
        #   向量模型是确定性的、无状态的——它不"思考"、不"发挥"，
        #   本质是个查表+计算的编码器，和我们平时说的"AI 生成"不是一回事。

        #就是把问句 转化为 向量  调用大模型
        resp = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        # resp.data 是对象数组，每个元素长这样：{"index":0, "embedding":[0.1,0.2,...]}
        # 用列表推导只把 embedding 字段抽出来，丢掉 index 等元信息
        #         resp.data = [
        #     { "embedding": [0.1, 0.2, 0.3], "index": 0 },  # 第1条文本的向量
        #     { "embedding": [0.4, 0.5, 0.6], "index": 1 },  # 第2条文本的向量
        # ]
        # 获取的结果
        # [
        #     [0.0023, -0.0093, 0.0156, ...],  # 第1段文本的向量（1536个数字）
        #     [0.0041, -0.0012, 0.0234, ...],  # 第2段文本的向量（1536个数字）
        # ]
        return [item.embedding for item in resp.data]

    def embed(self, text: str) -> List[float]:
        """生成单条文本的向量。"""
        # 空文本 / 纯空白 → 不发请求，直接返回一个「零向量」兜底。
        #
        # 【为什么要返回 [0.0] * self._dim，而不是 [] 或 None？】
        #   因为下游 pg_manager.save_history 会把它写进 pgvector 的 vector(1024) 列，
        #   维度/类型不对 INSERT 会直接报错。所以这里必须造一个「形状合法」的假向量，
        #   让整条链路（np.array → INSERT → 建 traces）照常走完，只是内容无意义。
        #
        # 【为什么用 0.0，而不是随机数？】
        #   零向量与任何向量做点积都得 0 → 余弦相似度必然低于 similarity_threshold
        #   （pgvector 那边度量为 NaN 也一样，Python 里 float('nan') >= 0.6 是 False）
        #   → 必定被 _recall_long_term 的阈值过滤掉。
        #   效果就是「存得进去，但永远不会被召回」——不报错、不污染记忆结果，这是安全降级。
        #
        # 【为什么不交给上面 @retry 重试？】
        #   智谱对空字符串必然返回 400，重试 3 次纯属浪费时间与额度；
        #   这里是「明知没必要发」，和下面 except（API 真的挂了）性质完全不同。
        if not text or not text.strip():
            return [0.0] * self._dim
        try:
            # 把单条文本包成「只有一个元素的列表」去调批处理接口，再把结果解开。
            # 本质是「借批量通道走单条」：
            #   ["数学计算:12*8=96"]  →  [[0.12, -0.34, ... 共1024个]]
            #                             ↑ 外面还套着一层列表
            #                               ↓ [0] 拆包
            #                             [0.12, -0.34, ... 共1024个]
            #
            # 【为什么不直接让接口接收单个字符串？】
            #   智谱 embeddings.create 的 input 参数只收 `list[str]`，不收裸字符串；
            #   而且统一走 _embed_batch 可以复用它的 @retry 重试、单次 64 条上限等逻辑，
            #   将来要做「批量生成向量」时上层直接调 _embed_batch 即可，无需另写一套。
            #
            # 【为什么又要 strip() 一次？】
            #   上面守卫里判断的只是「去空白后是否非空」，并没有改掉原值；
            #   这里再 strip 是为了喂给模型的是干净文本，首尾空格不参与编码。
            #
            # 【resp.data 的顺序契约】
            #   返回的第 0 个向量必定对应输入的第 0 条文本（API 保证一一对应），
            #   所以 vectors[0] 才是安全的取法，而不是按某个 id 去查。
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
