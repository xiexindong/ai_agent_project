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
# 【什么】标准库与项目内依赖
# 【作用】time 控批次间隔、tenacity 提供重试装饰器、LLMException 抛向量错误、get_settings 读模型与维度
import time                                                   # 批间停顿用：sleep(0.3) 避免打满智谱 QPS 被限流
from typing import List                                       # 类型标注：文本列表 / 向量列表
from tenacity import retry, stop_after_attempt, wait_exponential  # 重试三件套：装饰器 + 停止条件 + 指数退避间隔
from modules.common.logger import get_logger                  # 统一日志工厂，产出带 "embedding" 前缀的 logger
from modules.common.exceptions import LLMException, handle_exception  # LLM 层异常类型 + 统一兜底装饰器（本文件只用前者）
from config import get_settings                               # 全局配置：API Key / 模型名 / 向量维度都从它读

_logger = get_logger("embedding")                   # 模块级日志器，前缀 embedding


# 【什么】智谱 embedding-2 的向量生成服务（单例）
# 【作用】全项目唯一的文字转向量出口，RAG 入库与记忆召回共用同一份客户端
# 【为什么】SDK 客户端与 API Key 是进程级资源，单例避免每个调用方各建一份连接与配置
class EmbeddingService:
    """智谱 embedding-2 向量生成服务（1024维）。"""

    _instance = None                                # 类级单例对象；None 表示还没创建过
    _client = None                                  # zhipuai SDK 客户端，进程内只建一次

    # 【什么】控制实例只创建一次
    # 【作用】反复调用 EmbeddingService() 拿到同一个对象，避免重复建客户端
    # 【为什么】单例的"唯一性"在 __new__ 层保证：对象只分配一次，后续调用直接返回旧实例
    def __new__(cls):
        if cls._instance is None:                             # 还没建过实例才走创建分支
            cls._instance = super().__new__(cls)              # 调父类 object.__new__ 真正分配对象
        return cls._instance                                  # 已建过则直接返回旧实例，保证全局唯一

    # 【什么】初始化 SDK 客户端并固化模型名与维度
    # 【作用】_client 已存在就跳过，保证单例下连接只建一次；失败即抛，不做降级
    # 【为什么】没有 API Key 或 SDK 缺失时装出"能用"的样子，会在首次调用时才爆雷——启动期失败更易定位
    def __init__(self):
        if self._client is None:                              # 客户端已建过就整段跳过（单例下 __init__ 会反复跑）
            s = get_settings()                                # 读全局配置：API Key / 模型名 / 维度
            try:
                from zhipuai import ZhipuAI                   # 延迟导入 SDK：不用向量功能时不付加载代价
                self._client = ZhipuAI(api_key=s.zhipu_api_key)     # zhipuai SDK 客户端
                self._model = s.embedding_model                     # 模型名，默认 embedding-2
                self._dim = s.embedding_dimension                   # 向量维度，默认 1024
                _logger.info(f"Embedding服务初始化: model={self._model}, dim={self._dim}")  # 启动日志：用的哪个模型多少维
            except Exception as e:
                raise LLMException("Embedding服务初始化失败", str(e))  # 包装成 LLM 层异常向上抛，启动即失败

    # 【什么】唯一真正发网络请求算向量的一层
    # 【作用】单一出口：重试、日志、异常处理只在这处做，上层的 embed / embed_batch 只整理参数
    # 【为什么】所有网络调用收敛到一处：换供应商、改重试策略、加监控都只改这一个函数
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
        s = get_settings()                                    # 读全局配置（当前函数体未直接用，保留统一取数习惯）

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

        # 【设计意图】整批一次请求而非逐条循环：把网络往返从 N 次降到 1 次，吞吐量主导因素从延迟变成带宽
        resp = self._client.embeddings.create(
            model=self._model,   # 智谱 embedding-2
            input=texts,                                    # 整批文本列表，与返回向量按顺序一一对应
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
        return [item.embedding for item in resp.data]         # 剥掉 SDK 对象壳，只留 1024 维数值列表

    # 【什么】单条文本转向量
    # 【作用】记忆系统存取一条记忆向量时用；空文本直接返回全零向量，省一次 API 调用
    # 【为什么】空文本调 API 是浪费钱且结果无意义；全零向量与任何向量的相似度都低，天然被阈值滤掉
    def embed(self, text: str) -> List[float]:
        """生成单条文本的向量。"""
        if not text or not text.strip():                      # None / 空串 / 纯空白三种情况都按空文本处理
            # 意思是：把 [0.0] 这个列表，重复 5 次
            #result = [0.0] * 5
            # 运行结果：[0.0, 0.0, 0.0, 0.0, 0.0]

            return [0.0] * self._dim    # 全零向量，长度 = _dim（默认 1024）；不调 API
        try:
            vectors = self._embed_batch([text.strip()])       # 去首尾空白后包成单元素列表，复用批量通道
            return vectors[0]                                 # 批量返回里取第 0 条，还原成单向量
        except Exception as e:
            _logger.error(f"向量生成失败: {e}")               # 先落错误日志，保留原始异常信息
            raise LLMException("向量生成失败", str(e))        # 包装成 LLM 层异常向上抛，由调用方决定降级策略

    # 【什么】批量文本转向量，自动按 64 条切批
    # 【作用】RAG 文档入库用；单批失败只把该批填成全零向量，不拖垮其它批次
    # 【为什么】大批入库时某一批网络故障不该让整个任务重来——坏批填零向量占位，好批照常入库
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成向量，自动分批（智谱单次限制64条）。"""
        results = []                                          # 收集所有批次的向量，顺序与输入 texts 一一对应
        batch_size = 64                 # 智谱 API 单批硬上限 64 条；这里是切批步长
        for i in range(0, len(texts), batch_size):            # 按 64 条为步长切片：0-63、64-127、...
            batch = texts[i:i + batch_size]                   # 取出当前批（最后一批可能不足 64 条）
            try:
                vectors = self._embed_batch(batch)            # 调唯一网络出口（自带 3 次重试）
                results.extend(vectors)                       # 成功批次的向量按序并入结果
                _logger.debug(f"批量向量生成: batch {i//batch_size+1}, size={len(batch)}")  # debug 级进度，常规不刷屏
            except Exception as e:
                _logger.error(f"批次{i//batch_size+1}生成失败: {e}")  # error 级：哪一批坏了要能被监控看见
                results.extend([[0.0] * self._dim] * len(batch))   # 坏批填等长全零向量，保证结果条数与输入对齐
            # 批间停顿 0.3 秒：仅当还有下一批时才睡，避免打满智谱 QPS 被限流
            if i + batch_size < len(texts):                   # 当前批不是最后一批才需要停顿
                time.sleep(0.3)                               # 限流保护：0.3 秒间隔把 QPS 压在安全线内
        return results                                        # 返回全部向量；坏批位置是全零向量，会被相似度阈值天然滤掉

    # 【什么】对外暴露向量维度
    # 【作用】建表定 vector(1024)、造全零向量都读它，避免维度被写死在多处
    # 【为什么】维度是跨模块契约（PG 建表、记忆管理、本服务三处共用），单一出口改一处全局生效
    @property
    def dimension(self) -> int:
        return self._dim                                      # 只读属性：外部不能改，防止维度被运行期篡改


# 【什么】获取向量服务的全局单例
# 【作用】统一入口，调用方不必自己 new，全进程共用一个客户端
# 【为什么】把"是单例"这个事实从调用方代码里抹掉——调用方只关心"拿到服务"，不关心实例管理
def get_embedding_service() -> EmbeddingService:
    """获取 Embedding 服务单例。"""
    return EmbeddingService()                                 # 触发 __new__ 单例逻辑：首次建、后续返回同一对象
