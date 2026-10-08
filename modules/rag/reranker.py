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
from typing import List, Optional                           # 类型标注：List 列表 / Optional 可选参数
from modules.common.logger import get_logger                # 统一日志工厂
from modules.common.exceptions import handle_exception      # 统一异常兜底装饰器：精排挂了记日志返回默认值，不炸问答主链路

_logger = get_logger("reranker")                            # 本模块日志器，命名 reranker，便于日志定位来源


# 【什么】BGE 重排器
# 【作用】对向量召回的候选文档做二次精排，把"看起来相关"筛成"真正相关"
# 【为什么】向量召回是粗筛（快但精度一般）；精排用更强的 cross-encoder 模型对 (query, doc)
#   逐对打分。模型不可用时自动降级为关键词加权，保证链路不崩（详见模块 docstring 的两路径对比）
class BGEReranker:
    """
    BGE 重排器：对召回结果二次精排。
    优先使用本地 BGE-Reranker 模型，模型不可用时降级为关键词+相似度加权精排。
    """

    _model = None                                           # 类属性：模型全进程共享一份，避免多实例重复加载占内存/显存

    # 【什么】构造器：定模型名并立刻加载
    # 【作用】初始化即完成"模型可用 / 降级"的判定
    # 【为什么】构造时加载而非首次调用时加载：启动期就暴露模型问题，第一个用户请求不会被加载卡顿拖累
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):  # 默认用 BGE 重排 base 版
        self._model_name = model_name                       # 记下模型名（加载与日志里用）
        self._init_model()                                  # 立即尝试加载模型，失败自动降级

    # 【什么】模型加载
    # 【作用】尝试加载 FlagEmbedding 的 FlagReranker；任何失败都降级为 None
    # 【为什么】FlagEmbedding 是可选增强依赖：没装 / 加载失败不该让服务起不来，降级路径顶上
    def _init_model(self):
        """尝试加载本地 BGE-Reranker 模型，失败则降级。"""
        try:
            from FlagEmbedding import FlagReranker   # type: ignore   # 可选依赖，未安装由下面 except 兜底降级
            self._model = FlagReranker(self._model_name, use_fp16=True)  # 加载模型；use_fp16 半精度：省一半显存、打分更快，精度损失可忽略
            _logger.info(f"BGE-Reranker加载成功: {self._model_name}")  # 成功日志：确认走了高精度路径
        except ImportError:                                 # 分支 1：FlagEmbedding 压根没装
            _logger.warning("FlagEmbedding 未安装，降级为关键词加权精排（pip install FlagEmbedding 启用BGE精排）")  # 警告里附安装方法
            self._model = None                              # 置 None = rerank 时走降级分支
        except Exception as e:                              # 分支 2：装了但加载失败（模型没下载完 / 显存不足等）
            _logger.warning(f"BGE-Reranker加载失败，降级精排: {e}")  # 带原因警告，便于排查
            self._model = None                              # 同样走降级分支

    # 【什么】精排入口
    # 【作用】按模型是否可用分派到两条精排路径，返回 top_k 篇最相关文档
    # 【为什么】@handle_exception(default=None)：精排整体异常时返回 None 而不是抛错——
    #   上游 rag_chain.retrieve 拿到 None 会回退为空列表，问答继续（无知识可用）而不是 500
    @handle_exception(default=None, log_msg="BGE精排异常")  # 装饰器：异常时记日志并返回 None
    def rerank(self, query: str, documents: List[dict], top_k: Optional[int] = None) -> List[dict]:
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
        if not documents:                                   # 空候选直接短路
            return []                                       # 没东西可排，返回空列表
        k = top_k or len(documents)                         # 没传 top_k 就全量返回（不截断）

        # 分派设计：模型在不在两条路，调用方完全无感——同一个 rerank 接口，精度自动升降级
        if self._model is not None:                         # 模型可用
            return self._rerank_with_model(query, documents, k)  # 走 BGE 模型精排
        else:                                               # 模型不可用
            return self._rerank_fallback(query, documents, k)    # 走关键词加权降级精排

    # 【什么】模型精排路径
    # 【作用】用 cross-encoder 对 (query, content) 逐对打分，按分降序取 top_k
    # 【为什么】cross-encoder 把 query 和 doc 拼在一起过模型，能捕捉双向语义交互，
    #   比"各自编码再算余弦"的双塔召回准得多——这就是精排这一步存在的意义
    def _rerank_with_model(self, query: str, documents: List[dict], top_k: int) -> List[dict]:
        """使用 BGE-Reranker 模型精排。"""
        pairs = [[query, doc["content"]] for doc in documents]     # 组装 (query, doc) 对：FlagReranker 要求的输入格式
        scores = self._model.compute_score(pairs, max_length=512)  # 批量打分；max_length=512 截断超长文档（超长部分对相关性贡献递减，拿它换速度）
        if not isinstance(scores, list):                    # 只传 1 对时 compute_score 返回单个 float 而非列表
            scores = [scores]                               # 统一包成列表，下面 zip 才不会错位
        for doc, score in zip(documents, scores):           # 把分数逐个写回对应文档
            doc["rerank_score"] = float(score)              # 新增 rerank_score 字段（与向量召回的 score 区分开）
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)  # 按精排分降序重排
        result = documents[:top_k]                          # 截取前 top_k 篇
        _logger.info(f"BGE模型精排: 输入={len(documents)} -> 输出={len(result)}")  # 记录漏斗：进了几篇、留了几篇
        return result                                       # 返回精排结果

    # 【什么】降级精排路径
    # 【作用】没有模型时用"向量相似度 * 0.7 + 关键词命中率 * 0.3"合成打分排序
    # 【为什么】向量分占大头（它才是语义相关性的主证据），关键词命中做补充修正；
    #   纯关键词会把"堆词不相关"的文档顶上来，纯向量分又等于没精排——加权是降级下的折中
    def _rerank_fallback(self, query: str, documents: List[dict], top_k: int) -> List[dict]:
        """降级精排：向量相似度 + 关键词命中加权。"""
        query_terms = set(query.lower().split())            # 问句按空格分词、转小写、去重成词集合（对英文/数字词有效；中文整句会当一个词——降级方案的已知短板）
        for doc in documents:                               # 逐文档计算加权分
            content = doc.get("content", "").lower()        # 正文转小写，与问句同口径比较
            term_hits = sum(1 for t in query_terms if t in content)  # 数问句里有几个词出现在正文中
            term_score = term_hits / max(len(query_terms), 1)        # 命中率 = 命中数/总词数；max(...,1) 防空问句除零
            vec_score = doc.get("score", 0.5)               # 取向量召回分；缺省 0.5（关键词降级通道来的结果分数被统一置成阈值，见 vector_store.keyword_search）
            doc["rerank_score"] = round(vec_score * 0.7 + term_score * 0.3, 4)  # 加权合成精排分，保留 4 位小数
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)  # 按合成分降序重排
        result = documents[:top_k]                          # 截取前 top_k 篇
        _logger.info(f"关键词降级精排: 输入={len(documents)} -> 输出={len(result)}")  # 记录降级路径的漏斗
        return result                                       # 返回精排结果
