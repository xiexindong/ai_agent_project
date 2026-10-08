"""
Tool-RAG 模块
=============
把工具当成「可被检索的知识」：先用语义向量给每个工具建索引，
用户提问进来时按相似度只召回 top-k 个工具，而不是把全部工具塞给模型。

【解决什么问题：技能膨胀】
    工具数量 N 与单轮 prompt 成本是线性关系：
        全量暴露  → 每轮 N 份 JSON Schema 进 prompt，N=100 时模型要在 100 个候选里挑
        语义召回  → 每轮固定 k 份（默认 5），N 增长不影响单轮成本
    代价是「可能漏召」，所以本模块配了两道兜底：
        min_score 过滤（宁缺毋滥）+ 路由层默认工具回退（见 modules/agent/router.py）。

【索引里存什么 / 不存什么】
    存：工具名 + 描述 + 参数名（用途画像，用于算相似度）
    不存：完整 JSON Schema（那是调用契约，留在 ToolRegistry，按需取用）
    把调用契约也塞进向量库，等于把「怎么调」和「该不该调」混在一起，
    既浪费索引空间，也让召回结果里混入参数噪声。

【向量不可用时怎么办】
    embedding 服务无 key / 超时 / 报错，自动降级为关键词召回（2-gram 匹配），
    keyword 命中的分数按 0.6 折算 —— 弱证据不能盖过真实的语义命中。
"""
import re  # 正则模块：_fragments 里切英文词与中文字符用
from dataclasses import dataclass  # 数据类装饰器：ToolHit 用它自动生成 __init__ 等样板代码
from typing import Dict, List, Optional  # 类型注解：字典/列表/可选

from modules.common.logger import get_logger  # 统一日志工厂

_logger = get_logger("tool_rag")  # 模块级 logger：本文件日志统一带 "tool_rag" 前缀

# 关键词证据的折算系数：命中同样多的词，语义相似度的可信度更高
_KEYWORD_SCALE = 0.6  # 关键词命中分 ×0.6：弱证据打折，避免盖过真实的语义命中


# 【什么】一次工具召回的结果记录
# 【作用】带名字与分数返回给路由层，由它决定是否把工具放进 prompt
@dataclass  # 数据类装饰器：按下方字段声明自动生成 __init__/__repr__/__eq__
class ToolHit:  # 一次召回结果：名字 + 分数 + 召回通道
    """一次召回结果。

    Attributes:
        name:   工具名（与 ToolRegistry 中注册的名字一致，召回后靠它取实例）
        score:  相似度，向量命中为余弦值（0~1），关键词命中为折算后的比例分
        source: "vector"（语义召回）/ "keyword"（降级召回）
    """
    name: str        # 工具名，与 ToolRegistry 里注册的名字一致
    score: float     # 相似度：向量为余弦值(0~1)，关键词为折算后的比例分
    source: str      # 召回方式：vector（语义） / keyword（降级）


# 【什么】工具的语义检索索引
# 【作用】建索引 + 按提问召回 top-k 工具，让 prompt 里只有少量候选
class ToolRAGIndex:  # 工具语义索引类：负责建索引与按提问召回
    """工具语义索引：index_tools() 建索引，recall() 按提问召回。

    【三张内部表】
        _texts   工具名 → 用途画像文本（建向量与关键词匹配共用）
        _vectors 工具名 → 语义向量（embedding 可用时才有）
        _metas   工具名 → 原始元信息（来源 server / 描述，排查时用）
    """

    def __init__(self, top_k: int = 5, min_score: float = 0.25, enabled: bool = True):  # 构造：召回数量/分数门槛/总开关
        self._top_k = max(1, int(top_k))                  # 单次最多召回几个工具，默认 5，下限 1
        self._min_score = float(min_score)                # 召回分数门槛，默认 0.25，低于它一律不要
        self._enabled = enabled                           # 总开关：False 时 recall() 直接返回空列表
        self._texts: Dict[str, str] = {}                  # 工具名 → 用途画像文本
        self._vectors: Dict[str, List[float]] = {}        # 工具名 → 语义向量（embedding 可用时才有）
        self._metas: Dict[str, dict] = {}                 # 工具名 → 原始元信息（server 来源，排查用）
        self._vector_available = True                     # 向量是否可用；失败一次后永久降级关键词
        self._last_source = "vector"                      # 上次召回方式，供 stats() 观测

    # ------------------------------------------------------------------
    # 【建索引】工具可以是 BaseTool 实例，也可以是网关目录里的 dict
    #
    #   【为什么同名工具只索引一次】网关 refresh 后可能重复喂同一批工具，
    #   重复编码既浪费 embedding 调用（每次都要花钱），也会把索引撑大。
    #   文本发生变化时才重新编码（工具描述被改过的场景）。
    #
    #   【输入输出对照】
    #     输入  tools = [CalculatorTool(), {"name":"weather","description":"查询城市天气",
    #                                       "inputSchema":{"properties":{"city":{}}}}]
    #     输出  返回 2（新编码的工具数）
    #           _texts["calculator"] = "calculator。安全计算数学表达式 可填参数: expression、precision"
    #           _texts["weather"]    = "weather。查询城市天气 可填参数: city"
    # ------------------------------------------------------------------
    # 【什么】给工具建索引，返回新编码的工具数
    # 【作用】文本没变且已编码的工具跳过，避免重复调 embedding
    def index_tools(self, tools: list) -> int:  # 建/更新索引：返回本轮新编码的工具数
        # pending = 本轮真正需要编码的工具 [(名字, 画像文本), ...]；攒完再批量编码，
        #           一次 embed_batch 搞定，避免每个工具各调一次 embedding（省调用、省钱）
        pending: List[tuple] = []  # 待编码队列：[(工具名, 画像文本), ...]
        for tool in tools or []:  # 遍历传入工具（None 视为空列表，防崩）
            # 【归一化】BaseTool 实例与网关目录 dict 两种来源统一成 (name, desc, schema, meta)
            #   → _normalize() 见本文件 326 行
            name, desc, schema, meta = self._normalize(tool)  # 统一取出四元组
            if not name:                                    # 没有名字的工具无法召回（召回结果靠名字取实例），直接跳过
                continue  # 跳过无名工具
            # 【用途画像】"名字。描述 可填参数: a、b" —— 只描述"干什么"，不含 JSON Schema 细节，
            #   否则参数噪声会干扰相似度 → _routing_text() 见本文件 318 行
            text = self._routing_text(name, desc, schema)  # 拼出该工具的用途画像文本
            # 【是否已编码过】向量不可用时恒为 True（没有向量可比对，只靠文本判断）；
            #   向量可用时要求 _vectors 里已有它，否则补一次编码
            already_indexed = not self._vector_available or name in self._vectors  # 判定是否已建过索引
            if self._texts.get(name) == text and already_indexed:  # 画像文本没变且已编码
                continue                                    # 文本没变且已编码 → 跳过，网关 refresh 重喂同一批工具时不重复花钱
            self._texts[name] = text                        # 文本表：向量与关键词召回共用
            self._metas[name] = meta                        # 元信息表：记录来源 server，排查"这工具哪来的"用
            pending.append((name, text))                    # 文本变了或还没编码 → 加入待编码队列

        if not pending:  # 本轮没有需要新编码的工具
            return 0  # 直接返回 0，不调 embedding
        if self._vector_available:  # 向量通道还可用：尝试批量编码
            try:  # embedding 调用可能失败（无 key/超时），用 try 包住走降级
                from modules.memory.embedding_service import get_embedding_service  # 延迟 import：避免模块级依赖 embedding 服务
                # 一句话：把所有待编码的工具画像文本一次性批量转成向量，按入参顺序等长返回。
                vectors = get_embedding_service().embed_batch([t for _, t in pending])  # 批量编码：列表推导只取文本部分
                # ------------------------------------------------------------------
                # 【什么】zip(pending, vectors)：把两个等长序列像拉链一样一对一咬合
                #   每次循环吐出一对 (pending[i], vectors[i])，i 从 0 走到末尾：
                #     pending  —— 上面第 101 行攒的 [(工具名, 画像文本), ...]
                #     vectors  —— 上面第 108 行 embed_batch 批量编码的 [[float,...], ...]
                #
                # 【为什么敢直接配对】embed_batch 严格按入参顺序返回向量，
                #   第 i 个向量一定对应第 i 个工具，不需要再按名字回查一次。
                #
                # 【两层解包】(name, _), vec
                #   外层拆 zip 的一对：左边是 pending 元素，右边是对应向量
                #   内层拆 pending 元素：name = 工具名，_ = 画像文本（这里用不上，丢弃）
                #
                # 【输入输出对照】
                #   输入 pending  = [("web_search", "web_search。网络搜索..."),
                #                    ("get_current_time", "get_current_time。获取当前时间...")]
                #   输入 vectors  = [[0.012, -0.034, ...], [0.045, 0.021, ...]]
                #   输出 _vectors["web_search"]       = [0.012, -0.034, ...]
                #        _vectors["get_current_time"] = [0.045, 0.021, ...]
                #
                # 【依赖警告一】zip 以"更短的那个"为准截断：两边长度不一致时，
                #   多出来的尾部被直接丢掉，不报错也不告警（要严格校验得显式传 strict=True）
                # 【依赖警告二】zip 后面不要加类型下标写成 zip[...](...)。
                #   实测 Python 3.13 下 zip 无 __class_getitem__，会抛
                #   "TypeError: type 'zip' is not subscriptable"，而本行在 try 里，
                #   异常会被下面的 except 接住，表现是"向量不可用，降级为关键词召回"——
                #   看日志像是 embedding 服务挂了，实际是这一行写法的问题。
                # ------------------------------------------------------------------
                for (name, _), vec in zip(pending, vectors):  # 拉链配对：第 i 个工具对应第 i 个向量
                    self._vectors[name] = vec  # 把向量按工具名存进向量表
                _logger.info(f"Tool-RAG 已编码 {len(pending)} 个工具向量 | 索引总数={len(self._texts)}")  # 记编码成果日志
            except Exception as e:  # 捕获一切 embedding 异常（网络/鉴权/超时等）
                # 【降级而不是失败】向量只是召回手段之一，关键词照样能选出工具；
                #   这里若向上抛，整条对话就挂了 —— 召回不该成为可用性瓶颈。
                self._vector_available = False  # 永久关掉向量开关：后续轮次直接走关键词
                _logger.warning(f"Tool-RAG 向量不可用，降级为关键词召回: {e}")  # 记 warning 说明降级原因
        return len(pending)  # 返回本轮新编码的工具数（即便编码失败，文本/元信息也已入账）

    # ------------------------------------------------------------------
    # 【召回】返回按分数降序的 top-k
    #
    #   【为什么要有 min_score】top-k 是「取前 k 个」而不是「取好的 k 个」。
    #   工具库里根本没有相关工具时，前 k 个仍然是 k 个无关工具，
    #   它们一样会进 prompt 占 token、干扰模型选择。低于阈值的一律不要，
    #   召回为空由路由层回退默认工具兜底。
    #
    #   【输入输出对照】
    #     输入  query="今天北京天气怎么样"，top_k=5，min_score=0.25
    #           （索引里 32 个工具，含 weather / calculator / data_query / translate…）
    #     输出  [ToolHit("weather", 0.7412, "vector"), ToolHit("web_search", 0.5231, "vector")]
    #           其余 30 个工具相似度低于 0.25，不进结果
    # ------------------------------------------------------------------
    # 【什么】按提问召回最相关的若干工具
    # 【作用】向量优先、关键词兜底；无命中返回空列表交由路由层回退默认工具
    def recall(self, query: str, top_k: Optional[int] = None) -> List[ToolHit]:  # 按提问召回 top-k 工具，返回 ToolHit 列表
        # ------------------------------------------------------------------
        # 【前置短路：三种"没法召回"的情况直接给空】
        #   not self._enabled  总开关关了（配置关闭 Tool-RAG）
        #   not query          空字符串/None：没有意图可匹配
        #   not self._texts    索引里一个工具都没有：无候选可返回
        #   返回 [] 不是异常，上层路由层会回退默认工具（router.py:260）。
        # ------------------------------------------------------------------
        if not self._enabled or not query or not self._texts:  # 任一前置不满足
            return []  # 返回空列表（约定信号，上层据此回退默认工具）
        # ------------------------------------------------------------------
        # 【定本轮配额 k】显式传了 top_k 就用传入的，否则用索引默认值（5）。
        #   max(1, ...) 双保险：外部传 0/负数时至少召回 1 个，不会切成空名单。
        #   int() 兜底：配置可能被读成字符串（如 "5"）。
        # ------------------------------------------------------------------
        k = max(1, int(top_k or self._top_k))  # 本轮召回数量下限 1

        # ------------------------------------------------------------------
        # 【主策略：向量优先，关键词兜底】
        #   能走向量时先尝试语义召回——余弦相似度是强证据，比 2-gram 关键词准。
        # ------------------------------------------------------------------
        if self._vector_available:  # 向量通道开着：优先语义召回
            hits = self._recall_vector(query, k)  # 走向量召回
            # 语义召回到了 → 标记来源并直接返回，不再走关键词（避免弱证据混入）
            if hits:  # 语义有命中
                self._last_source = "vector"    # 记录实际生效通道，stats() 观测用
                return hits  # 直接返回语义结果
            # 语义通道跑通了但一个都没过门槛（hits 为 []）→ 降级关键词再试一次，
            #   而不是空手而归：库里也许只是向量相似度普遍偏低，关键词还能捞到。
            self._last_source = "keyword"  # 标记本次走了关键词通道
            return self._recall_keyword(query, k)  # 关键词兜底召回
        # 向量开关早已关闭（此前 embedding 失败过）→ 直接走关键词，不再尝试向量，
        #   省掉注定失败的 embedding 调用。
        self._last_source = "keyword"  # 标记本次走了关键词通道
        return self._recall_keyword(query, k)  # 直接关键词召回

    # 【什么】语义召回：编码 query 后与全部工具向量算余弦
    # 【作用】主力召回路径；编码失败就关掉开关交给关键词兜底
    def _recall_vector(self, query: str, k: int) -> List[ToolHit]:  # 语义召回内部实现：向量编码 + 余弦打分 + 过滤
        # ------------------------------------------------------------------
        # 【第 1 步：把提问变成向量】
        #   文本和文本没法直接算"像不像"，先调 embedding 把 query 编码成
        #   和工具向量同维度的 qv，之后才能两两算余弦。
        #   这里 embed() 是单条编码（工具侧批量编码用 embed_batch，见 117 行）。
        #
        # 【为什么失败要返回 [] 而不是抛异常】
        #   上层 recall()（172-206 行）的约定是：向量召回返回空 → 自动改走
        #   _recall_keyword。同时把 _vector_available 永久置 False：
        #   embedding 服务通常不是"抖一下"，而是无 key/配置错，再试只会重复失败。
        # ------------------------------------------------------------------
        try:  # embedding 调用可能失败，用 try 包住走降级
            from modules.memory.embedding_service import get_embedding_service  # 延迟 import：避免模块级依赖 embedding
            qv = get_embedding_service().embed(query)                  # qv = query vector，如 [0.021, -0.04, ...]（智谱 embedding 为 1024 维）
        except Exception as e:  # 编码失败：embedding 服务不可用
            self._vector_available = False                             # 关掉向量开关：本次后续轮次直接走关键词，不再反复撞 embedding
            _logger.warning(f"Tool-RAG 查询向量失败，降级关键词: {e}")  # 记 warning 说明降级原因
            return []                                                  # 空列表是约定信号，不是"没召回到工具"——上层据此切关键词通道
        # ------------------------------------------------------------------
        # 【第 2 步：给每个工具打分】
        #   遍历 _vectors 里全部工具向量，各和 qv 算一次余弦相似度（_cosine
        #   见本文件 364 行），结果包成 ToolHit：
        #     round(..., 4)  相似度保留 4 位小数，日志清爽、排序不受影响
        #     "vector"       来源标记：区分于关键词降级分（recall 合并时用）
        #
        # 【输入输出对照】query="今天北京天气怎么样"，假设索引 3 个工具：
        #   scored = [ToolHit("weather", 0.7412, "vector"),
        #             ToolHit("web_search", 0.5231, "vector"),
        #             ToolHit("calculator", 0.0318, "vector")]
        #   → 此时顺序还是字典序，分数高低还没排
        # ------------------------------------------------------------------
        scored = [  # 列表推导：给每个工具算相似度并包成 ToolHit
            ToolHit(name, round(_cosine(qv, vec), 4), "vector")  # 算余弦并保留 4 位小数，来源标记为 vector
            for name, vec in self._vectors.items()  # 遍历向量表里全部工具向量
        ]
        # ------------------------------------------------------------------
        # 【第 3 步：按分数降序】h.score 越大说明用途画像和提问越像，
        #   reverse=True 让最相关的排最前；同分时保持原相对顺序（排序稳定）。
        # ------------------------------------------------------------------
        scored.sort(key=lambda h: h.score, reverse=True)  # 按 score 降序排列（reverse=True 代表降序）
        # ------------------------------------------------------------------
        # 【第 4 步：先过门槛、再取前 k】两步顺序不能反：
        #   先 min_score 过滤 —— 工具库里没有相关工具时，前 k 个只是"最不
        #   差的 k 个无关工具"，宁缺毋滥（理由见 159-162 行总述）；
        #   再切片 [:k] —— 达标的可能不止 k 个，只带本轮配额内的数量。
        # ------------------------------------------------------------------
        hits = [h for h in scored if h.score >= self._min_score][:k]  # 先过滤低分，再切前 k
        _logger.debug(  # 记 debug 日志：只看前 5 名，防刷屏
            "Tool-RAG 语义召回: "
            + ", ".join(f"{h.name}={h.score}" for h in scored[:5])     # 只打印前 5 名：工具再多日志也不刷屏，够看排序对不对
            + f" | 命中={len(hits)}"                                   # "命中"= 过门槛且在 k 内的个数，0 时上层会切关键词
        )
        return hits                                                    # 可能为空列表（全都低于门槛），由 recall() 兜底走关键词或路由层默认工具

    # 【什么】关键词降级召回：按 2-gram 片段重合打分
    # 【作用】embedding 不可用时仍能选出工具；分数乘 _KEYWORD_SCALE=0.6 折算
    def _recall_keyword(self, query: str, k: int) -> List[ToolHit]:  # 关键词降级召回：2-gram 片段重合打分
        q_frags = self._fragments(query)  # 把提问切成片段（中文 2-gram + 英文词）
        if not q_frags:  # 提问切不出片段（比如全是标点）
            return []  # 直接返回空，没东西可比
        hits: List[ToolHit] = []  # 收集打分结果
        for name, text in self._texts.items():  # 遍历全部工具的画像文本
            t_frags = set(self._fragments(text))  # 工具画像也切成片段并去重成 set（O(1) 查找）
            hit = sum(1 for f in q_frags if f in t_frags)  # 数提问片段有几个命中工具片段
            if not hit:  # 一个片段都没中
                continue  # 跳过，不给分
            raw = min(1.0, hit / len(q_frags))  # 原始分 = 命中比例，封顶 1.0
            hits.append(ToolHit(name, round(raw * _KEYWORD_SCALE, 4), "keyword"))  # ×0.6 折算后包成 ToolHit
        hits.sort(key=lambda h: h.score, reverse=True)  # 按分数降序
        return [h for h in hits if h.score >= self._min_score][:k]  # 过门槛后切前 k

    # ------------------------------------------------------------------
    # 【分词】中文按 2-gram 切，英文/数字按词切
    #
    #   【为什么中文用 2-gram 而不是整词】没有分词器（jieba 等）时，
    #   2-gram 是成本最低的近似：既能表达"天气""报销"这类双字意图，
    #   又不需要额外依赖。
    #
    #   【输入输出对照】
    #     输入  "北京天气 3*7"
    #     输出  ['北京', '京天', '天气']
    #
    #   【实测】ToolRAGIndex._fragments("北京天气 3*7") → ['北京', '京天', '天气']
    #       英文/数字要求「连续 ≥2 位」（正则 [a-z0-9_]{2,}），
    #       "3" 与 "7" 被 * 隔开，各自只有 1 位，因此不产出片段；
    #       写成 "37" 才会得到 ['37']。
    # ------------------------------------------------------------------
    # 【什么】把文本切成可比对片段
    # 【作用】中文走 2-gram、英文数字走词，供关键词召回做重合统计
    @staticmethod  # 静态方法：不依赖实例状态
    def _fragments(text: str) -> List[str]:  # 文本切片器：产出可比对的小片段列表
        lower = (text or "").lower()  # 统一小写 + 防空（None 变空串）
        frags = [w for w in re.findall(r"[a-z0-9_]{2,}", lower)]  # 英文/数字片段：连续 ≥2 位的词
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lower))  # 抽出全部中文字符连成串
        frags.extend(chinese[i:i + 2] for i in range(max(0, len(chinese) - 1)))  # 中文按 2-gram 滑动窗口切片
        return frags  # 返回混合片段列表（英文词 + 中文双字片）

    # ------------------------------------------------------------------
    # 【用途画像】只取参数名，不取完整 schema
    #
    #   【输入输出对照】
    #     输入  name="data_query", desc="查询业务数据", schema={"properties":{"table":{},"limit":{}}}
    #     输出  "data_query。查询业务数据 可填参数: table、limit"
    # ------------------------------------------------------------------
    # 【什么】拼出工具的用途画像文本
    # 【作用】只取参数名（最多前 6 个），不把完整 schema 塞进索引
    @staticmethod  # 静态方法：纯拼接逻辑，不依赖实例
    def _routing_text(name: str, description: str, schema: dict) -> str:  # 生成用途画像文本
        props = (schema or {}).get("properties", {}) or {}  # 取 JSON Schema 的 properties 字段（三层防空）
        params = "、".join(list(props.keys())[:6]) or "无参数"  # 最多取前 6 个参数名顿号拼接，无参数时显示"无参数"
        return f"{name}。{description} 可填参数: {params}"  # 拼成 "名字。描述 可填参数: a、b" 的画像文本

    @staticmethod  # 静态方法：纯字段提取，不依赖实例
    # 【什么】把 dict 形态与 BaseTool 形态的工具归一化
    # 【作用】索引与网关目录两套来源共用一套取字段逻辑
    def _normalize(tool) -> tuple:  # 归一化：两种工具来源统一成 (name, desc, schema, meta) 四元组
        """把 BaseTool 实例或网关目录 dict 统一成 (name, description, schema, meta)。"""
        if isinstance(tool, dict):  # 网关目录形态：dict
            return (  # 按 dict 的 key 取字段
                str(tool.get("name", "")),  # 工具名，缺省空串
                str(tool.get("description", "")),  # 工具描述，缺省空串
                tool.get("inputSchema") or {},  # 输入 schema，缺省空 dict
                {"server": tool.get("server", "")},  # meta 记录来源 server
            )
        # BaseTool 家族：MCPToolAdapter 用 parameters_schema，本地工具用 parameters
        name = getattr(tool, "name", "")  # 实例形态：取 .name 属性
        desc = getattr(tool, "description", "")  # 取 .description 属性
        schema = getattr(tool, "parameters_schema", None) or getattr(tool, "parameters", None) or {}  # 兼容两种 schema 属性名
        return str(name), str(desc), schema, {"server": getattr(tool, "tool_type", "")}  # meta 里 server 记 tool_type

    # 【什么】输出索引运行态观测数据
    # 【作用】排查召回效果时看是否降级、当前 top_k 与 min_score 是多少
    def stats(self) -> dict:  # 返回索引运行态的快照字典
        """索引观测：总数、向量可用与否、上次召回方式。"""
        return {  # 拼一个观测 dict
            "indexed": len(self._texts),  # 已建索引的工具总数
            "vectorized": len(self._vectors),  # 已有向量的工具数（与 indexed 不相等说明有工具未编码）
            "vector_available": self._vector_available,  # 向量通道是否可用
            "last_source": self._last_source,  # 上次召回走的通道
            "top_k": self._top_k,  # 当前 top_k 配置
            "min_score": self._min_score,  # 当前分数门槛
        }

    # 【什么】清空三张内部表
    # 【作用】重建索引前调用；不清会残留旧工具的画像与向量
    def clear(self) -> None:  # 清空全部内部状态
        self._texts.clear()  # 清文本表
        self._vectors.clear()  # 清向量表
        self._metas.clear()  # 清元信息表


# 【什么】计算两个向量的余弦相似度
# 【作用】语义召回的打分函数；长度不等或零向量返回 0.0
def _cosine(a: List[float], b: List[float]) -> float:  # 模块级工具函数：算 a、b 两向量的余弦
    """余弦相似度（0~1 区间由调用方按 min_score 把关）。"""
    if not a or not b or len(a) != len(b):  # 空向量或维度不等
        return 0.0  # 无法计算，返回 0
    dot = sum(x * y for x, y in zip(a, b))  # 点积：Σ(x·y)
    na = sum(x * x for x in a) ** 0.5  # a 的 L2 模长：√Σ(x²)
    nb = sum(y * y for y in b) ** 0.5  # b 的 L2 模长：√Σ(y²)
    if na == 0 or nb == 0:  # 任一向量为零向量（除零保护）
        return 0.0  # 零向量无方向，余弦无意义，返回 0
    return dot / (na * nb)  # 余弦公式：dot / (|a|·|b|)
