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
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from modules.common.logger import get_logger

_logger = get_logger("tool_rag")

# 关键词证据的折算系数：命中同样多的词，语义相似度的可信度更高
_KEYWORD_SCALE = 0.6


# 【什么】一次工具召回的结果记录
# 【作用】带名字与分数返回给路由层，由它决定是否把工具放进 prompt
@dataclass
class ToolHit:
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
class ToolRAGIndex:
    """工具语义索引：index_tools() 建索引，recall() 按提问召回。

    【三张内部表】
        _texts   工具名 → 用途画像文本（建向量与关键词匹配共用）
        _vectors 工具名 → 语义向量（embedding 可用时才有）
        _metas   工具名 → 原始元信息（来源 server / 描述，排查时用）
    """

    def __init__(self, top_k: int = 5, min_score: float = 0.25, enabled: bool = True):
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
    def index_tools(self, tools: list) -> int:
        pending: List[tuple] = []
        for tool in tools or []:
            name, desc, schema, meta = self._normalize(tool)
            if not name:
                continue
            text = self._routing_text(name, desc, schema)
            already_indexed = not self._vector_available or name in self._vectors
            if self._texts.get(name) == text and already_indexed:
                continue
            self._texts[name] = text
            self._metas[name] = meta
            pending.append((name, text))

        if not pending:
            return 0
        if self._vector_available:
            try:
                from modules.memory.embedding_service import get_embedding_service
                vectors = get_embedding_service().embed_batch([t for _, t in pending])
                for (name, _), vec in zip(pending, vectors):
                    self._vectors[name] = vec
                _logger.info(f"Tool-RAG 已编码 {len(pending)} 个工具向量 | 索引总数={len(self._texts)}")
            except Exception as e:
                # 【降级而不是失败】向量只是召回手段之一，关键词照样能选出工具；
                #   这里若向上抛，整条对话就挂了 —— 召回不该成为可用性瓶颈。
                self._vector_available = False
                _logger.warning(f"Tool-RAG 向量不可用，降级为关键词召回: {e}")
        return len(pending)

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
    def recall(self, query: str, top_k: Optional[int] = None) -> List[ToolHit]:
        if not self._enabled or not query or not self._texts:
            return []
        k = max(1, int(top_k or self._top_k))

        if self._vector_available:
            hits = self._recall_vector(query, k)
            if hits:
                self._last_source = "vector"
                return hits
            self._last_source = "keyword"
            return self._recall_keyword(query, k)
        self._last_source = "keyword"
        return self._recall_keyword(query, k)

    # 【什么】语义召回：编码 query 后与全部工具向量算余弦
    # 【作用】主力召回路径；编码失败就关掉开关交给关键词兜底
    def _recall_vector(self, query: str, k: int) -> List[ToolHit]:
        try:
            from modules.memory.embedding_service import get_embedding_service
            qv = get_embedding_service().embed(query)
        except Exception as e:
            self._vector_available = False
            _logger.warning(f"Tool-RAG 查询向量失败，降级关键词: {e}")
            return []
        scored = [
            ToolHit(name, round(_cosine(qv, vec), 4), "vector")
            for name, vec in self._vectors.items()
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        hits = [h for h in scored if h.score >= self._min_score][:k]
        _logger.debug(
            "Tool-RAG 语义召回: "
            + ", ".join(f"{h.name}={h.score}" for h in scored[:5])
            + f" | 命中={len(hits)}"
        )
        return hits

    # 【什么】关键词降级召回：按 2-gram 片段重合打分
    # 【作用】embedding 不可用时仍能选出工具；分数乘 _KEYWORD_SCALE=0.6 折算
    def _recall_keyword(self, query: str, k: int) -> List[ToolHit]:
        q_frags = self._fragments(query)
        if not q_frags:
            return []
        hits: List[ToolHit] = []
        for name, text in self._texts.items():
            t_frags = set(self._fragments(text))
            hit = sum(1 for f in q_frags if f in t_frags)
            if not hit:
                continue
            raw = min(1.0, hit / len(q_frags))
            hits.append(ToolHit(name, round(raw * _KEYWORD_SCALE, 4), "keyword"))
        hits.sort(key=lambda h: h.score, reverse=True)
        return [h for h in hits if h.score >= self._min_score][:k]

    # ------------------------------------------------------------------
    # 【分词】中文按 2-gram 切，英文/数字按词切
    #
    #   【为什么中文用 2-gram 而不是整词】没有分词器（jieba 等）时，
    #   2-gram 是成本最低的近似：既能表达"天气""报销"这类双字意图，
    #   又不需要额外依赖。
    #
    #   【输入输出对照】
    #     输入  "北京天气 3*7"
    #     输出  ['北京', '京天', '天气', '3']
    #           （"3*7" 被正则切成 ['3','7']，长度 <2 的片段不参与）
    # ------------------------------------------------------------------
    # 【什么】把文本切成可比对片段
    # 【作用】中文走 2-gram、英文数字走词，供关键词召回做重合统计
    @staticmethod
    def _fragments(text: str) -> List[str]:
        lower = (text or "").lower()
        frags = [w for w in re.findall(r"[a-z0-9_]{2,}", lower)]
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lower))
        frags.extend(chinese[i:i + 2] for i in range(max(0, len(chinese) - 1)))
        return frags

    # ------------------------------------------------------------------
    # 【用途画像】只取参数名，不取完整 schema
    #
    #   【输入输出对照】
    #     输入  name="data_query", desc="查询业务数据", schema={"properties":{"table":{},"limit":{}}}
    #     输出  "data_query。查询业务数据 可填参数: table、limit"
    # ------------------------------------------------------------------
    # 【什么】拼出工具的用途画像文本
    # 【作用】只取参数名（最多前 6 个），不把完整 schema 塞进索引
    @staticmethod
    def _routing_text(name: str, description: str, schema: dict) -> str:
        props = (schema or {}).get("properties", {}) or {}
        params = "、".join(list(props.keys())[:6]) or "无参数"
        return f"{name}。{description} 可填参数: {params}"

    @staticmethod
    # 【什么】把 dict 形态与 BaseTool 形态的工具归一化
    # 【作用】索引与网关目录两套来源共用一套取字段逻辑
    def _normalize(tool) -> tuple:
        """把 BaseTool 实例或网关目录 dict 统一成 (name, description, schema, meta)。"""
        if isinstance(tool, dict):
            return (
                str(tool.get("name", "")),
                str(tool.get("description", "")),
                tool.get("inputSchema") or {},
                {"server": tool.get("server", "")},
            )
        # BaseTool 家族：MCPToolAdapter 用 parameters_schema，本地工具用 parameters
        name = getattr(tool, "name", "")
        desc = getattr(tool, "description", "")
        schema = getattr(tool, "parameters_schema", None) or getattr(tool, "parameters", None) or {}
        return str(name), str(desc), schema, {"server": getattr(tool, "tool_type", "")}

    # 【什么】输出索引运行态观测数据
    # 【作用】排查召回效果时看是否降级、当前 top_k 与 min_score 是多少
    def stats(self) -> dict:
        """索引观测：总数、向量可用与否、上次召回方式。"""
        return {
            "indexed": len(self._texts),
            "vectorized": len(self._vectors),
            "vector_available": self._vector_available,
            "last_source": self._last_source,
            "top_k": self._top_k,
            "min_score": self._min_score,
        }

    # 【什么】清空三张内部表
    # 【作用】重建索引前调用；不清会残留旧工具的画像与向量
    def clear(self) -> None:
        self._texts.clear()
        self._vectors.clear()
        self._metas.clear()


# 【什么】计算两个向量的余弦相似度
# 【作用】语义召回的打分函数；长度不等或零向量返回 0.0
def _cosine(a: List[float], b: List[float]) -> float:
    """余弦相似度（0~1 区间由调用方按 min_score 把关）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
