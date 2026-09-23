"""
Tool-RAG 召回编排
=================
流水线第 5 步：只有 local_skill / mcp_remote 两类意图才跑，召回 Top3 候选技能。

【为什么只有两类意图跑召回】
    rag 走知识库向量、continue_memory 走记忆库、chat 直接答 ——
    它们都不需要工具候选，跑一次召回 = 白花一次 embedding 调用 + 一次 PG 查询。

【过滤三件套】
    domain_tag          按业务域收窄候选（finance 的提问不该召回 it 域的工单工具）
    exclude_tool_list   LLM 给的本轮黑名单
    min_score           相似度下限，宁可召空也不塞噪声

【三级降级】
    ① 向量召回（PG + pgvector 可用）
    ② 关键词召回（PG 可用但 embedding 挂了 / 向量召回为空）
    ③ 本地候选池关键词匹配（PG 完全不可用，用传入的本地工具清单兜底）
    任何一级失败都不会抛异常，只是 degraded=True。
"""
from typing import Callable, List, Optional, Sequence

from modules.common.logger import get_logger
from modules.pipeline.models import IntentResult, IntentType, RecallResult, ToolHit
from config import get_settings

_logger = get_logger("pipeline.recall")


# 【什么】把提问编码成向量
# 【作用】编码失败返回 None 而不是抛异常，调用方据此切到关键词降级
def _embed_query(text: str) -> Optional[List[float]]:
    """把提问编码成向量；失败返回 None（调用方接着走关键词降级）。"""
    try:
        from modules.memory.embedding_service import get_embedding_service
        return get_embedding_service().embed(text)
    except Exception as e:
        _logger.warning(f"提问向量编码失败，走关键词降级: {e}")
        return None


# 【什么】用关键词在本地工具清单里匹配（最后兜底）
# 【作用】PG 完全不可用时仍能有候选；打分同关键词路径，上限 0.60
def _match_local_pool(
    keywords: Sequence[str],     # 关键词列表
    tools: Sequence,             # 本地工具对象清单，取 name/description 参与匹配
    exclude: Sequence[str],      # 黑名单工具名
    top_k: int,                  # 返回数量上限
) -> List[ToolHit]:
    """用关键词在本地工具清单里匹配（PG 完全不可用时的最后兜底）。

    【输入输出对照】
        keywords=["报销","发票"], tools=[DataQueryTool, CalculatorTool], top_k=3
            → [ToolHit(name="data_query", score=0.30, source="keyword")]
            （命中 1/2 个关键词 → 0.5 * 0.60 = 0.30；calculator 未命中不返回）
    """
    lowered = [str(k).lower() for k in keywords if str(k).strip()]
    exclude_set = {str(x) for x in (exclude or [])}
    hits: List[ToolHit] = []
    if not lowered:
        return hits

    for tool in tools or []:
        name = str(getattr(tool, "name", "") or "")
        if not name or name in exclude_set:
            continue
        desc = str(getattr(tool, "description", "") or "")
        blob = f"{name} {desc}".lower()
        hit = sum(1 for k in lowered if k in blob)
        if hit == 0:
            continue
        hits.append(ToolHit(
            name=name, description=desc, score=round(hit / len(lowered) * 0.60, 4),
            domain_tag=str(getattr(tool, "domain_tag", "") or "general"),
            source_type=str(getattr(tool, "tool_type", "") or "local"),
            source="keyword",
        ))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


# 【什么】Tool-RAG 召回总入口：非工具意图直接跳过，其余走三级降级
# 【作用】产出候选工具清单；任何一级失败都只置 degraded=True，不抛异常
def recall_tools(
    query: str,                    # Token 控制后的用户提问
    intent: IntentResult,          # 意图识别结果：决定是否召回、提供域与黑名单
    store=None,                    # ToolVectorStore；None 时内部创建
    top_k: int = 0,                # 召回数量；<=0 时取 settings.pipeline_tool_top_k（3）
    min_score: float = -1,         # 相似度下限；<0 时取 settings.tool_rag_min_score
    extra_exclude: Optional[List[str]] = None,  # 调用方追加的黑名单（如权限不允许的工具）
    fallback_tools: Optional[Sequence] = None,  # 本地工具清单，PG 不可用时兜底
    embed_fn: Optional[Callable[[str], Optional[List[float]]]] = None,  # 自定义编码函数，测试用
) -> RecallResult:
    """按意图与提问召回候选工具。

    Args:
        query: Token 控制后的用户提问
        intent: 意图识别结果（决定是否执行召回、提供 domain 与黑名单）
        store: ToolVectorStore 实例；为 None 时内部创建一个
        top_k: 召回数量，默认取 settings.pipeline_tool_top_k（3）
        min_score: 相似度下限，默认取 settings.tool_rag_min_score（0.25，见 config.py）
        extra_exclude: 调用方额外追加的黑名单（如权限不允许的工具）
        fallback_tools: 本地工具清单，PG 不可用时用它做关键词兜底
        embed_fn: 自定义编码函数（测试用）；默认用智谱 embedding

    Returns:
        RecallResult：executed=False 表示本次没跑召回（非工具类意图）

    【输入输出对照】
        intent=local_skill, query="上海明天天气怎么样", PG+embedding 可用
            → RecallResult(executed=True, source="pgvector",
                           tools=[ToolHit("city_weather", 0.82), ToolHit("web_search", 0.55)])
        intent=rag, query="报销标准是什么"
            → RecallResult(executed=False, source="none", tools=[],
                           reason="意图为 rag，不执行 Tool-RAG")
    """
    s = get_settings()
    top_k = top_k or int(s.pipeline_tool_top_k)                             # 0/None 都表示"用配置值"
    min_score = min_score if min_score >= 0 else float(s.tool_rag_min_score)  # 负数哨兵 = 用配置值
    relative_ratio = float(s.tool_rag_relative_ratio)                       # 相对下限系数

    # ------------------------------------------------------------------
    # 【第 0 关：意图类型】非工具意图直接返回，不碰 embedding、不碰 PG
    # ------------------------------------------------------------------
    if intent.intent_type not in IntentType.TOOL_INTENTS:
        return RecallResult(
            executed=False, tools=[], source="none",
            reason=f"意图为 {intent.intent_type}，不执行 Tool-RAG",
        )

    exclude = list(intent.exclude_tool_list or []) + list(extra_exclude or [])  # 模型黑名单 + 调用方黑名单合并
    keywords = intent.keyword_list or []                                        # 关键词，向量降级路径用

    if store is None:
        try:
            from modules.pipeline.tool_store_pg import ToolVectorStore
            store = ToolVectorStore()
        except Exception as e:
            _logger.warning(f"Tool-RAG 存储创建失败，走本地兜底: {e}")
            store = None

    # ------------------------------------------------------------------
    # 【第 1 关：向量召回】需要 embedding + PG 都可用
    # ------------------------------------------------------------------
    if store is not None and store.available:
        query_emb = embed_fn(query) if embed_fn else _embed_query(query)
        if query_emb:
            rows = store.search(query_emb, domain_tag=intent.domain_tag, exclude=exclude,
                                top_k=top_k, min_score=min_score, relative_ratio=relative_ratio)
            if rows:
                return RecallResult(
                    executed=True,
                    tools=[ToolHit(**{**r, "source": "pgvector"}) for r in rows],
                    source="pgvector",
                    reason=f"向量召回 {len(rows)} 个（domain={intent.domain_tag}, top_k={top_k}）",
                )
            # 向量召回为空：再试一次关键词，避免"明明有合适工具却召不到"
            rows = store.search_keyword(keywords or [query], domain_tag=intent.domain_tag,
                                        exclude=exclude, top_k=top_k)
            if rows:
                return RecallResult(
                    executed=True,
                    tools=[ToolHit(**{**r, "source": "keyword"}) for r in rows],
                    source="keyword", degraded=True,
                    reason="向量召回为空，改用关键词召回",
                    degrade_reason="向量召回无达标结果",
                )
            return RecallResult(executed=True, tools=[], source="none",
                               reason=f"向量与关键词召回均为空（min_score={min_score}）")

    # ------------------------------------------------------------------
    # 【第 2 关：本地候选池兜底】PG 不可用时，用传入的本地工具清单按关键词匹配
    # ------------------------------------------------------------------
    if fallback_tools:
        hits = _match_local_pool(keywords or [query], fallback_tools, exclude, top_k)
        return RecallResult(
            executed=True, tools=hits, source="keyword", degraded=True,
            reason="PG 不可用，用本地工具清单做关键词兜底",
            degrade_reason="pgvector 存储不可用",
        )

    return RecallResult(
        executed=True, tools=[], source="none", degraded=True,
        reason="无可用的工具候选来源", degrade_reason="pgvector 与本地候选池均不可用",
    )
