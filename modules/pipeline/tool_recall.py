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
from typing import Callable, List, Optional, Sequence       # 类型注解：Callable 给自定义编码函数 embed_fn 用

from modules.common.logger import get_logger                # 统一日志入口
from modules.pipeline.models import IntentResult, IntentType, RecallResult, ToolHit  # 召回环节用到的数据模型
from config import get_settings                             # 读取 top_k / min_score / relative_ratio 等阈值配置

_logger = get_logger("pipeline.recall")                     # 本模块日志器，带 pipeline.recall 前缀


# 【什么】把提问编码成向量
# 【作用】编码失败返回 None 而不是抛异常，调用方据此切到关键词降级
# 【为什么】embedding 服务是外部依赖，它挂了只损失"语义检索"，不该让召回环节整个失败
def _embed_query(text: str) -> Optional[List[float]]:
    """把提问编码成向量；失败返回 None（调用方接着走关键词降级）。"""
    try:
        from modules.memory.embedding_service import get_embedding_service  # 延迟导入：避免无依赖环境导入本模块即失败
        return get_embedding_service().embed(text)          # 单条文本编码成向量
    except Exception as e:                                  # 服务不可用 / 网络异常 / 未配 key 等
        _logger.warning(f"提问向量编码失败，走关键词降级: {e}")  # 记警告日志
        return None                                         # 返回 None，调用方据此走关键词降级


# 【什么】用关键词在本地工具清单里匹配（最后兜底）
# 【作用】PG 完全不可用时仍能有候选；打分同关键词路径，上限 0.60
# 【为什么】存储层全挂时，本地工具清单是最后一道候选来源，保证链路不至于零候选
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
    lowered = [str(k).lower() for k in keywords if str(k).strip()]  # 关键词统一转小写、去空白项
    exclude_set = {str(x) for x in (exclude or [])}         # 黑名单转集合，O(1) 查询
    hits: List[ToolHit] = []                                # 命中结果容器
    if not lowered:                                         # 没有可用关键词
        return hits                                         # 直接返回空

    for tool in tools or []:                                # 遍历本地工具清单
        name = str(getattr(tool, "name", "") or "")         # 取工具名（缺属性时兜底空串）
        if not name or name in exclude_set:                 # 无名或在黑名单
            continue                                        # 跳过
        desc = str(getattr(tool, "description", "") or "")  # 取工具描述
        blob = f"{name} {desc}".lower()                     # 名 + 描述拼成匹配文本并转小写
        hit = sum(1 for k in lowered if k in blob)          # 统计命中的关键词个数
        if hit == 0:                                        # 一个都没命中
            continue                                        # 跳过
        hits.append(ToolHit(
            name=name, description=desc, score=round(hit / len(lowered) * 0.60, 4),  # 命中率 * 0.60 作为分数
            domain_tag=str(getattr(tool, "domain_tag", "") or "general"),  # 业务域兜底 general
            source_type=str(getattr(tool, "tool_type", "") or "local"),    # 来源类型兜底 local
            source="keyword",                               # 标记来自关键词路径
        ))
    hits.sort(key=lambda h: h.score, reverse=True)          # 按分数降序
    return hits[:top_k]                                     # 截取前 top_k 个返回


# 【什么】Tool-RAG 召回总入口：非工具意图直接跳过，其余走三级降级
# 【作用】产出候选工具清单；任何一级失败都只置 degraded=True，不抛异常
# 【为什么】召回失败只意味着"没有工具候选"，模型还能空手答，不该让请求失败
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
    if intent.intent_type not in IntentType.TOOL_INTENTS:   # chat/rag/continue_memory/invalid 都不需要工具候选
        return RecallResult(
            executed=False, tools=[], source="none",
            reason=f"意图为 {intent.intent_type}，不执行 Tool-RAG",
        )

    exclude = list(intent.exclude_tool_list or []) + list(extra_exclude or [])  # 模型黑名单 + 调用方黑名单合并
    keywords = intent.keyword_list or []                                        # 关键词，向量降级路径用

    if store is None:                                       # 未注入存储层时内部创建
        try:
            from modules.pipeline.tool_store_pg import ToolVectorStore  # 延迟导入：避免 PG 依赖缺失时导入即失败
            store = ToolVectorStore()                       # 默认构造：表名与维度取配置
        except Exception as e:                              # 存储类不可用（缺依赖等）
            _logger.warning(f"Tool-RAG 存储创建失败，走本地兜底: {e}")  # 记警告日志
            store = None                                    # 置 None，后面走本地兜底分支

    # ------------------------------------------------------------------
    # 【第 1 关：向量召回】需要 embedding + PG 都可用
    # ------------------------------------------------------------------
    if store is not None and store.available:               # 存储层连上了才尝试向量路径
        query_emb = embed_fn(query) if embed_fn else _embed_query(query)  # 编码提问；失败返回 None
        if query_emb:                                       # 编码成功才走向量检索
            rows = store.search(query_emb, domain_tag=intent.domain_tag, exclude=exclude,
                                top_k=top_k, min_score=min_score, relative_ratio=relative_ratio)  # 向量召回（带域过滤与黑名单）
            if rows:                                        # 向量路径有候选
                return RecallResult(
                    executed=True,
                    tools=[ToolHit(**{**r, "source": "pgvector"}) for r in rows],  # 逐行转 ToolHit 并标记来源
                    source="pgvector",
                    reason=f"向量召回 {len(rows)} 个（domain={intent.domain_tag}, top_k={top_k}）",
                )
            # 向量召回为空：再试一次关键词，避免"明明有合适工具却召不到"
            rows = store.search_keyword(keywords or [query], domain_tag=intent.domain_tag,
                                        exclude=exclude, top_k=top_k)  # 关键词召回；无关键词时用整句
            if rows:                                        # 关键词路径有候选
                return RecallResult(
                    executed=True,
                    tools=[ToolHit(**{**r, "source": "keyword"}) for r in rows],  # 逐行转 ToolHit 并标记来源
                    source="keyword", degraded=True,
                    reason="向量召回为空，改用关键词召回",
                    degrade_reason="向量召回无达标结果",
                )
            return RecallResult(executed=True, tools=[], source="none",
                               reason=f"向量与关键词召回均为空（min_score={min_score}）")  # 两条路径都空：如实返回零候选

    # ------------------------------------------------------------------
    # 【第 2 关：本地候选池兜底】PG 不可用时，用传入的本地工具清单按关键词匹配
    # ------------------------------------------------------------------
    if fallback_tools:                                      # 调用方给了本地工具清单
        hits = _match_local_pool(keywords or [query], fallback_tools, exclude, top_k)  # 本地关键词匹配
        return RecallResult(
            executed=True, tools=hits, source="keyword", degraded=True,
            reason="PG 不可用，用本地工具清单做关键词兜底",
            degrade_reason="pgvector 存储不可用",
        )

    return RecallResult(
        executed=True, tools=[], source="none", degraded=True,
        reason="无可用的工具候选来源", degrade_reason="pgvector 与本地候选池均不可用",
    )                                                       # 所有来源都不可用：返回空候选 + 降级标记
