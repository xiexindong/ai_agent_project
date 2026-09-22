"""
分层 Agent 路由模块
===================
在 ReAct 开始之前先做一层「轻决策」：这一轮该用哪些技能、带哪几个工具。

【三层各管什么】
    L0 路由层（本模块）  决定名单：技能匹配 → 语义召回 → 合并截断 → 回退兜底
    L1 执行层（ReAct）   拿到名单后照常推理、调工具、收敛答案
    MCP 网关             只提供「有哪些远程技能」的目录，不参与决策，也不直接面对 LLM

【为什么要有 L0 这一层】
    没有它，ReAct 每轮面对的是「全部工具」，工具越多，模型越容易选错、
    prompt 越长（技能膨胀）。有了它，模型每轮只看本轮用得上的那几个，
    而且这份名单是可解释的：打日志能看到"从 32 个里选出 3 个，理由是什么"。

【决策顺序与优先级】
    1. 技能绑定工具（skill.bound_tools）—— 强约束，SOP 点名要用的一定带上
    2. Tool-RAG 语义召回 top-k        —— 补上没有写进技能、但语义相关的工具
    3. 默认工具回退                    —— 前两步都为空时的兜底，保证不会"无工具可用"
    4. max_tools_per_turn 硬截断       —— 最后一道闸，名单再合理也不超过这个数
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from modules.common.logger import get_logger

_logger = get_logger("agent_router")

# 召回全空时兜底带的工具：本地通用能力，保证每轮至少有事可做
_DEFAULT_TOOLS = ("calculator", "data_query")


@dataclass
class RoutingDecision:
    """一次路由决策的完整结果（可解释、可打日志、可回放）。

    Attributes:
        tool_names:  最终名单（已去重、已截断），交给 ReAct 组装 function schema
        tool_scores: 工具名 → 召回分数（技能绑定工具记 1.0，表示强约束）
        source:      "skill_bound" / "tool_rag" / "skill_bound+tool_rag" / "default"
        skill:       命中的技能实例（未命中为 None）
        skill_route: 技能命中层级 keyword / vector / llm / prepared（上游已匹配）
        pool_size:   候选池规模（用于日志里的"从 N 个里选出 k 个"）
        reason:      一句话决策理由
    """
    tool_names: List[str] = field(default_factory=list)
    tool_scores: Dict[str, float] = field(default_factory=dict)
    source: str = "default"
    skill: Optional[object] = None
    skill_route: str = ""
    skill_score: float = 0.0
    pool_size: int = 0
    reason: str = ""

    def describe(self) -> str:
        """一行摘要，给日志与 demo 用。

        【输入输出对照】
            输入  tool_names=["data_query","calculator"], pool_size=32,
                  skill=<报销咨询>, skill_route="keyword", source="skill_bound+tool_rag"
            输出  "技能=报销咨询(关键词,0.60) | 工具 2/32: data_query, calculator | 来源=技能绑定+语义召回"
        """
        skill_text = "无"
        if self.skill is not None:
            name = getattr(self.skill, "name", "")
            skill_text = f"{name}({_ROUTE_LABEL.get(self.skill_route, self.skill_route)},{self.skill_score:.2f})"
        tools_text = "、".join(self.tool_names) or "（无）"
        return (
            f"技能={skill_text} | 工具 {len(self.tool_names)}/{self.pool_size}: {tools_text}"
            f" | 来源={_SOURCE_LABEL.get(self.source, self.source)}"
        )


_ROUTE_LABEL = {"keyword": "关键词", "vector": "向量", "llm": "LLM", "prepared": "上游已匹配"}
_SOURCE_LABEL = {
    "skill_bound": "技能绑定",
    "tool_rag": "语义召回",
    "skill_bound+tool_rag": "技能绑定+语义召回",
    "default": "默认兜底",
}


class AgentRouter:
    """L0 路由：把「全部工具」收敛成「本轮工具名单」。

    Args:
        skill_registry: 技能注册表（可选）。命中技能时用它的 bound_tools 当种子。
        tool_index:     ToolRAGIndex（可选）。为 None 时退化为「只按技能绑定」。
        gateway:        MCPGateway（可选）。负责把召回到的远程工具按需装进 registry。
        top_k:          语义召回保留几个（通常取配置 tool_rag_top_k）。
        max_tools:      单轮名单硬上限（配置 max_tools_per_turn）。
        default_tools:  兜底工具名元组。

    【on-demand 的关键一步】
        route() 里会调 gateway.ensure_registered()：远程工具此刻才连 server、才注册。
        这一步之前，远程技能只是目录里的一行描述，模型看不到、进程也没起。
    """

    def __init__(
        self,
        skill_registry=None,
        tool_index=None,
        gateway=None,
        top_k: int = 5,
        max_tools: int = 8,
        default_tools=_DEFAULT_TOOLS,
    ):
        self.skill_registry = skill_registry
        self.tool_index = tool_index
        self.gateway = gateway
        self.top_k = max(1, int(top_k))
        self.max_tools = max(1, int(max_tools))
        self.default_tools = tuple(default_tools)

    # ------------------------------------------------------------------
    # 【主入口】一次路由 = 建候选池 → 取种子 → 语义召回 → 合并截断 → 按需加载
    #
    #   【候选池是什么】已注册的本地/远程工具 + 网关目录里尚未注册的远程工具。
    #     目录里的工具只有描述，没有实例 —— 它们能参与"该不该用"的召回，
    #     但还不具备"能被调用"的能力，后者由 ensure_registered 补上。
    #
    #   【为什么先种子后召回】技能 SOP 点名的工具是硬需求（如报销技能要查报销单），
    #     语义召回可能因为措辞差异漏掉它。种子优先保证 SOP 一定能跑通。
    #
    #   【输入输出对照】
    #     输入  query="我的报销单到哪一步了"，候选池 32 个工具
    #     输出  RoutingDecision(tool_names=["query_reimbursement","data_query","calculator"],
    #                           tool_scores={"query_reimbursement":1.0,"data_query":0.62,"calculator":0.31},
    #                           source="skill_bound+tool_rag", pool_size=32,
    #                           skill=<报销进度查询>, skill_route="keyword", skill_score=0.6)
    # ------------------------------------------------------------------
    def route(
        self,
        query: str,
        registry=None,
        skill_name: str = "",
        skill_match=None,
        query_emb=None,
    ) -> RoutingDecision:
        pool: Dict[str, object] = {}
        pool_names: List[str] = []

        for tool in (registry.list_tools() if registry else []):
            name = getattr(tool, "name", "")
            if name and name not in pool:
                pool[name] = tool
                pool_names.append(name)

        remote_catalog = []
        if self.gateway:
            try:
                remote_catalog = self.gateway.catalog()
            except Exception as e:
                _logger.warning(f"网关目录读取失败，本轮只按本地工具路由: {e}")
        for entry in remote_catalog:
            name = entry.get("name", "")
            if name and name not in pool:
                pool[name] = entry
                pool_names.append(name)

        # ------------------------------------------------------------------
        # 【索引增量更新】只对新出现的工具编码；已索引的不会重复调 embedding。
        # ------------------------------------------------------------------
        if self.tool_index:
            self.tool_index.index_tools([pool[n] for n in pool_names])

        # ------------------------------------------------------------------
        # 【技能是怎么来的】三种来源，按代价从小到大：
        #   prepared —— 上游 prepare_state 已经跑过三级路由，这里直接用结果（零额外开销）
        #   按名字取 —— 上游只留了技能名，用注册表反查（一次字典查找）
        #   现算    —— 都没有时才跑一次三级路由（可能触发 L3 的 LLM 调用，最贵）
        # ------------------------------------------------------------------
        skill, route, score = None, "", 0.0
        if skill_match is not None:
            skill, route, score = skill_match.skill, "prepared", getattr(skill_match, "score", 0.0)
        elif skill_name and self.skill_registry:
            skill = self.skill_registry.get_skill(skill_name)
            route = "prepared" if skill else ""
        elif self.skill_registry:
            m = self.skill_registry.match(query, query_emb=query_emb)
            if m:
                skill, route, score = m.skill, m.route, m.score

        chosen: List[str] = []
        scores: Dict[str, float] = {}
        if skill is not None:
            for name in getattr(skill, "bound_tools", []) or []:
                if name in pool and name not in chosen:
                    chosen.append(name)
                    scores[name] = 1.0          # 1.0 = 技能强绑定，不参与分数排序

        rag_hits: List[str] = []
        if self.tool_index:
            for hit in self.tool_index.recall(query, top_k=self.top_k):
                if hit.name in pool and hit.name not in scores:
                    rag_hits.append(hit.name)
                    scores[hit.name] = hit.score

        merged = chosen + rag_hits
        truncated = merged[: self.max_tools]

        if not truncated:
            truncated = [n for n in self.default_tools if n in pool]
            scores = {n: 0.0 for n in truncated}
            reason = "技能未命中且语义召回为空，回退默认工具兜底"
        elif chosen and rag_hits:
            reason = f"技能绑定 {len(chosen)} 个 + 语义召回 {len(rag_hits)} 个"
        elif chosen:
            reason = f"技能绑定 {len(chosen)} 个（语义召回无达标结果）"
        else:
            reason = f"无技能命中，按语义召回 {len(rag_hits)} 个"

        if len(merged) > self.max_tools:
            reason += f"；候选 {len(merged)} 个超过单轮上限 {self.max_tools}，已截断"

        source = "default"
        if chosen and rag_hits:
            source = "skill_bound+tool_rag"
        elif chosen:
            source = "skill_bound"
        elif rag_hits:
            source = "tool_rag"

        # ------------------------------------------------------------------
        # 【按需加载】名单里的远程工具此刻才连 server + 装进 registry。
        #   这一步之前它们只是目录里的一行描述；这一步之后才具备可执行能力。
        # ------------------------------------------------------------------
        if self.gateway and registry is not None:
            self.gateway.ensure_registered(registry, truncated)

        decision = RoutingDecision(
            tool_names=truncated,
            tool_scores={n: scores.get(n, 0.0) for n in truncated},
            source=source,
            skill=skill,
            skill_route=route,
            skill_score=float(score or 0.0),
            pool_size=len(pool_names),
            reason=reason,
        )
        _logger.info(f"[L0路由] {decision.describe()} | {reason} | query='{query[:30]}'")
        return decision
