"""
分支路由
========
流水线第 6 步：把意图识别结果翻译成「这一轮到底走哪条支路 + 允许带哪些工具」。

【六条支路】
    reject          输入无效/被风控拦截，直接拒答或要求澄清
    chat            普通对话，不带工具
    local_skill     本地函数工具链路（带召回到的工具）
    rag             知识库检索链路（不带工具，走 Milvus）
    mcp_remote      MCP 远程技能链路（带召回到的工具）
    continue_memory 记忆续聊链路（先取历史记忆再回答）

【最重要的一条硬规则】
    intent_confidence < min_confidence（默认 0.5）→ tool_calls_allowed=False，工具列表清空。
    没把握的时候什么都不调，让模型用已有知识回答 —— 错调工具的代价远大于少调一次工具。
"""
from typing import Optional

from modules.common.logger import get_logger
from modules.pipeline.models import (
    Branch, IntentResult, IntentType, RecallResult, RouteDecision, ToolHit,
)
from config import get_settings

_logger = get_logger("pipeline.route")

# 意图 → 分支：invalid 落到 reject，其余同名
_INTENT_TO_BRANCH = {
    IntentType.INVALID: Branch.REJECT,
    IntentType.CHAT: Branch.CHAT,
    IntentType.LOCAL_SKILL: Branch.LOCAL_SKILL,
    IntentType.RAG: Branch.RAG,
    IntentType.MCP_REMOTE: Branch.MCP_REMOTE,
    IntentType.CONTINUE_MEMORY: Branch.CONTINUE_MEMORY,
}

# 只有这两条支路真的会调工具，其余支路即使召回有结果也不带
_TOOL_BRANCHES = (Branch.LOCAL_SKILL, Branch.MCP_REMOTE)


# 【什么】分支路由总入口：意图 + 召回 → 一条决策
# 【作用】定支路并给出工具白名单，置信度不足时一律禁工具、清单清空
def route_branch(
    intent: IntentResult,                          # 意图识别结果
    recall: Optional[RecallResult] = None,         # 召回结果；None 时按空结果处理
    min_confidence: float = -1,                    # 置信度下限；< 0 表示取配置值（默认 0.5）
) -> RouteDecision:
    """根据意图与召回结果决定分支与工具白名单。

    Args:
        intent: 意图识别结果
        recall: Tool-RAG 召回结果（可为 None，如被风控提前拦截的场景）
        min_confidence: 置信度下限，默认取 settings.pipeline_min_confidence（0.5）

    Returns:
        RouteDecision

    【输入输出对照】
        intent=mcp_remote/0.86, recall 命中 2 个
            → RouteDecision(branch="mcp_remote", tool_calls_allowed=True, tools=[2 个])
        intent=mcp_remote/0.35, recall 命中 2 个
            → RouteDecision(branch="mcp_remote", tool_calls_allowed=False, tools=[],
                             reason="置信度 0.35 < 0.50，禁止调用工具")
        intent=rag/0.80
            → RouteDecision(branch="rag", tool_calls_allowed=False, tools=[],
                             reason="rag 分支走专用链路，不需要工具")
        intent=invalid/0.90
            → RouteDecision(branch="reject", tool_calls_allowed=False, tools=[],
                             reason="输入判定为无效，走拒答/澄清支路")
    """
    # min_confidence<0 是"未指定"的哨兵值，此时回落到配置项的 0.5
    if min_confidence < 0:
        min_confidence = float(get_settings().pipeline_min_confidence)

    branch = _INTENT_TO_BRANCH.get(intent.intent_type, Branch.CHAT)   # 未知类别兜底为 chat
    recall = recall or RecallResult()
    conf = float(intent.intent_confidence or 0.0)

    # ------------------------------------------------------------------
    # 【硬规则：置信度不足 → 一律禁工具】
    #   注意是"禁工具"不是"改分支"：用户问的还是那件事，
    #   只是系统没有足够把握去调外部能力，让模型空手作答即可。
    # ------------------------------------------------------------------
    confidence_ok = conf >= min_confidence   # 置信度达标（>= 阈值）
    invalid = branch == Branch.REJECT        # 支路为 reject，即输入被判为无效

    # ------------------------------------------------------------------
    # 【allowed 的三重条件】置信度够 + 不是无效输入 + 走的是工具支路
    #   chat / rag / continue_memory 这些支路本来就不调工具，
    #   它们的 allowed 也必须是 False —— 否则上游看到 allowed=True 会误以为可以调。
    # ------------------------------------------------------------------
    tool_calls_allowed = confidence_ok and not invalid and branch in _TOOL_BRANCHES

    if invalid:
        reason = "输入判定为无效，走拒答/澄清支路"
    elif not confidence_ok:
        reason = f"置信度 {conf:.2f} < 阈值 {min_confidence:.2f}，禁止调用任何工具"
    elif branch in _TOOL_BRANCHES:
        reason = f"{branch} 分支，可用工具 {len(recall.tools)} 个"
    else:
        reason = f"{branch} 分支走专用链路，不需要工具" if branch != Branch.CHAT else "普通对话"

    # 只有工具支路 + 允许调工具时才带工具，其余情况一律空列表
    tools: list = list(recall.tools) if (tool_calls_allowed and branch in _TOOL_BRANCHES) else []

    decision = RouteDecision(
        branch=branch,
        tool_calls_allowed=tool_calls_allowed,
        tools=tools,
        reason=reason,
        min_confidence=min_confidence,
    )
    _logger.info(
        f"分支路由: branch={branch} allowed={tool_calls_allowed} "
        f"tools={[t.name for t in tools]} | {reason}"
    )
    return decision
