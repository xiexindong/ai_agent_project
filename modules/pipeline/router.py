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
from typing import Optional                                 # 类型注解：recall 参数可为 None

from modules.common.logger import get_logger                # 统一日志入口
from modules.pipeline.models import (                       # 路由用到的数据模型
    Branch, IntentResult, IntentType, RecallResult, RouteDecision, ToolHit,  # 分支常量 / 意图结果 / 召回结果 / 路由决策 / 工具命中
)
from config import get_settings                             # 读取置信度阈值配置

_logger = get_logger("pipeline.route")                      # 本模块日志器，带 pipeline.route 前缀

# 意图 → 分支：invalid 落到 reject，其余同名
_INTENT_TO_BRANCH = {
    IntentType.INVALID: Branch.REJECT,                      # 无效输入 → 拒答/澄清支路
    IntentType.CHAT: Branch.CHAT,                           # 闲聊 → 普通对话支路
    IntentType.LOCAL_SKILL: Branch.LOCAL_SKILL,             # 本地工具意图 → 本地工具支路
    IntentType.RAG: Branch.RAG,                             # 知识库意图 → RAG 支路
    IntentType.MCP_REMOTE: Branch.MCP_REMOTE,               # 远程技能意图 → MCP 支路
    IntentType.CONTINUE_MEMORY: Branch.CONTINUE_MEMORY,     # 续聊意图 → 记忆支路
}

# 只有这两条支路真的会调工具，其余支路即使召回有结果也不带
_TOOL_BRANCHES = (Branch.LOCAL_SKILL, Branch.MCP_REMOTE)


# 【什么】分支路由总入口：意图 + 召回 → 一条决策
# 【作用】定支路并给出工具白名单，置信度不足时一律禁工具、清单清空
# 【为什么】把"能不能调工具"收敛成一个布尔值：上层只看 tool_calls_allowed，不用自己算阈值
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
    if min_confidence < 0:                                  # 调用方未显式指定阈值
        min_confidence = float(get_settings().pipeline_min_confidence)  # 取配置值（默认 0.5）

    branch = _INTENT_TO_BRANCH.get(intent.intent_type, Branch.CHAT)   # 未知类别兜底为 chat
    recall = recall or RecallResult()                       # None 时按空召回结果处理
    conf = float(intent.intent_confidence or 0.0)           # 置信度转 float，None 兜底 0.0

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

    if invalid:                                             # 无效输入
        reason = "输入判定为无效，走拒答/澄清支路"
    elif not confidence_ok:                                 # 置信度不达标
        reason = f"置信度 {conf:.2f} < 阈值 {min_confidence:.2f}，禁止调用任何工具"
    elif branch in _TOOL_BRANCHES:                          # 工具支路且达标
        reason = f"{branch} 分支，可用工具 {len(recall.tools)} 个"
    else:                                                   # rag / continue_memory / chat
        reason = f"{branch} 分支走专用链路，不需要工具" if branch != Branch.CHAT else "普通对话"

    # 只有工具支路 + 允许调工具时才带工具，其余情况一律空列表
    tools: list = list(recall.tools) if (tool_calls_allowed and branch in _TOOL_BRANCHES) else []

    decision = RouteDecision(
        branch=branch,                                      # 落点支路
        tool_calls_allowed=tool_calls_allowed,              # 是否允许调工具
        tools=tools,                                        # 工具白名单（禁工具时为空）
        reason=reason,                                      # 决策说明
        min_confidence=min_confidence,                      # 本次生效的阈值
    )
    _logger.info(
        f"分支路由: branch={branch} allowed={tool_calls_allowed} "
        f"tools={[t.name for t in tools]} | {reason}"
    )                                                       # 决策日志：支路/是否允许工具/工具名清单/原因
    return decision                                         # 返回路由决策
