"""
请求处理流水线
==============
用户输入 → 文本清洗 → 前置高危风控 → Token 控制与压缩 → 意图识别 → Tool-RAG 召回 → 分支路由

【一句话】
    用最便宜的手段（规则、清洗、压缩）先把请求收窄，
    只在必要环节才花 LLM 的钱，且置信度不足时绝不调用工具。

【各环节所在文件】
    models.py         数据结构（IntentResult / ToolHit / RouteDecision / PipelineResult）
    text_clean.py     ① 文本清洗
    risk_guard.py     ② 前置高危规则风控
    token_control.py  ③ Token 长度控制与超长压缩
    intent.py         ④ LLM 结构化意图识别（6 选 1）
    tool_store_pg.py  Tool-RAG 存储层：pgvector 建表 / 向量入库 / 语义检索
    tool_recall.py    ⑤ Tool-RAG 召回编排（仅 local_skill / mcp_remote 执行）
    router.py         ⑥ 分支路由（置信度 < 0.5 禁工具）
    pipeline.py       编排层：RequestPipeline.run()
"""
from modules.pipeline.models import (
    Branch, CleanResult, IntentResult, IntentType, PipelineResult,
    RecallResult, RiskResult, RouteDecision, TokenResult, ToolHit,
)
from modules.pipeline.text_clean import clean_text
from modules.pipeline.risk_guard import check_risk
from modules.pipeline.token_control import control_tokens
from modules.pipeline.intent import recognize_intent
from modules.pipeline.tool_store_pg import ToolVectorStore
from modules.pipeline.tool_recall import recall_tools
from modules.pipeline.router import route_branch
from modules.pipeline.pipeline import RequestPipeline

__all__ = [
    "Branch", "IntentType", "CleanResult", "RiskResult", "TokenResult",
    "IntentResult", "ToolHit", "RecallResult", "RouteDecision", "PipelineResult",
    "clean_text", "check_risk", "control_tokens", "recognize_intent",
    "ToolVectorStore", "recall_tools", "route_branch", "RequestPipeline",
]
