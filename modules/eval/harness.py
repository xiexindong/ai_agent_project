"""
被测对象装配（System Under Test）
================================
把「要评测的那套东西」按模式装配出来，让 runner 只管跑、不管怎么造。

【为什么要分 offline / online 两档】
    同一套评测在两种场合都要能跑：
        CI / 本地没网没密钥  → 要跑得动、跑得快、结果可复现
        发版前人工验收       → 要跑完整链路（含 embedding 与 LLM）
    两档的差异只在于「外部依赖开不开」，被测的业务逻辑完全一样，
    否则评测的就不是同一件东西了。

【offline 档为什么强制关掉向量与 LLM 路由】
    可复现性。向量要调 embedding 服务、L3 要调 LLM，两者都受网络、
    配额、模型版本影响 —— 同一份代码今天 0.82 明天 0.79，没法做回归门禁。
    offline 档只跑确定性部分（L1 关键词 + 关键词召回 + 规则风控 + 规则意图），
    它测的是「我们的编排逻辑」，不是「外部模型的表现」。

【offline 档测不到什么，必须写清楚】
    测不到语义召回的真实效果、测不到答案质量。这两项归 online 档，
    报告里会标注 mode，避免拿 offline 的分数冒充端到端效果。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# 模拟 RAG 召回的长上下文：约上千 token，用来稳定触发预算裁剪
# 【为什么用固定文本而不是随机生成】随机文本会让 token 数每次不同，
#   报告里的 avg_tokens 就没法跨次比较，回归判定也失去意义。
_MOCK_RAG = (
    "《差旅费用报销管理办法》第三章 报销标准："
    "员工因公出差产生的交通费、住宿费、市内交通费按本办法执行。"
    "一、交通费：高铁二等座、飞机经济舱据实报销，超标准部分自理。"
    "二、住宿费：一线城市每晚不超过 600 元，二线城市 450 元，其他城市 350 元。"
    "三、市内交通：凭票据实报销，单日上限 100 元。"
    "四、审批流程：提交后由直属主管初审，财务复核，超过 5000 元需分管副总审批。"
    "五、报销时限：出差结束后 30 个工作日内提交，逾期不予受理。"
    "六、票据要求：发票抬头须为公司全称，税号正确，否则退回重开。"
) * 6

_MOCK_PERMISSION = "可访问：本人报销单、本人订单；禁止：他人数据、薪资、人事档案"
_MOCK_MEMORY = "用户上月咨询过差旅报销标准；偏好分点作答"
_MOCK_TOOLS = "calculator / data_query / web_search / get_current_time / knowledge_stats"


class StubGateway:
    """离线网关替身：只提供工具目录，不起子进程。

    【为什么要这个替身】
        真网关会 Popen 一个 MCP 子进程，评测跑 40 条就要起 40 次进程，
        既慢又引入"子进程启动失败"这种与被测逻辑无关的不稳定因素。
        而路由决策真正需要的只是「目录里有哪些工具、描述是什么」，
        这份数据已经落在 data/mcp_catalog.json 里 —— 直接读它即可，
        既离线可复现，又是项目的真实数据。

    【ensure_registered 为什么是空实现】
        评测只测"该选谁"，不测"选完之后能不能真的调用"。
        装工具要连 server，属于集成测试范畴，留给用户自己跑 demo 验证。
    """

    def __init__(self, catalog_path: str = "data/mcp_catalog.json"):
        self.catalog_path = catalog_path
        self._catalog: List[Dict] = []
        try:
            data = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
            self._catalog = list(data.get("tools", []) or [])
        except Exception as e:
            print(f"[warn] 工具目录读取失败，候选池只剩本地工具: {e}")

    # 【什么】返回远程工具目录
    # 【作用】与真网关同名同签名，router 无需区分
    def catalog(self) -> List[Dict]:
        return list(self._catalog)

    # 【什么】按需注册（空实现）
    # 【作用】满足 router 的调用契约；评测不测可执行性
    def ensure_registered(self, registry, names) -> int:
        return 0


@dataclass
class Sut:
    """被测系统句柄：持有装配好的一套组件。

    Attributes:
        mode:         "offline" 或 "online"
        notes:        装配过程中的降级说明（如"embedding 不可用，已降级"）
    """

    mode: str = "offline"
    notes: List[str] = field(default_factory=list)
    skill_registry: Any = None
    tool_registry: Any = None
    tool_index: Any = None
    router: Any = None

    # ------------------------------------------------------------------
    # 【四个被测面】技能路由 / 工具路由 / 意图 / 风控，加一个提示词组装
    #   每个方法都返回「标准化 dict + 耗时」，异常时返回 error 而不是抛出，
    #   保证一条用例炸了不影响其余 39 条。
    # ------------------------------------------------------------------
    # 【什么】跑一次路由（技能 + 工具名单）
    # 【作用】返回标准化结果；异常时 ok=False 并带 error
    def route(self, query: str) -> Dict:
        started = time.time()
        try:
            decision = self.router.route(query, registry=self.tool_registry)
            skill = getattr(decision, "skill", None)
            return {
                "ok": True,
                "skill": getattr(skill, "name", "") if skill else "",
                "skill_route": getattr(decision, "skill_route", ""),
                "tools": list(getattr(decision, "tool_names", []) or []),
                "source": getattr(decision, "source", ""),
                "reason": getattr(decision, "reason", ""),
                "latency_ms": round((time.time() - started) * 1000, 1),
                "error": "",
            }
        except Exception as e:
            return {"ok": False, "skill": "", "skill_route": "", "tools": [],
                    "source": "", "reason": "", "latency_ms": 0.0,
                    "error": f"{type(e).__name__}: {e}"}

    # 【什么】跑一次意图识别
    # 【作用】offline 档强制 use_llm=False 走规则，保证确定性
    def intent(self, query: str) -> Dict:
        from modules.pipeline.intent import recognize_intent

        started = time.time()
        try:
            use_llm = self.mode == "online"
            r = recognize_intent(query, llm=None, use_llm=use_llm)
            return {"ok": True, "intent": str(getattr(r, "intent_type", "")),
                    "confidence": float(getattr(r, "confidence", 0.0) or 0.0),
                    "latency_ms": round((time.time() - started) * 1000, 1), "error": ""}
        except Exception as e:
            return {"ok": False, "intent": "", "confidence": 0.0,
                    "latency_ms": 0.0, "error": f"{type(e).__name__}: {e}"}

    # 【什么】跑一次前置风控
    # 【作用】纯规则，两档行为一致
    def risk(self, query: str) -> Dict:
        from modules.pipeline.risk_guard import check_risk

        try:
            r = check_risk(query)
            return {"ok": True, "passed": bool(getattr(r, "passed", True)),
                    "level": str(getattr(r, "level", "")),
                    "action": str(getattr(r, "action", "")), "error": ""}
        except Exception as e:
            return {"ok": False, "passed": True, "level": "", "action": "",
                    "error": f"{type(e).__name__}: {e}"}

    # 【什么】跑一次提示词组装（主链路模板 react.reason）
    # 【作用】评测 prompt 这一层：能否渲染成功、是否守住 token 预算、裁剪是否记录
    def build_prompt(self, query: str) -> Dict:
        from modules.prompt import PromptBuilder

        started = time.time()
        try:
            b = PromptBuilder(
                template_name="react.reason",
                strategy="react",
                role="你是一个企业级智能AI助手。",
                max_iter=10,
                max_context_tokens=3000,
            )
            rec = b.build_record(
                query,
                permission_context=_MOCK_PERMISSION,
                skill_context="（未匹配到专门技能，按通用企业助手流程处理）",
                rag_context=_MOCK_RAG,
                memory_context=_MOCK_MEMORY,
                tool_history="（未提供）",
                observation="（初始状态，暂无工具观察）",
                tools_desc=_MOCK_TOOLS,
            )
            return {
                "ok": True,
                "prompt_tokens": int(rec.get("prompt_tokens", 0)),
                "context_tokens": int(rec.get("context_tokens", 0)),
                "trimmed": list(rec.get("trimmed", []) or []),
                "template": str(rec.get("template", "")),
                "latency_ms": round((time.time() - started) * 1000, 1),
                "error": "",
            }
        except Exception as e:
            return {"ok": False, "prompt_tokens": 0, "context_tokens": 0,
                    "trimmed": [], "template": "",
                    "latency_ms": 0.0, "error": f"{type(e).__name__}: {e}"}


# 【什么】装配一套被测对象
# 【作用】按 mode 决定外部依赖开不开，返回 Sut
def build_sut(mode: str = "offline", max_tools: int = 8, top_k: int = 5,
              catalog_path: str = "data/mcp_catalog.json") -> Sut:
    """装配被测系统。

    Args:
        mode:         "offline"（确定性，无外部依赖）/ "online"（完整链路）
        max_tools:    单轮工具名单上限，对齐生产配置
        top_k:        语义召回保留数
        catalog_path: 远程工具目录（离线替身的数据源），给空串则只用本地工具

    【为什么要显式关掉 L2/L3 而不是等它自己失败降级】
        等它失败要付出一次真实的网络超时（可能几十秒 × 每条用例），
        而且"失败后永久降级"意味着前几条用例的耗时和后面的不一样，
        测出来的延迟没有可比性。显式关闭 = 行为确定 + 跑得快。
    """
    from modules.skills import SkillRegistry, get_default_skills
    from modules.tools.tool_registry import ToolRegistry
    from modules.tools.tool_rag import ToolRAGIndex
    from modules.agent.router import AgentRouter

    notes: List[str] = []
    offline = mode != "online"

    skill_registry = SkillRegistry(
        llm_route_enabled=False if offline else None,
    )
    skill_registry.register_batch(get_default_skills())

    tool_registry = ToolRegistry()
    tool_registry.register_batch(tool_registry.get_default_tools())

    tool_index = ToolRAGIndex(top_k=top_k)

    if offline:
        # 强制关掉两个非确定性来源，把被测面收敛到"我们的编排逻辑"
        skill_registry._vector_available = False
        tool_index._vector_available = False
        notes.append("L2 向量路由已关闭（离线档）")
        notes.append("L3 LLM 路由已关闭（离线档）")
        notes.append("Tool-RAG 走关键词召回（离线档）")
    else:
        notes.append("完整三级路由 + embedding 语义召回")

    gateway = StubGateway(catalog_path) if catalog_path else None
    if gateway is not None:
        notes.append(f"远程工具目录：{catalog_path}（{len(gateway.catalog())} 个，离线替身）")

    router = AgentRouter(
        skill_registry=skill_registry,
        tool_index=tool_index,
        gateway=gateway,          # 不起子进程，只提供目录
        top_k=top_k,
        max_tools=max_tools,
    )

    return Sut(
        mode=mode,
        notes=notes,
        skill_registry=skill_registry,
        tool_registry=tool_registry,
        tool_index=tool_index,
        router=router,
    )
