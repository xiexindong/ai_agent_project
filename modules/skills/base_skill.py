"""
技能基类模块
============
定义技能（Skill）的标准结构与匹配接口。

Skill 与 Tool（Function Call）的分工：
- Tool：原子能力，一个函数做一件事（如 calculator 计算一个表达式），由 LLM 通过 Function Calling 调用；
- Skill：能力包，包含「触发词 + SOP 作业流程 + 推荐工具集合」，
  系统在 ReAct 开始前按用户意图匹配技能，把专属流程提示词注入推理上下文，
  引导 LLM 按标准步骤组合调用一个或多个 Tool。

子类只需声明类属性即可定义一个技能；需要复杂意图判断时可重写 match_score。
"""
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional
from modules.common.logger import get_logger

_logger = get_logger("skill")

# 三级路由的分层标签（日志 / 注入上下文展示用）
ROUTE_LABELS = {
    "keyword": "L1关键词路由",
    "vector": "L2向量语义路由",
    "llm": "L3 LLM兜底路由",
}


@dataclass
class SkillMatch:
    """技能匹配结果。

    三级路由共用这一个对象，用 route 字段区分是哪一层判定的：
        keyword → score 是关键词命中权重累加值（base_skill.match_score）
        vector  → score 是 query 与技能描述文本的余弦相似度
        llm     → score 是 LLM 自评的置信度 confidence
    """
    skill: "BaseSkill"
    score: float          # 匹配得分 0~1（不同路由层含义不同，见上）
    hit_keywords: List[str]
    route: str = "keyword"      # 命中层级：keyword / vector / llm
    vector_score: float = 0.0   # L2 余弦相似度，仅 route="vector" 时有值
    confidence: float = 0.0     # L3 LLM 自评置信度，仅 route="llm" 时有值
    route_reason: str = ""      # L3 LLM 给出的选择理由（便于排查误判）

    def __iter__(self):
        # 便于 `skill, score = registry.match(...)` 之外的元组式解包场景
        yield self.skill
        yield self.score


class BaseSkill(ABC):
    """
    技能基类。
    子类通过类属性声明：
        name:             技能唯一标识
        display_name:     展示名
        description:      技能描述（给匹配器与 LLM 看）
        trigger_keywords: 触发关键词/短语（命中即计入得分）
        instructions:     激活后注入给 LLM 的 SOP 作业流程
        bound_tools:      绑定推荐的工具名（Function Calling 工具名）
        priority:         同分时的优先级，数值越大越优先
    """

    name: str = ""
    display_name: str = ""
    description: str = ""
    trigger_keywords: List[str] = []
    instructions: str = ""
    bound_tools: List[str] = []
    priority: int = 0

    # 命中证据权重：短语越长意图越强；单字词证据弱，需多个叠加才足以激活
    _WEIGHT_PHRASE = 0.6   # >=3 字的强意图短语（如「等于多少」「查一下」）
    _WEIGHT_WORD = 0.4     # 2 字意图词（如「订单」「计算」「新闻」）
    _WEIGHT_CHAR = 0.15    # 单字弱词（如「加」「除」），防闲聊误命中

    def match_score(self, query: str) -> SkillMatch:
        """
        基于触发关键词计算「意图证据强度」得分（0~1，封顶 1.0）。

        规则（与 reranker 降级精排同样的轻量风格，零外部依赖）：
        - 按关键词长度分档给证据权重：短语0.6 / 双字词0.4 / 单字0.15；
        - 得分 = 命中权重累加（不除以词表总权重，避免词表越大越难激活）；
        - 是否激活由 SkillRegistry 的阈值统一裁决（默认需 >=0.35，
          即命中 1 个双字意图词，或 3 个以上单字弱词）。
        """
        text = (query or "").lower()
        hits = [kw for kw in self.trigger_keywords if kw.lower() in text]
        if not hits:
            return SkillMatch(skill=self, score=0.0, hit_keywords=[])
        hit_weight = sum(self._keyword_weight(kw) for kw in hits)
        score = round(min(1.0, hit_weight), 4)
        return SkillMatch(skill=self, score=score, hit_keywords=hits)

    @classmethod
    def _keyword_weight(cls, keyword: str) -> float:
        """关键词长度分档：越长的短语越能代表明确意图。"""
        if len(keyword) >= 3:
            return cls._WEIGHT_PHRASE
        if len(keyword) == 2:
            return cls._WEIGHT_WORD
        return cls._WEIGHT_CHAR

    @property
    def routing_text(self) -> str:
        """
        供【L2 向量语义路由】编码的语义文本。

        为什么不是直接用 description：单独的描述太短，向量区分度不够。
        这里把「展示名 + 职责说明 + 典型触发词 + SOP 首句」拼成一段话，
        相当于给这个技能画一张语义画像，让 query 与它算相似度时更有判别力。
        """
        kws = "、".join(self.trigger_keywords)
        sop_head = ""
        if self.instructions:
            sop_head = self.instructions.strip().splitlines()[0]
        return (
            f"{self.display_name or self.name}。"
            f"{self.description} 典型问法: {kws}。{sop_head}"
        )

    def format_context(self, match: Optional[SkillMatch] = None) -> str:
        """
        格式化为注入 ReAct Prompt 的技能上下文文本。
        """
        hit_text = ""
        if match and match.hit_keywords:
            hit_text = f"\n命中意图词: {', '.join(match.hit_keywords)}"
        # 非关键词层命中时，把路由来源写进上下文，方便排查「为什么激活了这个技能」
        route_text = ""
        if match and match.route and match.route != "keyword":
            route_text = (
                f"\n路由来源: {ROUTE_LABELS.get(match.route, match.route)}"
                f"（得分 {match.score:.2f}）"
            )
        tools_text = "、".join(self.bound_tools) if self.bound_tools else "（无专用工具，依据知识库与通用能力处理）"
        return (
            f"【激活技能】{self.display_name or self.name}{hit_text}{route_text}\n"
            f"技能说明: {self.description}\n"
            f"推荐工具: {tools_text}\n"
            f"作业流程(SOP):\n{self.instructions}"
        )

    def to_dict(self) -> dict:
        """技能元信息（CLI 展示 / API 返回用）。"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "trigger_keywords": self.trigger_keywords,
            "bound_tools": self.bound_tools,
            "priority": self.priority,
        }
