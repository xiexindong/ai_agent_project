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
from abc import ABC  # 抽象基类标记：BaseSkill 不允许直接实例化
from dataclasses import dataclass  # 数据类装饰器：自动生成 __init__/__repr__ 等样板代码
from typing import List, Optional  # 类型注解：列表 / 可选值
from modules.common.logger import get_logger  # 统一日志器工厂

_logger = get_logger("skill")  # 创建本模块专属日志器，日志来源名固定为 skill

# 【什么】路由层 → 中文展示名的映射表
# 【作用】把 route 字段翻成人话写进注入文本与日志，方便肉眼判断命中层级
# 【为什么】keyword/vector/llm 这类内部代号对排查不直观，中文名让日志与提示词更易读
ROUTE_LABELS = {
    "keyword": "L1关键词路由",  # L1：关键词路由层的中文名
    "vector": "L2向量语义路由",  # L2：向量语义路由层的中文名
    "llm": "L3 LLM兜底路由",  # L3：LLM 兜底路由层的中文名
}


# 【什么】一次技能匹配的结果对象
# 【作用】三级路由共用这一个返回结构，靠 route 区分是谁判定的
# 【为什么】统一返回结构让上层不必关心命中层级；各层把原始分数写进各自字段，互不覆盖便于追溯
@dataclass
class SkillMatch:  # 技能匹配结果数据类
    """技能匹配结果。

    三级路由共用这一个对象，用 route 字段区分是哪一层判定的：
        keyword → score 是关键词命中权重累加值（base_skill.match_score）
        vector  → score 是 query 与技能描述文本的余弦相似度
        llm     → score 是 LLM 自评的置信度 confidence
    """
    skill: "BaseSkill"          # 命中的技能实例
    score: float                # 综合得分：命中权重累加值 / 余弦相似度 / LLM 置信度
    hit_keywords: List[str]     # 命中的触发词列表；L2/L3 命中时为空列表
    route: str = "keyword"      # 判定层级：keyword / vector / llm
    vector_score: float = 0.0   # L2 原始余弦相似度；非 L2 时为 0.0
    confidence: float = 0.0     # L3 原始置信度；非 L3 时为 0.0
    route_reason: str = ""      # 路由理由，目前只有 L3 会填

    # 【什么】让 SkillMatch 支持解包成 (skill, score)
    # 【作用】兼容旧调用方式 tuple(match) 写法，避免二元组与新结构混用报错
    # 【为什么】历史代码按 (skill, score) 二元组解包；实现 __iter__ 后新旧写法都能跑，迁移零成本
    def __iter__(self):  # 迭代器协议：按序产出 skill 与 score
        yield self.skill  # 第一个元素：技能实例
        yield self.score  # 第二个元素：综合得分


# 【什么】技能基类
# 【作用】子类只声明类属性就得到一个技能；需要复杂意图判断时可重写 match_score
# 【为什么】「声明即技能」把定义成本压到最低；匹配/渲染/导出的通用逻辑集中在这里，子类不重复造轮子
class BaseSkill(ABC):  # 技能抽象基类
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

    name: str = ""                      # 技能唯一标识，注册器用它做字典键
    display_name: str = ""              # 展示名，注入文本与 CLI 展示用
    description: str = ""               # 技能职责说明，给匹配器和 LLM 看
    trigger_keywords: List[str] = []    # 触发词/短语列表，命中即计入得分
    instructions: str = ""              # 激活后注入给 LLM 的 SOP 作业流程
    bound_tools: List[str] = []         # 推荐的工具名列表（Function Calling 工具名）
    priority: int = 0                   # 同分时的优先级，数值越大越优先

    _WEIGHT_PHRASE = 0.6    # 长度 >= 3 的短语权重：最能代表明确意图
    _WEIGHT_WORD = 0.4      # 长度 == 2 的双字词权重：中等强度证据
    _WEIGHT_CHAR = 0.15     # 长度 <= 1 的单字权重：弱证据，凑够 3 个才够阈值 0.35

    # 【什么】按触发词计算「意图证据强度」得分
    # 【作用】L1 关键词路由的打分来源；得分是否够由 SkillRegistry 的阈值裁决
    # 【为什么】打分只负责「累证据」、裁决交给注册器阈值，职责分离后调阈值不用改每个技能
    def match_score(self, query: str) -> SkillMatch:  # 入参为用户问句，返回匹配结果
        """
        基于触发关键词计算「意图证据强度」得分（0~1，封顶 1.0）。

        规则（与 reranker 降级精排同样的轻量风格，零外部依赖）：
        - 按关键词长度分档给证据权重：短语0.6 / 双字词0.4 / 单字0.15；
        - 得分 = 命中权重累加（不除以词表总权重，避免词表越大越难激活）；
        - 是否激活由 SkillRegistry 的阈值统一裁决（默认需 >=0.35，
          即命中 1 个双字意图词，或 3 个以上单字弱词）。
        """
        text = (query or "").lower()  # 问句统一转小写，保证中英文大小写不影响命中；None 兜底为空串
        hits: list[str] = [kw for kw in self.trigger_keywords if kw.lower() in text]  # 筛出所有出现在问句里的触发词（子串匹配）
        if not hits:  # 一个触发词都没命中
            return SkillMatch(skill=self, score=0.0, hit_keywords=[])  # 返回 0 分结果，不参与候选
        hit_weight = sum(self._keyword_weight(kw) for kw in hits)  # 把每个命中词按长度分档换成权重并累加
        score = round(min(1.0, hit_weight), 4)  # 权重累加值封顶 1.0，保留 4 位小数
        return SkillMatch(skill=self, score=score, hit_keywords=hits)  # 返回带命中词明细的匹配结果

    # 【什么】按关键词长度分档返回权重
    # 【作用】越长的短语越能代表明确意图，给高权重；单字只当弱证据
    # 【为什么】「报销标准」比「销」更能说明意图；按长度分档用零成本规则逼近这个常识
    @classmethod
    def _keyword_weight(cls, keyword: str) -> float:  # 入参为单个触发词，返回它的证据权重
        """关键词长度分档：越长的短语越能代表明确意图。"""
        if len(keyword) >= 3:  # 长度 >= 3 视为短语
            return cls._WEIGHT_PHRASE  # 短语权重 0.6
        if len(keyword) == 2:  # 长度 == 2 视为双字词
            return cls._WEIGHT_WORD  # 双字词权重 0.4
        return cls._WEIGHT_CHAR  # 其余（单字）权重 0.15

    # 【什么】给技能画「语义画像」的文本
    # 【作用】供 L2 向量路由编码；比单独用 description 语义区分度更高
    # 【为什么】description 太短向量区分度不够；拼上触发词与 SOP 首句后画像更丰满，相似度更有判别力
    @property
    def routing_text(self) -> str:  # 只读属性：返回语义画像文本
        """
        供【L2 向量语义路由】编码的语义文本。

        为什么不是直接用 description：单独的描述太短，向量区分度不够。
        这里把「展示名 + 职责说明 + 典型触发词 + SOP 首句」拼成一段话，
        相当于给这个技能画一张语义画像，让 query 与它算相似度时更有判别力。
        """
        kws = "、".join(self.trigger_keywords)  # 把全部触发词用顿号连成一串
        sop_head = ""  # SOP 首句默认为空
        if self.instructions:  # 技能配了 SOP 才取首句
            sop_head = self.instructions.strip().splitlines()[0]  # 取 SOP 的第一行作为代表句
        return (  # 拼接语义画像文本
            f"{self.display_name or self.name}。"  # 展示名（缺省退回内部名）
            f"{self.description} 典型问法: {kws}。{sop_head}"  # 职责说明 + 触发词 + SOP 首句
        )

    # 【什么】把技能渲染成注入 ReAct prompt 的上下文文本
    # 【作用】告诉 LLM 当前激活了哪个技能、能用哪些工具、按什么 SOP 走
    # 【为什么】技能只做「软引导」：把证据与 SOP 摆给模型看但不硬屏蔽工具，既引导方向又保留模型自由度
    def format_context(self, match: Optional[SkillMatch] = None) -> str:  # 可选地带入匹配结果以展示命中信息
        """
        格式化为注入 ReAct Prompt 的技能上下文文本。
        """
        hit_text = ""  # 命中意图词片段，默认空
        if match and match.hit_keywords:  # 有匹配结果且有命中词（主要是 L1 命中）
            hit_text = f"\n命中意图词: {', '.join(match.hit_keywords)}"  # 拼出命中词展示行
        route_text = ""  # 路由来源片段，默认空
        if match and match.route and match.route != "keyword":  # 非 L1 命中才展示路由来源（L1 是默认定律不必说）
            route_text = (  # 拼出路由来源展示行
                f"\n路由来源: {ROUTE_LABELS.get(match.route, match.route)}"  # 路由层代号翻成中文名，查不到就用原值
                f"（得分 {match.score:.2f}）"  # 附上该层得分，保留两位小数
            )
        tools_text = "、".join(self.bound_tools) if self.bound_tools else "（无专用工具，依据知识库与通用能力处理）"  # 绑定工具列表，空则给兜底说明
        return (  # 拼出完整注入文本
            f"【激活技能】{self.display_name or self.name}{hit_text}{route_text}\n"  # 技能标题行：展示名 + 命中词 + 路由来源
            f"技能说明: {self.description}\n"  # 技能职责说明行
            f"推荐工具: {tools_text}\n"  # 推荐工具行
            f"作业流程(SOP):\n{self.instructions}"  # SOP 正文
        )

    # 【什么】导出技能的元信息字典
    # 【作用】给 CLI 展示和 API 返回用；不含 instructions，避免回传过长正文
    # 【为什么】对外只需要元信息摘要；SOP 正文可能很长且属内部实现细节，不应泄漏到 API 响应里
    def to_dict(self) -> dict:  # 返回技能元信息字典
        """技能元信息（CLI 展示 / API 返回用）。"""
        return {
            "name": self.name,  # 技能唯一标识
            "display_name": self.display_name,  # 展示名
            "description": self.description,  # 职责说明
            "trigger_keywords": self.trigger_keywords,  # 触发词列表
            "bound_tools": self.bound_tools,  # 绑定工具列表
            "priority": self.priority,  # 优先级
        }
