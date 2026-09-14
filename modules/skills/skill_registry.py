"""
技能注册器模块
==============
统一管理技能的注册、意图匹配、上下文构建。

匹配策略（关键词加权打分，与项目 reranker 降级精排同一风格，零外部依赖）：
- 对所有已注册技能计算 match_score；
- 选取得分最高的技能，同分按 priority 决出；
- 得分低于 match_threshold 视为「未命中专门技能」，走通用助手流程。

设计原则：
- 技能系统全程可降级：匹配异常时返回 None，不影响 ReAct 主链路；
- 技能只做「软引导」：注入 SOP 与推荐工具，不硬屏蔽其他工具，避免误匹配导致无法作答。
"""
from typing import Dict, List, Optional, Tuple
from modules.common.logger import get_logger
from modules.common.exceptions import handle_exception
from modules.skills.base_skill import BaseSkill, SkillMatch

_logger = get_logger("skill_registry")


class SkillRegistry:
    """技能注册器（本地内存注册表）。"""

    def __init__(self, match_threshold: float = 0.35):
        # 阈值含义：命中 1 个双字意图词(0.4)即可激活；仅命中单个单字弱词(0.15)不激活，防闲聊误命中
        self._skills: Dict[str, BaseSkill] = {}
        self._threshold = match_threshold

    def register(self, skill: BaseSkill):
        """注册技能，重名时覆盖并告警。"""
        if not skill.name:
            raise ValueError("技能缺少 name 标识")
        if skill.name in self._skills:
            _logger.warning(f"技能重复注册，覆盖: {skill.name}")
        self._skills[skill.name] = skill
        _logger.info(
            f"技能注册: {skill.name} | 触发词{len(skill.trigger_keywords)}个 "
            f"| 绑定工具={skill.bound_tools or '无'}"
        )

    def register_batch(self, skills: List[BaseSkill]):
        """批量注册技能。"""
        for skill in skills:
            self.register(skill)

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """按名称获取技能。"""
        return self._skills.get(name)

    def list_skills(self) -> List[BaseSkill]:
        """列出全部已注册技能。"""
        return list(self._skills.values())

    def match(self, query: str) -> Optional[SkillMatch]:
        """
        按用户提问匹配最佳技能。
        返回 SkillMatch；无技能达到阈值时返回 None（走通用流程）。
        """
        if not query or not self._skills:
            return None

        candidates: List[SkillMatch] = []
        for skill in self._skills.values():
            try:
                m = skill.match_score(query)
            except Exception as e:
                _logger.warning(f"技能[{skill.name}]匹配异常，跳过: {e}")
                continue
            if m.score > 0:
                candidates.append(m)

        if not candidates:
            _logger.info(f"技能未命中: query='{query[:30]}...' 走通用助手流程")
            return None

        # 得分降序，同分按 priority 降序
        candidates.sort(
            key=lambda m: (m.score, m.skill.priority),
            reverse=True,
        )
        best = candidates[0]
        if best.score < self._threshold:
            _logger.info(
                f"技能最高得分{best.score}低于阈值{self._threshold}，不激活: "
                f"{best.skill.name}"
            )
            return None

        _logger.info(
            f"技能命中: {best.skill.name} | score={best.score} "
            f"| 命中词={best.hit_keywords} | 候选数={len(candidates)}"
        )
        return best

    @handle_exception(default=(None, ""), log_msg="技能上下文构建异常")
    def build_skill_context(self, query: str) -> Tuple[Optional[BaseSkill], str]:
        """
        一步完成「匹配 + 上下文文本构建」，供 ReActAgent.prepare_state 调用。
        返回 (匹配到的技能, 注入Prompt的文本)；未命中时返回 (None, "")。
        """
        match = self.match(query)
        if not match:
            return None, ""
        return match.skill, match.skill.format_context(match)

    def describe_all(self) -> str:
        """生成全部技能的清单文本（CLI / 调试用）。"""
        if not self._skills:
            return "（未注册任何技能）"
        lines = []
        for i, skill in enumerate(self._skills.values(), 1):
            tools = "、".join(skill.bound_tools) if skill.bound_tools else "无（走RAG/通用能力）"
            keywords = "、".join(skill.trigger_keywords[:8])
            if len(skill.trigger_keywords) > 8:
                keywords += " …"
            lines.append(
                f"{i}. [{skill.name}] {skill.display_name}\n"
                f"   说明: {skill.description}\n"
                f"   触发词: {keywords}\n"
                f"   绑定工具: {tools} | 优先级: {skill.priority}"
            )
        return "\n".join(lines)
