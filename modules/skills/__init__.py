"""
技能（Skills）子系统
====================
Skill 是比 Function Call（原子工具）更高一层的能力封装：
- 触发条件：trigger_keywords 描述什么意图下激活该技能
- 作业流程：instructions 给 LLM 的专属 SOP 提示词
- 工具绑定：bound_tools 声明该技能推荐使用的函数工具
运行时由 SkillRegistry 按用户提问匹配技能，ReActAgent 将技能 SOP 注入推理上下文。
"""
from modules.skills.base_skill import BaseSkill, SkillMatch
from modules.skills.skill_registry import SkillRegistry
from modules.skills.builtin_skills import get_default_skills

__all__ = [
    "BaseSkill",
    "SkillMatch",
    "SkillRegistry",
    "get_default_skills",
]
