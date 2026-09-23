"""
技能（Skills）子系统
====================
Skill 是比 Function Call（原子工具）更高一层的能力封装：
- 触发条件：trigger_keywords 描述什么意图下激活该技能
- 作业流程：instructions 给 LLM 的专属 SOP 提示词
- 工具绑定：bound_tools 声明该技能推荐使用的函数工具
运行时由 SkillRegistry 按用户提问匹配技能，ReActAgent 将技能 SOP 注入推理上下文。
"""
# 【什么】导入技能基类与匹配结果类型
# 【作用】自定义新技能时只需继承 BaseSkill，返回值统一是 SkillMatch
from modules.skills.base_skill import BaseSkill, SkillMatch
# 【什么】导入技能注册器（三级路由的持有者）
# 【作用】所有技能的注册与意图匹配都经过它这一个入口
from modules.skills.skill_registry import SkillRegistry
# 【什么】导入内置技能集的工厂函数
# 【作用】默认 4 个技能由它一次性产出，启动时批量注册即可
from modules.skills.builtin_skills import get_default_skills

# 【什么】对外可见的名字清单
# 【作用】from modules.skills import * 时只导出这些，不泄露内部符号
__all__ = [
    "BaseSkill",
    "SkillMatch",
    "SkillRegistry",
    "get_default_skills",
]
