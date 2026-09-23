"""
提示词工程模块（Prompt Engineering）
===================================
把项目里散落的提示词收成一层：模板可注册带版本、变量渲染前校验、
策略可切换、上下文按 token 预算裁剪、输出格式可约束并解析。

【三层职责划分】
    templates.py  一条提示词长什么样（模板 + 版本 + 变量 + 渲染校验）
    strategies.py 这轮怎么问（zero_shot / few_shot / cot / react + 少样本示例库）
    builder.py    最终怎么拼（角色 + 策略 + 上下文 + 输出格式 + token 预算）

【最小用法（两行拿到 messages）】
    from modules.prompt import PromptBuilder
    msgs = PromptBuilder(strategy="few_shot", role="你是企业助手").build(
        "报销单什么时候到账？", rag_context="...", memory_context="..."
    )

【已接入的位置】
    react_agent.py       reason_step 用 PromptBuilder 生成 [system, user]；
                         system 在 __init__ 算一次（角色 + react 策略块），每轮只重拼 user
    react_agent.py       generate_answer 复用同一个 system_prompt
    summary_generator.py 两处改用 registry.render("summary" / "merge", ...)
    迁移前后措辞是否一致，由 examples/prompt_engineering_demo.py 第六幕回归比对。
"""
# 【什么】模板层导出：模板类、注册表、提示词异常
# 【作用】variables 校验与版本管理都在这层，builder/strategies 依赖它
from modules.prompt.templates import (
    PromptError,
    PromptRegistry,
    PromptTemplate,
)
# 【什么】内置模板注册入口：全局注册表单例与重建函数
# 【作用】进程一起来就有内置模板可用；测试用 reset_registry 隔离
from modules.prompt.builtin import get_registry, register_builtin, reset_registry
# 【什么】策略层导出：四种策略名 + 少样本示例库 + 策略块生成
# 【作用】builder 靠 build_strategy_block 拼出「怎么问」那一段指令
from modules.prompt.strategies import (
    STRATEGIES,
    FewShotExample,
    FewShotStore,
    build_strategy_block,
)
# 【什么】组装层导出：提示词组装器与 JSON 输出解析
# 【作用】对外最常用的两个入口，绝大多数场景只用这两个
from modules.prompt.builder import PromptBuilder, parse_json_output

# 【什么】本包对外的符号白名单
# 【作用】限制 import * 能拿到的名字，内部结构不对外承诺
__all__ = [
    "PromptTemplate",
    "PromptRegistry",
    "PromptError",
    "get_registry",
    "reset_registry",
    "register_builtin",
    "STRATEGIES",
    "FewShotExample",
    "FewShotStore",
    "build_strategy_block",
    "PromptBuilder",
    "parse_json_output",
]
