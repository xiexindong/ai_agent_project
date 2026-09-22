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
from modules.prompt.templates import (
    PromptError,
    PromptRegistry,
    PromptTemplate,
)
from modules.prompt.builtin import get_registry, register_builtin, reset_registry
from modules.prompt.strategies import (
    STRATEGIES,
    FewShotExample,
    FewShotStore,
    build_strategy_block,
)
from modules.prompt.builder import PromptBuilder, parse_json_output

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
