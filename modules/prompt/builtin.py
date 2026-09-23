"""
内置提示词模板
==============
把项目里已有的提示词搬进注册表，并补几个常用模板，附带版本号。

【为什么把现有提示词搬进来，而不是重写一份】
    搬进来 = 行为不变，但从此可以在一处改、按版本回滚、被 demo 直接对比。
    若在这里另写一套措辞，ReActAgent 的输出会悄悄变化，
    而"悄悄变化"正是提示词工程最忌讳的事：效果变了却没人知道是哪句改的。
    所以 v1 的正文与现有代码逐字一致，v2 才允许改动（用于 A/B 对比）。

【变量一览（照抄现有代码，避免漂移）】
    react.system   max_iter
    react.reason   query / skill_context / permission_context / rag_context
                   memory_context / tool_history / observation / tools_desc
    qa.grounded    strategy_block / query / permission_context / rag_context / memory_context
    summary        query / answer
    merge          summaries
    classify.intent query / labels
"""
from modules.prompt.templates import PromptRegistry, PromptTemplate

# ----------------------------------------------------------------------
# react.system v1：与迁移前写在 react_agent.py 类里的 SYSTEM_PROMPT 正文逐字一致
# ----------------------------------------------------------------------
_REACT_SYSTEM_V1 = (
    "你是一个企业级智能AI助手，具备知识库问答、历史记忆、工具调用能力。\n\n"
    "## 工作流程（ReAct）\n"
    "1. Reason：分析用户问题，判断是否需要调用工具\n"
    "2. Action：如需工具，调用对应工具获取信息\n"
    "3. Observation：观察工具返回结果，决定是否继续调用或给出答案\n\n"
    "## 核心规则\n"
    "- 仅基于知识库和工具返回的事实回答，禁止编造\n"
    "- 知识库无相关内容时，如实告知并尝试工具补充\n"
    "- 工具调用需确认参数正确性\n"
    " - 最多循环{max_iter}次，超过后直接给出当前最佳答案\n\n"
    "## 回答要求\n"
    "- 提供完整详细的答案（包含推理依据、数据来源）\n"
    "- 末尾附【摘要】标签，用一句话总结本轮对话核心"
)

# v2：在 v1 基础上追加两条约束，用于演示「改提示词 = 改一个版本」
#   追加内容对应两个真实问题：
#     1. 模型把工具结果复述一遍却不给结论（加"必须给出明确结论"）
#     2. 知识库为空时模型仍自行发挥（加"必须声明依据不足"）
_REACT_SYSTEM_V2 = _REACT_SYSTEM_V1 + (
    "\n\n## 补充约束\n"
    "- 工具已返回数据时，必须给出明确结论，不能只复述观察结果\n"
    "- 依据不足时，先说'当前资料不足以确定'，再说明还缺什么"
)

# ----------------------------------------------------------------------
# react.role：react.system v1 去掉「## 工作流程（ReAct）」那一节
#   【为什么要拆出一个 role 版】
#     流程那一节现在由 strategies.py 的 react 策略块提供（_REACT_HEAD）。
#     拆开之后：角色与规则归模板，流程归策略，两者可以各自换 ——
#     换问法（react → cot）不动模板，改模板不动问法。
#   【拆掉的那 5 行去哪了】
#     "1. Reason / 2. Action / 3. Observation" 三条，被策略块里更完整的版本取代：
#     补了「同一个工具用相同参数重复调用没有意义」这条，
#     针对的就是实测到的现象：同一句"3*7"曾被连续调 9 次计算器。
#   【保留 react.system v1 的用途】
#     它仍是改造前的原文（与 react_agent.py 旧 SYSTEM_PROMPT 逐字一致），
#     用于回归比对（demo 第六幕）与「想回到旧文案」的场景。
# ----------------------------------------------------------------------
_REACT_ROLE = (
    "你是一个企业级智能AI助手，具备知识库问答、历史记忆、工具调用能力。\n\n"
    "## 核心规则\n"
    "- 仅基于知识库和工具返回的事实回答，禁止编造\n"
    "- 知识库无相关内容时，如实告知并尝试工具补充\n"
    "- 工具调用需确认参数正确性\n"
    " - 最多循环{max_iter}次，超过后直接给出当前最佳答案\n\n"
    "## 回答要求\n"
    "- 提供完整详细的答案（包含推理依据、数据来源）\n"
    "- 末尾附【摘要】标签，用一句话总结本轮对话核心"
)

# ----------------------------------------------------------------------
# react.reason：与迁移前写在 react_agent.py 类里的 REASON_PROMPT 正文逐字一致
# ----------------------------------------------------------------------
_REACT_REASON = (
    "## 用户提问\n{query}\n\n"
    "## 激活技能（SOP作业流程）\n{skill_context}\n\n"
    "## 权限边界（硬性约束，越界即违规）\n{permission_context}\n\n"
    "## 知识库参考\n{rag_context}\n\n"
    "## 历史记忆\n{memory_context}\n\n"
    "## 历史工具调用\n{tool_history}\n\n"
    "## 工具观察结果\n{observation}\n\n"
    "## 可用工具（Function Calling）\n{tools_desc}\n\n"
    "请分析当前情况，决定下一步行动：\n"
    "- 如果已激活技能，优先遵循该技能的作业流程(SOP)，并优先使用其推荐工具\n"
    "- 如果已有足够信息回答，直接给出最终答案\n"
    "- 如果需要更多信息，选择合适的工具调用\n"
    "- 严格使用 Function Calling 格式输出工具调用"
)

# ----------------------------------------------------------------------
# qa.grounded：知识库问答（PromptBuilder 的默认模板）
#   与 react.reason 的区别：不含工具相关变量，多一个 strategy_block 插位，
#   供四种策略自由替换（"怎么问"这一段由 strategies.py 生成）。
# ----------------------------------------------------------------------
_QA_GROUNDED = (
    "## 回答策略\n{strategy_block}\n\n"
    "## 用户提问\n{query}\n\n"
    "## 权限边界\n{permission_context}\n\n"
    "## 知识库参考\n{rag_context}\n\n"
    "## 历史记忆\n{memory_context}\n\n"
    "请按上面的回答策略作答。"
)

# ----------------------------------------------------------------------
# summary / merge：与 summary_generator.py:17、25 正文一致
# ----------------------------------------------------------------------
_SUMMARY = (
    "请将以下用户提问与AI回答，提炼为一句话精简摘要（不超过50字）。"
    "摘要需保留核心信息（问题主题+关键结论），用于后续语义检索匹配。\n\n"
    "用户提问: {query}\n"
    "AI回答: {answer}\n\n"
    "精简摘要:"
)

_MERGE = (
    "请将以下多轮对话摘要合并为一个全局摘要（不超过100字），"
    "保留时间顺序与关键信息，去除冗余细节，用于压缩上下文历史。\n\n"
    "历史摘要列表:\n{summaries}\n\n"
    "合并摘要:"
)

# ----------------------------------------------------------------------
# classify.intent：演示「JSON 输出约束」与花括号转义
#   【转义坑】模板走 str.format，正文里要输出字面的 { } 必须写成 {{ }}。
#   写成单花括号会被当成变量，渲染时抛 KeyError 或静默错位。
#   下面 {{"intent": "报销咨询"}} 渲染后正是 {"intent": "报销咨询"}。
# ----------------------------------------------------------------------
_CLASSIFY_INTENT = (
    "请判断用户提问属于下列哪一类意图，只输出一个 JSON 对象。\n\n"
    "候选意图: {labels}\n"
    "用户提问: {query}\n\n"
    "输出格式（严格照此，不要多余文字）:\n"
    '{{"intent": "意图名", "confidence": 0.9, "reason": "一句话理由"}}\n\n'
    "JSON:"
)

_BUILTINS = [
    PromptTemplate("react.role", _REACT_ROLE, "v1", "react", "ReAct 角色与规则（流程段交给 react 策略块）"),
    PromptTemplate("react.system", _REACT_SYSTEM_V1, "v1", "react", "ReAct 系统提示词（与现有代码一致）"),
    PromptTemplate("react.system", _REACT_SYSTEM_V2, "v2", "react", "ReAct 系统提示词（补充结论与依据约束）"),
    PromptTemplate("react.reason", _REACT_REASON, "v1", "react", "ReAct 推理步骤提示词"),
    PromptTemplate("qa.grounded", _QA_GROUNDED, "v1", "zero_shot", "知识库问答通用模板（builder 默认）"),
    PromptTemplate("summary", _SUMMARY, "v1", "zero_shot", "单轮对话摘要"),
    PromptTemplate("merge", _MERGE, "v1", "zero_shot", "多轮摘要合并"),
    PromptTemplate("classify.intent", _CLASSIFY_INTENT, "v1", "zero_shot", "意图分类（JSON 输出）"),
]

_registry: PromptRegistry | None = None   # 全局注册表单例；None 表示尚未创建过


# 【什么】把 8 个内置模板注册进给定注册表
# 【作用】让新建的注册表也有全套内置模板，返回它便于链式调用
def register_builtin(registry: PromptRegistry) -> PromptRegistry:
    """把内置模板注册进一个（通常是新建的）注册表。"""
    for tpl in _BUILTINS:
        registry.register(tpl)
    return registry


# 【什么】懒加载的全局注册表单例
# 【作用】全进程共用一份模板表，避免「A 注册的模板 B 看不到」
def get_registry() -> PromptRegistry:
    """全局注册表单例：首次调用时创建并注册内置模板。

    【为什么是单例】
        与 LLMClient 同样的理由：注册表是只读的共享资源，
        每个模块各建一个会导致「A 模块注册的模板 B 模块看不到」。
    """
    global _registry
    if _registry is None:
        _registry = register_builtin(PromptRegistry())
    return _registry


# 【什么】重建全局注册表单例
# 【作用】测试之间互相隔离用，返回新注册表以免用例互相污染
def reset_registry() -> PromptRegistry:
    """清空单例并重建（测试用，避免用例之间互相污染）。"""
    global _registry
    _registry = register_builtin(PromptRegistry())
    return _registry
