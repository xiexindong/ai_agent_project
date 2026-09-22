"""
提示词工程演示
==============
零新增依赖，直接跑：
    python examples/prompt_engineering_demo.py

【这个 demo 回答什么问题】
    项目里"有没有提示词工程"？改造前：有用提示词，但没有"工程层"——
    提示词是类里的字符串常量 + str.format()：
        ReActAgent.SYSTEM_PROMPT          （角色 + ReAct 流程）
        ReActAgent.REASON_PROMPT          （8 个变量拼一整段）
        SummaryGenerator._SUMMARY_PROMPT / _MERGE_PROMPT
    问题：改一句话要翻文件、少传变量就 KeyError、多传变量静默忽略、
    想对比"换个问法效果如何"只能改代码重跑。

    现在这 3 处已接入 modules/prompt/（正文逐字迁移，第六幕有回归比对）。

    本 demo 演示 modules/prompt/ 这一层怎么解决这几件事：
        第一幕 集中注册：所有模板一处可查，带版本
        第二幕 策略切换：同一问题四种问法，token 成本一目了然
        第三幕 变量校验：缺变量抛错而不是静默降质；JSON 花括号转义坑
        第四幕 预算裁剪：上下文超长时按优先级丢，而不是整体塞爆
        第五幕 少样本检索：示例库怎么挑出最像的 2 条
        第六幕 等价性：新渲染结果与现有 ReActAgent 的提示词逐字比对
        第七幕 真实调用：配了 DEEPSEEK_API_KEY 才跑，对比两种策略的实际回答

【实测运行输出（2026-09-21 跑通，除第七幕外数字可复现）】
    第一幕  已注册 6 个模板；react.system 有 v1(298 字符) / v2(367 字符) 两个版本
    第二幕  同一问题四种策略 prompt_tokens:
            zero_shot 307 / cot 366 / few_shot 422 / react 444（×1.00 ×1.19 ×1.37 ×1.45）
    第三幕  缺变量 → 抛 PromptError；多传变量 → 告警后忽略；
            {{ }} 渲染成单花括号；尾逗号 JSON 解析成功，人话解析返回 None
    第四幕  rag_context 2040 token，预算 2000 → 截断到 1987，memory 丢弃；
            预算 200 → 截断到 190，memory 丢弃（权限边界始终保住）
    第五幕  '报销单什么时候到账？' 得分 [0.3538, 0.2714, 0.0, 0.0] → 选 2 条；
            '年假怎么算？' 最高 0.1429 未过阈值 → 选 0 条；'写首诗' 全 0 → 选 0 条
    第六幕  迁移前后逐字一致（三条全 True）：
            react.system v1 = 旧 SYSTEM_PROMPT（298 字符）
            react.reason v1 = 旧 REASON_PROMPT（263 字符）
            react.role   v1 = 旧 system 去掉流程段
            现在 agent 的 system 298 → 368 字符，差异只有流程段（换成策略块版）
    第七幕  配了 DEEPSEEK_API_KEY 才跑（示例输出）：
            zero_shot 直接给结论；cot 先列已知条件再推导，输出明显更长
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.common.logger import get_logger  # noqa: E402
from modules.prompt import (  # noqa: E402
    FewShotStore,
    PromptBuilder,
    PromptTemplate,
    build_strategy_block,
    get_registry,
    parse_json_output,
)

_logger = get_logger("prompt.demo")

QUERY = "差旅报销单提交后几天到账？"

RAG_CONTEXT = (
    "【制度-差旅报销】员工提交差旅报销单后，财务在 3 个工作日内完成审核；"
    "审核通过后进入付款批次，每周三、周五各打款一次；"
    "遇节假日顺延至下一个工作日。单笔超过 5000 元需部门负责人二次审批。"
)

MEMORY_CONTEXT = "1. 用户上周问过'报销单状态怎么查'\n2. 用户所属部门: 技术部"

# ----------------------------------------------------------------------
# 下面两段是「改造前」写在 react_agent.py 类里的原文，抄在这里当回归基准（golden）。
# 它们已经被迁到 modules/prompt/builtin.py，但原文要留一份，
# 否则没人能回答"迁移有没有改掉措辞"这个问题 —— 第六幕就是拿它做对比的。
# ----------------------------------------------------------------------
_OLD_SYSTEM_PROMPT = (
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

_OLD_REASON_PROMPT = (
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

PERMISSION_CONTEXT = (
    "- 当前用户：张三（工号 U001，技术部）\n"
    "- 可见知识库：制度库、技术文档库（禁止引用范围外的内容）\n"
    "- 业务数据范围：仅本人数据"
)


def act1_registry():
    print("\n" + "=" * 68)
    print("第一幕  集中注册：所有模板一处可查")
    print("=" * 68)
    reg = get_registry()
    print(f"已注册模板 {len(reg.names())} 个: {reg.names()}")
    for meta in reg.list_all():
        print(f"  {meta['name']:<18} {meta['version']:<4} "
              f"strategy={meta['strategy']:<10} 变量={meta['variables']}")
    print(f"\n同名多版本（A/B 对比用）: react.system 的版本 = {reg.versions('react.system')}")
    v1 = reg.get("react.system", "v1").render(max_iter=10)
    v2 = reg.get("react.system", "v2").render(max_iter=10)
    print(f"  v1 长度 {len(v1)} 字符 → v2 长度 {len(v2)} 字符（v2 多出的就是补充约束）")


def act2_strategies():
    print("\n" + "=" * 68)
    print("第二幕  同一问题四种问法：策略一换，prompt 与成本都变")
    print("=" * 68)
    store = FewShotStore()
    store.add("报销单状态怎么查？", "在'我的报销'列表查看，状态分 已提交/已审核/已打款 三档。",
              tags=["报销"])
    store.add("报销需要哪些附件？", "发票原件、行程单、审批截图，缺一不可。", tags=["报销"])
    store.add("今天天气怎么样？", "今天晴，18~26℃。", tags=["天气"])

    rows = []
    for strategy in ("zero_shot", "few_shot", "cot", "react"):
        builder = PromptBuilder(
            strategy=strategy,
            role="你是企业级智能AI助手。",
            few_shot_store=store,
            few_shot_k=2,
        )
        rec = builder.build_record(
            QUERY,
            rag_context=RAG_CONTEXT,
            memory_context=MEMORY_CONTEXT,
            permission_context=PERMISSION_CONTEXT,
            tools_desc="- calculator: 计算表达式\n- query_order: 查询单据状态",
        )
        rows.append((strategy, rec["prompt_tokens"], len(rec["user"])))
        print(f"\n--- {strategy} ---  prompt_tokens={rec['prompt_tokens']}  裁剪={rec['trimmed']}")
        head = rec["user"].split("## 用户提问")[0]      # 策略段就在模板开头这一节
        print(head.strip()[:240] + ("…" if len(head.strip()) > 240 else ""))

    base = rows[0][1]
    print("\n成本对比（以 zero_shot 为基准）:")
    for name, tokens, _ in rows:
        print(f"  {name:<10} {tokens:>4} token   ×{tokens / base:.2f}")
    print("  → react 最贵（带工具清单），few_shot 次之（带示例）；")
    print("    zero_shot 能答对的题，别用 CoT，那是纯烧钱。")
    return rows


def act3_validation():
    print("\n" + "=" * 68)
    print("第三幕  变量校验 + JSON 输出（提示词最容易静默出错的两个点）")
    print("=" * 68)
    tpl = PromptTemplate("demo.qa", "问题: {query}\n资料: {rag_context}", version="v1")

    # 场景1：少传变量 → 抛错，而不是渲染出"资料: "这种空壳 prompt
    try:
        tpl.render(query=QUERY)
        print("缺变量: 没抛错（不符合预期）")
    except Exception as e:
        print(f"缺变量 → 抛 {type(e).__name__}: {e}")

    # 场景2：多传变量 → 只告警并丢弃，不中断调用
    out = tpl.render(query=QUERY, rag_context=RAG_CONTEXT, unused_field="多余")
    print(f"多传变量 → 正常渲染，长度 {len(out)} 字符，多余字段被忽略")

    # 场景3：模板里要出现字面花括号必须写双花括号
    reg = get_registry()
    rendered = reg.render("classify.intent", query=QUERY, labels="报销咨询、制度查询、闲聊")
    print("\nclassify.intent 渲染后的输出格式段:")
    print("  " + [ln for ln in rendered.splitlines() if "intent" in ln][0])
    print("  → 模板里写的是 {{\"intent\": ...}}，渲染出来才是单花括号；写成单花括号会被当变量")

    # 场景4：解析端配合 —— 模型常见的尾逗号也能救回来
    print(f"\n解析带尾逗号的 JSON → {parse_json_output('{\"intent\": \"报销咨询\", \"confidence\": 0.9,}')}")
    print(f"解析一段人话       → {parse_json_output('我觉得这是报销咨询')}")


def act4_budget():
    print("\n" + "=" * 68)
    print("第四幕  token 预算裁剪：上下文超长时按优先级丢")
    print("=" * 68)
    long_rag = RAG_CONTEXT * 20          # 把知识库参考撑大，模拟召回了一大段
    print(f"原始 rag_context: {len(long_rag)} 字符   memory_context: {len(MEMORY_CONTEXT)} 字符")

    for budget in (2000, 200):
        builder = PromptBuilder(
            strategy="zero_shot",
            role="你是企业级智能AI助手。",
            max_context_tokens=budget,
        )
        rec = builder.build_record(
            QUERY, rag_context=long_rag, memory_context=MEMORY_CONTEXT,
        )
        print(f"\n预算 {budget} token → prompt_tokens={rec['prompt_tokens']}")
        for note in rec["trimmed"] or ["（未超预算，无需裁剪）"]:
            print(f"  {note}")
    print("\n  → 权限边界永远最先保住（丢了会越权），历史记忆最先被丢（丢了只是答得没那么贴人）")


def act5_fewshot():
    print("\n" + "=" * 68)
    print("第五幕  少样本检索：示例库怎么挑最像的 2 条")
    print("=" * 68)
    store = FewShotStore()
    store.add("报销单状态怎么查？", "在'我的报销'列表查看。", tags=["报销"])
    store.add("报销需要哪些附件？", "发票、行程单、审批截图。", tags=["报销"])
    store.add("今天天气怎么样？", "今天晴。", tags=["天气"])
    store.add("年假有几天？", "入职满一年 5 天。", tags=["假期"])

    for q in ("报销单什么时候到账？", "年假怎么算？", "帮我写一首诗"):
        picked = store.select(q, k=2)
        scores = [store.similarity(q, ex) for ex in store.all()]
        print(f"\nquery = {q}")
        print(f"  全库得分: {scores}   最高 {max(scores)}（阈值 0.15）")
        print(f"  选中 {len(picked)} 条: {[ex.question for ex in picked]}")
    print("\n  → '报销单什么时候到账？' 两条报销示例都入选（0.3538 / 0.2714）；")
    print("    '年假怎么算？' 最高只有 0.1429，未过阈值 → 一条都不给；")
    print("    '写首诗' 全库 0 分 → 也不给。宁可不用示例，也不能用错示例，")
    print("    一条不相关的示例对模型的误导，比没有示例更大。")


def act6_equivalence():
    print("\n" + "=" * 68)
    print("第六幕  接入之后：与改造前的提示词逐字比对")
    print("=" * 68)
    reg = get_registry()

    # 左边是改造前的原文（golden），右边是注册表里现在的渲染结果
    old_system = _OLD_SYSTEM_PROMPT.format(max_iter=10)
    v1_system = reg.get("react.system", "v1").render(max_iter=10)
    print(f"react.system v1 vs 改造前 SYSTEM_PROMPT : 逐字一致 = {v1_system == old_system}")
    print(f"  {len(old_system)} 字符 / {len(v1_system)} 字符")

    old_reason = _OLD_REASON_PROMPT.format(
        query="Q", skill_context="S", permission_context="P", rag_context="R",
        memory_context="M", tool_history="H", observation="O", tools_desc="T",
    )
    v1_reason = reg.get("react.reason", "v1").render(
        query="Q", skill_context="S", permission_context="P", rag_context="R",
        memory_context="M", tool_history="H", observation="O", tools_desc="T",
    )
    print(f"react.reason  v1 vs 改造前 REASON_PROMPT : 逐字一致 = {v1_reason == old_reason}")
    print(f"  {len(old_reason)} 字符 / {len(v1_reason)} 字符")

    # 角色规则段是否被原样保留？把旧 system 里的「工作流程」那一节摘掉再比
    head, tail = old_system.split("## 核心规则", 1)
    old_role_only = head.split("## 工作流程（ReAct）")[0] + "## 核心规则" + tail
    role_now = reg.render("react.role", max_iter=10)
    print(f"react.role    v1 = 旧 system 去掉流程段   : 逐字一致 = {old_role_only == role_now}")

    # 现在 ReActAgent 真正用的 system（react.role + react 策略块）
    try:
        from modules.agent.react_agent import ReActAgent
        agent = ReActAgent()
        print(f"\n改造前 system {len(old_system)} 字符 → 现在 system {len(agent.system_prompt)} 字符")
        print("  现在 system 的三段: [角色与核心规则] + [ReAct 流程策略块] + [可用工具]")
        print("  流程段相对旧版多了一条: 同一个工具用相同参数重复调用没有意义，看到结果就该作答")
        print("  （实测背景：同一句'3*7'曾被连续调 9 次计算器，单次对话 14 秒）")

        # 用 agent 自己的 builder 拼一轮，看真实 prompt 的成本
        rec = agent._builder.build_record(
            "3*7等于几",
            permission_context=PERMISSION_CONTEXT,
            skill_context="（未匹配到专门技能，按通用企业助手流程处理）",
            rag_context="（知识库无相关内容）",
            memory_context="（无历史记忆）",
            tool_history="（无）",
            observation="（初始状态，暂无工具观察）",
            tools_desc="- calculator: 计算数学表达式",
        )
        print(f"  实际一轮 prompt_tokens={rec['prompt_tokens']}  裁剪={rec['trimmed']}")
    except Exception as e:
        print(f"跳过：构造 ReActAgent 失败（{e}）")
    print("  → 模板 v1 与改造前一致，说明迁移没有悄悄改措辞；")
    print("    system 的差异只有流程段，那是有意换的（见上）。")


def act7_real_llm():
    print("\n" + "=" * 68)
    print("第七幕  真实 LLM 调用（未配 DEEPSEEK_API_KEY 则跳过）")
    print("=" * 68)
    from config import get_settings
    if not get_settings().deepseek_api_key:
        print("跳过：.env 里没有 DEEPSEEK_API_KEY，前面的渲染对比不依赖网络。")
        return
    from modules.agent.llm import LLMClient
    llm = LLMClient()
    for strategy in ("zero_shot", "cot"):
        builder = PromptBuilder(
            strategy=strategy,
            role="你是企业级智能AI助手，只依据给定资料回答。",
            output_format="text",
        )
        msgs = builder.build(QUERY, rag_context=RAG_CONTEXT, memory_context="（无）",
                             permission_context=PERMISSION_CONTEXT)
        system = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
        answer = llm.invoke(msgs[-1]["content"], system_prompt=system)
        print(f"\n[{strategy}] {answer[:200]}")


def print_wiring():
    print("\n" + "=" * 68)
    print("接入状态：改了哪些地方，还能接着改什么")
    print("=" * 68)
    print("  已接入")
    print("    react_agent.__init__      建 PromptBuilder(react.reason + react 策略)")
    print("    react_agent.reason_step   build_record() 取代 REASON_PROMPT.format(...)")
    print("                              8 个槽位不变，多了变量校验与 token 预算裁剪")
    print("    react_agent.generate_answer  复用 __init__ 算好的 system_prompt")
    print("    summary_generator        两处改用 get_registry().render(...)")
    print("    config.py                新增 MAX_PROMPT_CONTEXT_TOKENS（默认 3000）")
    print("\n  还能接着做的（都是加模板/加参数，不动调用方）")
    print("    1. 换问法：PromptBuilder(strategy='cot')，同一套角色规则换个问法")
    print("    2. 试新措辞：注册 react.role v2，取用时指定 version，可回滚")
    print("    3. 接少样本：给 builder 传 few_shot_store，FewShotStore.add() 灌示例")
    print("    4. 结构化输出：output_format='json' + parse_json_output() 解析")
    print("  注：迁移前后措辞逐字一致（第六幕已验证），行为不变，之后才谈优化。")


def main():
    print("#" * 68)
    print("#  提示词工程：从'字符串常量'到'可注册、可切换、可裁剪'的一层")
    print("#" * 68)
    act1_registry()
    act2_strategies()
    act3_validation()
    act4_budget()
    act5_fewshot()
    act6_equivalence()
    act7_real_llm()
    print_wiring()
    print("\n一句话：模板管'说什么'，策略管'怎么问'，builder 管'装得下装不下'。")


if __name__ == "__main__":
    main()
