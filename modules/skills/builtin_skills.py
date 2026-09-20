"""
内置技能集
==========
开箱即用的 4 个技能，演示 Skill（SOP + 工具绑定）与 Function Call 的协作方式：

1. MathCalculatorSkill  数学计算技能  → 绑定 calculator 函数工具
2. BusinessDataSkill    业务数据查询  → 绑定 data_query 函数工具
3. KnowledgeQASkill     知识库问答    → 走 RAG 检索，无专用函数工具
4. WebSearchSkill       联网信息检索  → 绑定 MCP web_search / get_current_time 函数工具

新增技能：继承 BaseSkill 声明类属性，然后加入本文件 get_default_skills() 即可被自动装配。
"""
from typing import List
from modules.skills.base_skill import BaseSkill


class MathCalculatorSkill(BaseSkill):
    """数学计算技能：把自然语言问题转成表达式，调用计算器函数完成精确计算。"""

    name = "math_calculator"
    display_name = "数学计算"
    description = "处理四则运算、幂运算、三角函数、开方、取模等各类数学计算问题，保证结果精确。"
    trigger_keywords = [
        "计算", "算一下", "算一算", "等于几", "等于多少", "结果是多少",
        "加", "减", "乘", "除", "乘以", "除以",
        "平方", "开方", "次方", "根号",
        "sin", "cos", "tan", "sqrt", "对数",
        "数学", "百分数", "百分比", "求和",
    ]
    bound_tools = ["calculator"]
    priority = 10
    instructions = (
        "1. 从用户问题中识别已知数值与运算关系，转换为合法数学表达式（如 '2加3' → '2+3'）；\n"
        "2. 必须调用 calculator 函数完成计算，禁止凭记忆心算给出数值结果；\n"
        "3. 向用户展示「计算式 → 结果」，带单位的问题需注明单位；\n"
        "4. 若表达式无法唯一确定（如缺少必要数值），先向用户追问澄清，不要猜测参数。"
    )


class BusinessDataSkill(BaseSkill):
    """企业业务数据查询技能：按 SOP 调用 data_query 函数查询用户/订单/库存。"""

    name = "business_data_query"
    display_name = "企业数据查询"
    description = "查询企业内部业务数据：用户信息、订单状态、库存数量与价格等。"
    trigger_keywords = [
        "订单", "用户信息", "库存", "查一下", "查询",
        "客户", "销量", "销售额", "发货", "配送", "付款",
        "单价", "价格", "还有多少", "存货", "业务数据",
    ]
    bound_tools = ["data_query"]
    priority = 10
    instructions = (
        "1. 先明确查询的数据类型（users 用户 / orders 订单 / inventory 库存）与记录ID；\n"
        "2. 调用 data_query 函数获取数据，record_id 不明确时可先查该类型全部记录ID再精查；\n"
        "3. 用表格或条目形式结构化展示结果，金额保留两位小数；\n"
        "4. 查不到记录时如实告知，禁止编造业务数据；涉及敏感信息按脱敏结果展示。"
    )


class KnowledgeQASkill(BaseSkill):
    """知识库问答技能：依据 RAG 召回内容回答，强调引用来源、无据不答。"""

    name = "knowledge_qa"
    display_name = "知识库问答"
    description = "基于企业知识库（制度文档、产品手册、FAQ）回答问题，答案附来源。"
    trigger_keywords = [
        "根据公司", "公司规定", "制度", "手册", "文档里",
        "知识库", "政策", "流程是怎样", "规范", "FAQ",
        "条款", "报销标准", "年假", "操作指南", "产品说明",
    ]
    bound_tools = []
    priority = 5
    instructions = (
        "1. 严格以【知识库参考】中召回的内容为事实依据回答，不得引入库外信息或编造；\n"
        "2. 回答中标注信息来源文件名与相关度，多来源时分点列出；\n"
        "3. 知识库无相关内容或内容明显不足时，明确告知「知识库未检索到相关规定」，\n"
        "   可建议用户补充关键词或转人工，不要用通用常识冒充制度结论；\n"
        "4. 涉及制度条款解读时，原文引用关键句后再给解释。"
    )


class WebSearchSkill(BaseSkill):
    """联网信息检索技能：时效性问题调用 MCP 搜索函数；时间问题调用报时函数。"""

    name = "web_search"
    display_name = "联网信息检索"
    description = "获取互联网实时信息：新闻、天气、最新政策、近期事件，以及当前时间查询。"
    trigger_keywords = [
        "最新", "新闻", "网上", "联网", "搜索一下", "帮我搜",
        "实时", "今天新闻", "现在几点", "当前时间", "天气预报",
        "最近发生", "最新消息", "网上怎么说",
    ]
    bound_tools = ["web_search", "get_current_time"]
    priority = 8
    instructions = (
        "1. 判断需求：事实/资讯类调用 web_search 函数，时间类调用 get_current_time 函数；\n"
        "2. 搜索关键词使用 2~4 个核心词，必要时分多次搜索交叉验证；\n"
        "3. 回答基于搜索返回的内容，并附来源链接；信息之间有冲突时说明分歧；\n"
        "4. 搜索工具不可用（未配置 Key 或服务降级）时，明确告知无法联网，不要编造实时信息。"
    )


def get_default_skills() -> List[BaseSkill]:
    """内置默认技能集（注意：返回的是实例，供 SkillRegistry 注册）。"""
    return [
        MathCalculatorSkill(),
        BusinessDataSkill(),
        KnowledgeQASkill(),
        WebSearchSkill(),
    ]
