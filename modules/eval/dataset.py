"""
评测数据集定义
==============
一条用例 = 一句提问 + 一组「应该发生什么」的断言。

【为什么期望值要写死而不是迁就实现】
    评测的价值在于「暴露差距」，不是「证明当前实现对」。
    所以 expect_* 一律按「正确答案」写；实现没达到就是缺陷，
    体现在报告里就是分数低 —— 那正是要修的东西，不能反过来改期望值凑分。

【用例的六个可校验维度】
    expect_skill       技能路由应命中哪个技能（"" = 应走通用流程，不命中任何技能）
    expect_tools       工具名单应包含哪些工具（子集匹配：期望 ⊆ 实际即算召回成功）
    expect_intent      意图分类应落在哪一类（空串 = 本条不校验意图）
    expect_risk_pass   风控应放行(True)还是拦截(False)
    expect_branch      流水线最终支路（空串 = 不校验）
    must_contain /
    must_not_contain   仅在线档校验：答案里必须/不得出现的片段

【三档运行模式与用例的关系】
    offline  只跑确定性维度（技能 L1 / 工具关键词召回 / 风控 / 意图规则 / 提示词组装）
    online   追加端到端答案校验（must_contain）与 LLM-as-judge
    离线档保证可复现、无外部依赖，适合做 CI 回归门禁。

【JSONL 而不是 JSON 的理由】
    一行一条，diff 友好：新增用例只多一行，评审时看得出改了哪条，
    也不像大 JSON 那样容易产生整文件冲突。
"""
from __future__ import annotations  # 启用注解延迟求值：支持 str | Path 这类新写法在低版本语法树上安全存在

import json  # 标准库 JSON：JSONL 逐行解析/序列化用
from dataclasses import dataclass, field, asdict  # 数据类工具；asdict 用于一键序列化
from pathlib import Path  # 跨平台路径对象
from typing import Dict, List, Optional  # 泛型类型注解工具


# 【什么】一条评测用例的数据结构
# 【作用】贯穿"加载 → 执行 → 判定 → 报告"全链路的统一数据单元
# 【为什么】用 dataclass 而不是裸 dict：字段有默认值、有类型标注，
#           新增维度时旧文件仍可读（from_dict 忽略未知字段），双向兼容
@dataclass  # 自动生成 __init__/__repr__ 等样板代码
class EvalCase:
    """一条评测用例。

    Attributes:
        id:                唯一标识，建议「类别-序号」，如 calc-01
        query:             用户提问原文
        tags:              分类标签，用于按组统计（报告里的 by_tag 分组）
        expect_skill:      期望命中的技能名；"" 表示不应命中任何技能
        expect_tools:      期望出现在工具名单里的工具（子集匹配）
        expect_intent:     期望的意图大类；"" 表示本条不校验
        expect_risk_pass:  True=应放行；False=应被风控拦截
        expect_branch:     期望的最终支路；"" 表示不校验
        must_contain:      在线档：答案必须包含这些片段（任一命中即算通过）
        must_not_contain:  在线档：答案不得包含这些片段（任一命中即算失败）
        note:              给人看的备注，说明这条用例在测什么
    """

    id: str  # 唯一标识：基线对比按它对齐
    query: str  # 用户提问原文
    tags: List[str] = field(default_factory=list)  # 分组标签；用 field 工厂防止共享可变默认值
    expect_skill: str = ""  # 期望技能名；空串 = 应走通用流程
    expect_tools: List[str] = field(default_factory=list)  # 期望工具子集；空 = 本条不校验工具
    expect_intent: str = ""  # 期望意图；空串 = 本条不校验
    expect_risk_pass: bool = True  # 期望风控放行与否；默认放行（大多数用例是正常提问）
    expect_branch: str = ""  # 期望支路；空串 = 不校验
    must_contain: List[str] = field(default_factory=list)  # 在线档：答案必含片段
    must_not_contain: List[str] = field(default_factory=list)  # 在线档：答案禁含片段
    note: str = ""  # 给人看的备注

    # 【什么】转成普通 dict
    # 【作用】写 JSONL 与报告明细都用它
    # 【为什么】asdict 会递归展开嵌套 dataclass，且字段顺序与声明一致，落盘稳定可 diff
    def to_dict(self) -> Dict:  # 序列化出口
        return asdict(self)  # 标准库 asdict 一键展开

    # 【什么】从 dict 还原用例
    # 【作用】忽略未知字段，方便以后给用例加新维度而不破坏旧文件
    # 【为什么】用例文件是长期资产，结构必须向前兼容：老文件不认识的字段跳过，
    #           新字段有默认值兜底，新旧版本双向都炸不了
    @classmethod  # 工厂方法：从类调用而不是实例
    def from_dict(cls, data: Dict) -> "EvalCase":  # 反序列化入口
        known = {f for f in cls.__dataclass_fields__}          # 只认已声明的字段
        return cls(**{k: v for k, v in data.items() if k in known})  # 过滤掉未知字段后按关键字参数构造

    # 【什么】列出本条实际启用了哪些校验维度
    # 【作用】报告里标出"这条只评了技能"，避免拿不同维度的用例横向比总分
    # 【为什么】不同用例的校验面不同，混在一起比总分会产生"这条要求多所以分低"的误读
    def active_checks(self) -> List[str]:  # 返回启用的维度名列表
        checks = ["skill", "tools", "risk"]                    # 这三项永远参与
        if self.expect_intent:  # 声明了期望意图
            checks.append("intent")  # 意图维度启用
        if self.expect_branch:  # 声明了期望支路
            checks.append("branch")  # 支路维度启用
        if self.must_contain or self.must_not_contain:  # 声明了答案片段约束
            checks.append("answer")  # 答案维度启用
        return checks  # 返回维度清单


# 【什么】从 JSONL 文件加载用例集
# 【作用】跳过空行与 # 开头的注释行；缺 id 的自动补行号
# 【为什么】用例文件由人工维护，容错优先：坏行警告跳过而不是让整批评测跑不起来
def load_cases(path: str | Path) -> List[EvalCase]:
    """加载 JSONL 用例文件。

    【为什么不抛异常而是跳过坏行】
        用例文件是人工维护的，一行写错不该让整个评测跑不起来。
        坏行记 warning 后跳过，缺 id 的行用行号补一个，保证后续统计不串。
    """
    p = Path(path)  # 统一成 Path 对象，兼容 str 入参
    if not p.exists():  # 文件不存在是配置错误而不是数据质量问题 → 直接抛
        raise FileNotFoundError(f"评测用例文件不存在: {p}")  # 早失败，报错信息说人话

    cases: List[EvalCase] = []  # 结果列表
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):  # 按 UTF-8 读全文再逐行枚举；行号从 1 起，与编辑器显示一致
        line = raw.strip()  # 去首尾空白：空行与缩进都归一
        if not line or line.startswith("#"):  # 空行与注释行
            continue  # 跳过
        try:  # 单行 JSON 解析
            data = json.loads(line)  # 解析本行
        except json.JSONDecodeError as e:  # JSON 语法错误
            print(f"[warn] 第 {lineno} 行不是合法 JSON，已跳过: {e}")  # 警告并带行号，方便人去修
            continue  # 跳过坏行
        if not isinstance(data, dict):  # 解析出来不是对象（如数组/标量）
            print(f"[warn] 第 {lineno} 行不是对象，已跳过")  # 警告
            continue  # 跳过
        if not data.get("id"):  # 缺 id 或 id 为空
            data["id"] = f"line-{lineno}"  # 用行号补一个：保证后续基线对齐有锚点
        if not data.get("query"):  # 缺 query：这条没法跑
            print(f"[warn] 第 {lineno} 行缺少 query，已跳过")  # 警告
            continue  # 跳过
        cases.append(EvalCase.from_dict(data))  # 反序列化入列（未知字段被忽略）
    return cases  # 返回用例集


# 【什么】把用例集写回 JSONL
# 【作用】新增/修正用例后落盘，保持一行一条便于 diff
# 【为什么】ensure_ascii=False 保留中文原文：用例文件是给人看给人改的，
#           转成 \uXXXX 转义序列就没法评审了
def save_cases(cases: List[EvalCase], path: str | Path) -> int:  # 返回写入条数
    p = Path(path)  # 统一成 Path 对象
    lines = [  # 逐条序列化成一行 JSON
        json.dumps(c.to_dict(), ensure_ascii=False)  # 保留中文不转义
        for c in cases  # 遍历用例
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 换行连接、末尾补换行（POSIX 习惯），按 UTF-8 落盘
    return len(lines)  # 返回写入条数，供调用方打印确认


# 【什么】用例集自检
# 【作用】查重 id 与空集，返回问题列表；空列表表示健康
# 【为什么】数据质量问题要在跑评测之前暴露：跑完才发现 id 重复，
#           基线对比已经算错了，浪费一轮时间
def validate_cases(cases: List[EvalCase]) -> List[str]:
    """返回问题描述列表（空 = 健康）。

    【为什么必须查重 id】
        基线对比是按 id 对齐的：两条同 id 用例会让"这次比上次掉了多少"算错，
        而且错得很隐蔽 —— 看起来是效果回归，实际是数据重复。
    """
    problems: List[str] = []  # 问题列表：空 = 健康
    seen: Dict[str, int] = {}  # id → 出现次数 的计数表
    for c in cases:  # 逐条计数
        seen[c.id] = seen.get(c.id, 0) + 1  # 计数 +1
    for cid, n in seen.items():  # 逐个 id 检查
        if n > 1:  # 出现超过一次
            problems.append(f"用例 id 重复: {cid} × {n}")  # 记录重复问题
    if not cases:  # 整个集合为空
        problems.append("用例集为空")  # 记录空集问题
    return problems  # 返回问题列表


# 【什么】按标签筛选用例
# 【作用】只想跑某一类（如 --tag risk）时用，支持多标签任一命中
# 【为什么】回归全量跑要几分钟，改完一个局部只想验证相关标签的用例，
#           筛选能让反馈循环快一个量级
def filter_by_tag(cases: List[EvalCase], tags: Optional[List[str]]) -> List[EvalCase]:  # 入参：全量用例与标签过滤条件
    if not tags:  # 没给过滤条件
        return cases  # 原样返回
    wanted = {t.strip() for t in tags if t.strip()}  # 条件归一化：去空白、去空串、转集合求 O(1) 命中
    return [c for c in cases if wanted & set(c.tags)]  # 集合求交非空即命中（任一标签匹配即保留）
