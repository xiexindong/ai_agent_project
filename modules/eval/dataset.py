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
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
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

    id: str
    query: str
    tags: List[str] = field(default_factory=list)
    expect_skill: str = ""
    expect_tools: List[str] = field(default_factory=list)
    expect_intent: str = ""
    expect_risk_pass: bool = True
    expect_branch: str = ""
    must_contain: List[str] = field(default_factory=list)
    must_not_contain: List[str] = field(default_factory=list)
    note: str = ""

    # 【什么】转成普通 dict
    # 【作用】写 JSONL 与报告明细都用它
    def to_dict(self) -> Dict:
        return asdict(self)

    # 【什么】从 dict 还原用例
    # 【作用】忽略未知字段，方便以后给用例加新维度而不破坏旧文件
    @classmethod
    def from_dict(cls, data: Dict) -> "EvalCase":
        known = {f for f in cls.__dataclass_fields__}          # 只认已声明的字段
        return cls(**{k: v for k, v in data.items() if k in known})

    # 【什么】列出本条实际启用了哪些校验维度
    # 【作用】报告里标出"这条只评了技能"，避免拿不同维度的用例横向比总分
    def active_checks(self) -> List[str]:
        checks = ["skill", "tools", "risk"]                    # 这三项永远参与
        if self.expect_intent:
            checks.append("intent")
        if self.expect_branch:
            checks.append("branch")
        if self.must_contain or self.must_not_contain:
            checks.append("answer")
        return checks


# 【什么】从 JSONL 文件加载用例集
# 【作用】跳过空行与 # 开头的注释行；缺 id 的自动补行号
def load_cases(path: str | Path) -> List[EvalCase]:
    """加载 JSONL 用例文件。

    【为什么不抛异常而是跳过坏行】
        用例文件是人工维护的，一行写错不该让整个评测跑不起来。
        坏行记 warning 后跳过，缺 id 的行用行号补一个，保证后续统计不串。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"评测用例文件不存在: {p}")

    cases: List[EvalCase] = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[warn] 第 {lineno} 行不是合法 JSON，已跳过: {e}")
            continue
        if not isinstance(data, dict):
            print(f"[warn] 第 {lineno} 行不是对象，已跳过")
            continue
        if not data.get("id"):
            data["id"] = f"line-{lineno}"
        if not data.get("query"):
            print(f"[warn] 第 {lineno} 行缺少 query，已跳过")
            continue
        cases.append(EvalCase.from_dict(data))
    return cases


# 【什么】把用例集写回 JSONL
# 【作用】新增/修正用例后落盘，保持一行一条便于 diff
def save_cases(cases: List[EvalCase], path: str | Path) -> int:
    p = Path(path)
    lines = [
        json.dumps(c.to_dict(), ensure_ascii=False)
        for c in cases
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


# 【什么】用例集自检
# 【作用】查重 id 与空集，返回问题列表；空列表表示健康
def validate_cases(cases: List[EvalCase]) -> List[str]:
    """返回问题描述列表（空 = 健康）。

    【为什么必须查重 id】
        基线对比是按 id 对齐的：两条同 id 用例会让"这次比上次掉了多少"算错，
        而且错得很隐蔽 —— 看起来是效果回归，实际是数据重复。
    """
    problems: List[str] = []
    seen: Dict[str, int] = {}
    for c in cases:
        seen[c.id] = seen.get(c.id, 0) + 1
    for cid, n in seen.items():
        if n > 1:
            problems.append(f"用例 id 重复: {cid} × {n}")
    if not cases:
        problems.append("用例集为空")
    return problems


# 【什么】按标签筛选用例
# 【作用】只想跑某一类（如 --tag risk）时用，支持多标签任一命中
def filter_by_tag(cases: List[EvalCase], tags: Optional[List[str]]) -> List[EvalCase]:
    if not tags:
        return cases
    wanted = {t.strip() for t in tags if t.strip()}
    return [c for c in cases if wanted & set(c.tags)]
