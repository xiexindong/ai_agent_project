"""
评测指标计算
============
把「每条用例的实际输出 vs 期望」折算成可比较、可归档的分数。

【为什么工具名单用集合 F1 而不是简单对错】
    名单型输出有三种错法，混在一起看不出问题在哪：
        漏召   该带的没带（用户问订单却没给 data_query）→ 召回率低
        多带   带了一堆无关工具 → 精确率低，白白烧 token 且容易误调用
        都对
    精确率与召回率分开统计，才能判断该调 min_score 还是调 top_k。

【为什么每个指标都带 total/passed】
    光看 0.85 不知道是 17/20 还是 34/40。小样本上 0.85 的置信度完全不一样，
    归档与对比时必须保留分母，否则两次运行无法判断差异是否显著。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# 【什么】集合的精确率 / 召回率 / F1
# 【作用】工具名单的匹配度量；expected 为空集时精确率记为 1.0（没期待就不该怪它多带）
def prf(expected: Set[str], actual: Set[str]) -> Tuple[float, float, float]:
    """返回 (precision, recall, f1)。

    【边界】expected 为空：召回率记 1.0（不期待任何东西，没漏），
            精确率记 1.0（名单里带什么都不算错，因为本条不校验工具）；
            两者都为空 → 返回 (1.0, 1.0, 1.0)。
    """
    if not expected and not actual:
        return 1.0, 1.0, 1.0
    if not expected:
        return 1.0, 1.0, 1.0
    if not actual:
        return 1.0, 0.0, 0.0
    hit = len(expected & actual)
    precision = hit / len(actual)
    recall = hit / len(expected)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


@dataclass
class Metric:
    """一个指标的汇总结果。

    Attributes:
        name:    指标名，如 skill_acc
        passed:  通过条数
        total:   参与统计的条数
        score:   passed/total，保留 4 位
        errors:  执行异常导致无法判分的条数（不计入 total）
    """

    name: str
    passed: int = 0
    total: int = 0
    errors: int = 0
    score: float = 0.0

    # 【什么】累加一次判定结果
    # 【作用】ok=None 表示本条不适用/执行异常，只记 errors 不进分母
    def add(self, ok: Optional[bool]) -> None:
        if ok is None:
            self.errors += 1
            return
        self.total += 1
        if ok:
            self.passed += 1
        self.score = round(self.passed / self.total, 4) if self.total else 0.0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "total": self.total,
            "score": self.score,
            "errors": self.errors,
        }


@dataclass
class ScoreBoard:
    """一块记分板：按维度累加所有用例的判定结果。

    【为什么用累加器而不是先收集再统计】
        用例可能上万条，先存全量明细再统计会吃掉内存；
        累加器只需 O(维度数) 的内存，明细只保留失败项（报告要看的）。
    """

    skill: Metric = field(default_factory=lambda: Metric("skill_acc"))
    intent: Metric = field(default_factory=lambda: Metric("intent_acc"))
    risk: Metric = field(default_factory=lambda: Metric("risk_acc"))
    branch: Metric = field(default_factory=lambda: Metric("branch_acc"))
    answer: Metric = field(default_factory=lambda: Metric("answer_quality"))

    _tool_p: List[float] = field(default_factory=list)      # 每条用例的精确率
    _tool_r: List[float] = field(default_factory=list)      # 每条用例的召回率
    _tool_f1: List[float] = field(default_factory=list)
    _prompt_tokens: List[int] = field(default_factory=list)
    _budget_ok: List[bool] = field(default_factory=list)
    _latency_ms: List[float] = field(default_factory=list)

    # 【什么】记一次工具名单比对
    # 【作用】expected 为空的用例不参与统计（那条本来就不校验工具）
    def add_tools(self, expected: Set[str], actual: Set[str]) -> Tuple[float, float, float]:
        if not expected:
            return 1.0, 1.0, 1.0
        p, r, f = prf(expected, actual)
        self._tool_p.append(p)
        self._tool_r.append(r)
        self._tool_f1.append(f)
        return p, r, f

    def add_prompt(self, tokens: int, within_budget: bool, latency_ms: float = 0.0) -> None:
        self._prompt_tokens.append(tokens)
        self._budget_ok.append(within_budget)
        if latency_ms > 0:
            self._latency_ms.append(latency_ms)

    # 【什么】汇总成可归档的 dict
    # 【作用】这份结构就是基线文件的内容，也是报告的数据源
    def summary(self) -> Dict:
        def avg(xs: List[float]) -> float:
            return round(sum(xs) / len(xs), 4) if xs else 0.0

        return {
            "skill_acc": self.skill.to_dict(),
            "intent_acc": self.intent.to_dict(),
            "risk_acc": self.risk.to_dict(),
            "branch_acc": self.branch.to_dict(),
            "answer_quality": self.answer.to_dict(),
            "tool_precision": {
                "name": "tool_precision", "score": avg(self._tool_p),
                "total": len(self._tool_p), "passed": 0, "errors": 0,
            },
            "tool_recall": {
                "name": "tool_recall", "score": avg(self._tool_r),
                "total": len(self._tool_r), "passed": 0, "errors": 0,
            },
            "tool_f1": {
                "name": "tool_f1", "score": avg(self._tool_f1),
                "total": len(self._tool_f1), "passed": 0, "errors": 0,
            },
            "prompt": {
                "avg_tokens": round(sum(self._prompt_tokens) / len(self._prompt_tokens), 1)
                if self._prompt_tokens else 0,
                "max_tokens": max(self._prompt_tokens) if self._prompt_tokens else 0,
                "budget_ok_rate": round(
                    sum(1 for b in self._budget_ok if b) / len(self._budget_ok), 4
                ) if self._budget_ok else 0.0,
                "avg_latency_ms": round(avg(self._latency_ms), 1),
            },
        }


# 【什么】两次运行结果的逐指标对比
# 【作用】标出改善 / 回归 / 持平，供回归门禁判定是否放行
def diff_summary(base: Dict, cur: Dict, tolerance: float = 0.005) -> List[Dict]:
    """对比两份 summary，返回每个指标的变化。

    Args:
        base:      基线 summary
        cur:       本次 summary
        tolerance: 小于该差值视为持平，避免浮点噪声造成"假装回归"

    Returns:
        [{"metric":..., "base":..., "cur":..., "delta":..., "status":...}]
        status ∈ {"improved", "regressed", "same", "new"}
    """
    keys = ["skill_acc", "intent_acc", "risk_acc", "branch_acc",
            "answer_quality", "tool_precision", "tool_recall", "tool_f1"]
    out: List[Dict] = []
    for k in keys:
        b = base.get(k, {}).get("score")
        c = cur.get(k, {}).get("score")
        if b is None:
            out.append({"metric": k, "base": None, "cur": c, "delta": None, "status": "new"})
            continue
        delta = round((c or 0) - (b or 0), 4)
        if delta > tolerance:
            status = "improved"
        elif delta < -tolerance:
            status = "regressed"
        else:
            status = "same"
        out.append({"metric": k, "base": b, "cur": c, "delta": delta, "status": status})

    # ---- token 与延迟：越高越差，且必须用相对阈值 ----
    # 【为什么这里不能用 tolerance 那个绝对差值】
    #   耗时受机器负载、CPU 调度、冷热缓存影响，同一份代码两次跑差 50% 很常见
    #   （实测 12.5ms → 19.5ms，纯噪声）。用绝对差值会把每次运行都判成回归，
    #   门禁天天红灯，最后没人看它 —— 那才是门禁真正的死法。
    #   所以这两项按"相对变化超过 X% 才算数"判定：token 10%、延迟 25%。
    bp = base.get("prompt", {})
    cp = cur.get("prompt", {})
    for field_name, rel_tol in (("avg_tokens", 0.10), ("avg_latency_ms", 0.25)):
        b, c = bp.get(field_name, 0) or 0, cp.get(field_name, 0) or 0
        delta = round(c - b, 2)
        if not b:
            status = "same"
        elif abs(delta) / abs(b) < rel_tol:
            status = "same"
        else:
            status = "regressed" if delta > 0 else "improved"
        out.append({"metric": field_name, "base": b, "cur": c, "delta": delta, "status": status})

    b_rate = bp.get("budget_ok_rate", 0)
    c_rate = cp.get("budget_ok_rate", 0)
    d_rate = round((c_rate or 0) - (b_rate or 0), 4)
    out.append({
        "metric": "budget_ok_rate", "base": b_rate, "cur": c_rate, "delta": d_rate,
        "status": "regressed" if d_rate < -tolerance else ("improved" if d_rate > tolerance else "same"),
    })
    return out
