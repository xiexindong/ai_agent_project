"""
评测运行器
==========
逐条跑用例、逐维度判分、汇总成一块记分板，并保留失败明细。

【为什么要把「判定」和「执行」分开】
    执行（Sut.route 等）可能因为环境原因失败，判定可能因为期望写得不对而误判。
    两者混在一起，报告里只会看到"分数低"，看不出是该修代码还是该修用例。
    所以每条用例都记录 ok / error 两个独立状态：
        error != ""  → 执行炸了，是环境问题或代码缺陷
        ok = False   → 执行成功但不符合期望，是效果问题

【为什么失败明细只留前 N 条】
    全量明细在用例上千时是几 MB，报告没人看。默认留 50 条足够定位，
    --verbose 时才输出全量，兼顾排查与体积。
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from modules.eval.dataset import EvalCase
from modules.eval.harness import Sut
from modules.eval.metrics import ScoreBoard


# 单条用例最多保留多少失败维度说明
_MAX_FAIL_DETAIL = 50


@dataclass
class CaseResult:
    """一条用例的执行与判定结果。"""

    id: str
    query: str
    tags: List[str] = field(default_factory=list)
    dims: Dict[str, Dict] = field(default_factory=dict)   # {维度: {expected, actual, ok}}
    actual: Dict[str, Any] = field(default_factory=dict)  # 原始输出，排查用
    error: str = ""                                       # 执行异常；空 = 执行正常
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """所有参与判定的维度都通过才算通过。

        【为什么写 `is not False` 而不是直接 `all(...)`】
            维度有三种状态：True 通过 / False 未通过 / None 本条不适用
            （如闲聊用例不校验工具名单）。
            `all(d["ok"] ...)` 会把 None 当假，于是"不校验工具"的闲聊用例
            永远判失败 —— 实测 4 条闲聊全红，而它们实际上全对。
            正确口径是：只要没有明确的 False 就算通过。
        """
        return all(d.get("ok") is not False for d in self.dims.values()) and not self.error

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "query": self.query, "tags": self.tags,
            "dims": self.dims, "actual": self.actual, "error": self.error,
            "latency_ms": self.latency_ms, "ok": self.ok,
        }


class EvalRunner:
    """评测运行器：拿一套 Sut 跑一批用例。

    Args:
        sut:            被测对象（见 harness.build_sut）
        check_prompt:   是否每条都做一次提示词组装评测（测预算与裁剪）
        max_context_tokens: 提示词预算上限，用于判定是否守住预算
    """

    def __init__(self, sut: Sut, check_prompt: bool = True,
                 max_context_tokens: int = 3000):
        self.sut = sut
        self.check_prompt = check_prompt
        self.max_context_tokens = max_context_tokens

    # ------------------------------------------------------------------
    # 【单条用例的判定流程】
    #   跑四个面 → 逐维度比对期望 → 记进记分板
    #   任一环节抛异常都记成 error，不中断整批评测。
    # ------------------------------------------------------------------
    def run_case(self, case: EvalCase) -> CaseResult:
        dims: Dict[str, Dict] = {}
        actual: Dict[str, Any] = {}
        total_ms = 0.0

        # ---- ① 路由：技能 + 工具名单 ----
        r = self.sut.route(case.query)
        total_ms += float(r.get("latency_ms", 0))
        actual["skill"] = r.get("skill", "")
        actual["skill_route"] = r.get("skill_route", "")
        actual["tools"] = r.get("tools", [])
        actual["source"] = r.get("source", "")
        if r.get("error"):
            return CaseResult(case.id, case.query, list(case.tags), {}, actual,
                              error=f"route: {r['error']}", latency_ms=total_ms)

        # 技能：期望 "" 表示应走通用流程，所以是相等判定而不是包含判定
        dims["skill"] = {
            "expected": case.expect_skill or "（通用流程）",
            "actual": r.get("skill", "") or "（通用流程）",
            "ok": (r.get("skill", "") == case.expect_skill),
        }
        if case.expect_tools:
            exp, act = set(case.expect_tools), set(r.get("tools", []))
            dims["tools"] = {
                "expected": sorted(exp), "actual": sorted(act),
                "ok": exp.issubset(act),
                "missing": sorted(exp - act),
                "extra": sorted(act - exp),
            }
        else:
            dims["tools"] = {"expected": [], "actual": sorted(r.get("tools", [])),
                             "ok": None, "missing": [], "extra": []}

        # ---- ② 意图 ----
        if case.expect_intent:
            it = self.sut.intent(case.query)
            total_ms += float(it.get("latency_ms", 0))
            actual["intent"] = it.get("intent", "")
            if it.get("error"):
                dims["intent"] = {"expected": case.expect_intent, "actual": "",
                                  "ok": None, "error": it["error"]}
            else:
                dims["intent"] = {"expected": case.expect_intent,
                                  "actual": it.get("intent", ""),
                                  "ok": it.get("intent", "") == case.expect_intent}

        # ---- ③ 风控 ----
        rk = self.sut.risk(case.query)
        actual["risk_passed"] = rk.get("passed")
        actual["risk_level"] = rk.get("level", "")
        if rk.get("error"):
            dims["risk"] = {"expected": case.expect_risk_pass, "actual": None,
                            "ok": None, "error": rk["error"]}
        else:
            dims["risk"] = {"expected": case.expect_risk_pass,
                            "actual": rk.get("passed"),
                            "ok": bool(rk.get("passed")) == case.expect_risk_pass}

        # ---- ④ 提示词组装（默认开）----
        if self.check_prompt:
            bp = self.sut.build_prompt(case.query)
            total_ms += float(bp.get("latency_ms", 0))
            actual["prompt_tokens"] = bp.get("prompt_tokens", 0)
            actual["trimmed"] = bp.get("trimmed", [])
            if bp.get("error"):
                dims["prompt"] = {"expected": "渲染成功且不超预算", "actual": "",
                                  "ok": False, "error": bp["error"]}
            else:
                within = bp.get("context_tokens", 0) <= self.max_context_tokens
                dims["prompt"] = {
                    "expected": f"context <= {self.max_context_tokens}",
                    "actual": bp.get("context_tokens", 0),
                    "ok": within,
                }

        return CaseResult(case.id, case.query, list(case.tags), dims, actual,
                          latency_ms=round(total_ms, 1))

    # ------------------------------------------------------------------
    # 【整批跑】
    #   返回可直接归档的报告 dict（也是基线文件格式）
    # ------------------------------------------------------------------
    def run(self, cases: List[EvalCase], verbose: bool = False) -> Dict:
        board = ScoreBoard()
        results: List[CaseResult] = []

        for c in cases:
            res = self.run_case(c)
            results.append(res)

            # 执行异常：只记 errors，不进分母（否则环境抖动会被算成效果下降）
            if res.error:
                board.skill.add(None)
                board.risk.add(None)
                board.intent.add(None)
                continue

            board.skill.add(res.dims.get("skill", {}).get("ok"))
            board.risk.add(res.dims.get("risk", {}).get("ok"))
            if "intent" in res.dims:
                board.intent.add(res.dims["intent"].get("ok"))
            if res.dims.get("tools", {}).get("ok") is not None:
                exp = set(c.expect_tools)
                act = set(res.actual.get("tools", []))
                board.add_tools(exp, act)
            if self.check_prompt and "prompt" in res.dims:
                board.add_prompt(
                    int(res.actual.get("prompt_tokens", 0)),
                    bool(res.dims["prompt"].get("ok")),
                    res.latency_ms,
                )

        passed = sum(1 for r in results if r.ok)
        failures = [r for r in results if not r.ok][:_MAX_FAIL_DETAIL]

        report: Dict[str, Any] = {
            "meta": {
                "mode": self.sut.mode,
                "cases": len(cases),
                "passed": passed,
                "pass_rate": round(passed / len(cases), 4) if cases else 0.0,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "python": platform.python_version(),
                "notes": list(self.sut.notes),
            },
            "summary": board.summary(),
            "by_tag": _tag_stats(results),
            "failures": [r.to_dict() for r in failures],
        }
        if verbose:
            report["cases"] = [r.to_dict() for r in results]
        return report


# 【什么】按标签分组统计
# 【作用】报告里能看出"是计算类还行但知识库类崩了"，而不是只有一个总分
def _tag_stats(results: List[CaseResult]) -> Dict[str, Dict]:
    agg: Dict[str, Dict[str, List]] = {}
    for r in results:
        for t in (r.tags or ["(untagged)"]):
            d = agg.setdefault(t, {"skill": [], "risk": [], "intent": [], "all": []})
            if "skill" in r.dims and r.dims["skill"].get("ok") is not None:
                d["skill"].append(bool(r.dims["skill"]["ok"]))
            if "risk" in r.dims and r.dims["risk"].get("ok") is not None:
                d["risk"].append(bool(r.dims["risk"]["ok"]))
            if "intent" in r.dims and r.dims["intent"].get("ok") is not None:
                d["intent"].append(bool(r.dims["intent"]["ok"]))
            d["all"].append(bool(r.ok))

    def rate(xs: List[bool]) -> float:
        return round(sum(1 for x in xs if x) / len(xs), 4) if xs else 0.0

    return {
        tag: {
            "cases": len(d["all"]),
            "pass_rate": rate(d["all"]),
            "skill_acc": rate(d["skill"]),
            "risk_acc": rate(d["risk"]),
            "intent_acc": rate(d["intent"]),
        }
        for tag, d in sorted(agg.items())
    }
