"""
评测模块（Evaluation）
======================
给 AI Agent 的行为建立「可重复测量」的能力，是效果类改动能不能上线的依据。

【三层职责】
    dataset.py  一条用例长什么样（提问 + 应发生什么）+ JSONL 读写与自检
    harness.py  被测对象装配（offline 确定性档 / online 完整档）
    metrics.py  判定结果的度量：准确率、集合 F1、基线对比
    runner.py   逐条跑、逐维判分、汇总成记分板
    report.py   Markdown 报告与 CI 摘要渲染

【最小用法】
    from modules.eval import run_eval
    report = run_eval("eval_cases.jsonl", mode="offline")
    print(report["meta"]["pass_rate"])

【命令行用法】
    python scripts/run_eval.py                      # 离线档，出报告
    python scripts/run_eval.py --update-baseline    # 把本次结果存为基线
    python scripts/run_eval.py --mode online        # 完整链路（需密钥与网络）

【为什么这层是必需的】
    没有它，"改了提示词/改了路由效果变好还是变坏"只能靠人工试几句凭感觉判断，
    而且改坏了往往几周后才被发现。有了它，每次改动跑一遍就知道有没有回归，
    这也是区分"做过 Agent"和"工程化地做过 Agent"的分水岭。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from modules.eval.dataset import EvalCase, load_cases, save_cases, validate_cases
from modules.eval.harness import build_sut
from modules.eval.metrics import diff_summary
from modules.eval.report import render_ci_summary, render_markdown
from modules.eval.runner import CaseResult, EvalRunner


# 【什么】一步跑完评测
# 【作用】给脚本与测试用的便捷入口：装载用例 → 装配被测对象 → 跑 → 出报告
def run_eval(cases_path: str = "eval_cases.jsonl", mode: str = "offline",
             tags: Optional[List[str]] = None, max_tools: int = 8,
             check_prompt: bool = True, verbose: bool = False) -> Dict:
    """跑一轮评测并返回报告 dict。

    Args:
        cases_path:   用例文件路径（JSONL）
        mode:         offline / online
        tags:         只跑带这些标签的用例
        max_tools:    单轮工具名单上限
        check_prompt: 是否评测提示词组装（预算与裁剪）
        verbose:      是否在报告里带上全量用例明细
    """
    from modules.eval.dataset import filter_by_tag

    cases = load_cases(cases_path)
    if tags:
        cases = filter_by_tag(cases, tags)
    sut = build_sut(mode=mode, max_tools=max_tools)
    return EvalRunner(sut, check_prompt=check_prompt).run(cases, verbose=verbose)


__all__ = [
    "EvalCase",
    "CaseResult",
    "EvalRunner",
    "load_cases",
    "save_cases",
    "validate_cases",
    "build_sut",
    "diff_summary",
    "render_markdown",
    "render_ci_summary",
    "run_eval",
]
