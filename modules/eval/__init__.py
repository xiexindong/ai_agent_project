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
from __future__ import annotations  # 启用注解延迟求值：类型标注可引用后方才定义的名字

from typing import Dict, List, Optional  # 泛型类型注解工具

from modules.eval.dataset import EvalCase, load_cases, save_cases, validate_cases  # 用例数据结构 + JSONL 读写 + 用例集自检
from modules.eval.harness import build_sut  # 被测对象装配工厂
from modules.eval.metrics import diff_summary  # 基线对比
from modules.eval.report import render_ci_summary, render_markdown  # 报告渲染双出口（Markdown / CI 摘要）
from modules.eval.runner import CaseResult, EvalRunner  # 运行器与单条结果结构


# 【什么】一步跑完评测
# 【作用】给脚本与测试用的便捷入口：装载用例 → 装配被测对象 → 跑 → 出报告
# 【为什么】把四步流程固化成一个函数：脚本/测试不会各自拼出口径不一的调用序列，
#           所有人跑出来的评测都遵循同一条路径
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
    from modules.eval.dataset import filter_by_tag  # 延迟导入：只在本函数用到，避免模块顶层依赖过重

    cases = load_cases(cases_path)  # ① 装载用例（JSONL → EvalCase 列表）
    if tags:  # 给了标签过滤条件
        cases = filter_by_tag(cases, tags)  # 只留命中标签的用例
    sut = build_sut(mode=mode, max_tools=max_tools)  # ② 按档位装配被测对象
    return EvalRunner(sut, check_prompt=check_prompt).run(cases, verbose=verbose)  # ③④ 造运行器并整批跑，直接返回报告 dict


__all__ = [  # 显式声明公开 API：from modules.eval import * 只导出这些
    "EvalCase",  # 用例数据结构
    "CaseResult",  # 单条结果结构
    "EvalRunner",  # 评测运行器
    "load_cases",  # 加载 JSONL 用例
    "save_cases",  # 写回 JSONL 用例
    "validate_cases",  # 用例集自检
    "build_sut",  # 装配被测对象
    "diff_summary",  # 基线对比
    "render_markdown",  # Markdown 报告渲染
    "render_ci_summary",  # CI 摘要渲染
    "run_eval",  # 一站式评测入口
]
