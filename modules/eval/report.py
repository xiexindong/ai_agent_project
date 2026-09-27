"""
评测报告渲染
============
把 report dict 渲染成人能直接读的 Markdown，以及给 CI 用的纯文本摘要。

【为什么要同时产出 Markdown 和 JSON】
    JSON 给机器：基线对比、CI 门禁、趋势图都吃它。
    Markdown 给人：评审、简历附件、PR 描述里直接贴。
    两者同源，都来自 runner 返回的同一份 dict，不会出现"报告和数据对不上"。

【为什么对比表要标出方向】
    分数下降是回归，但 token 上升也是回归、下降才是改善。
    不同指标的好坏方向不同，统一标成"高了就是好"会得出相反结论。
"""
from __future__ import annotations

from typing import Dict, List, Optional

_STATUS_ICON = {
    "improved": "⬆ 改善",
    "regressed": "⬇ 回归",
    "same": "— 持平",
    "new": "🆕 新增",
}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_markdown(report: Dict, diff: Optional[List[Dict]] = None,
                    baseline_name: str = "") -> str:
    """渲染 Markdown 报告。

    Args:
        report:        runner.run() 的返回值
        diff:          metrics.diff_summary() 的结果；传了才输出对比段
        baseline_name: 基线文件说明，写进对比段标题
    """
    meta = report.get("meta", {})
    s = report.get("summary", {})
    lines: List[str] = []

    lines.append("# AI Agent 评测报告")
    lines.append("")
    lines.append(f"- 运行模式：**{meta.get('mode')}**"
                 f"（offline = 确定性链路，不含 embedding/LLM 外部依赖）")
    lines.append(f"- 用例数：{meta.get('cases')} ｜ 全维度通过：{meta.get('passed')}"
                 f" ｜ 通过率：**{_pct(meta.get('pass_rate', 0))}**")
    lines.append(f"- 生成时间：{meta.get('created_at')} ｜ Python {meta.get('python')}")
    for n in meta.get("notes", []):
        lines.append(f"- 说明：{n}")
    lines.append("")

    # ---- 口径说明 ----
    # 【为什么必须写这段】offline 档关掉了 embedding 与 LLM，语义类指标天然偏低。
    #   不写清楚，读者会把它当成端到端效果，得出"系统只有 10 分"的错误结论。
    lines.append("> **口径说明**：全维度通过率 = 技能、工具、风控、意图、提示词"
                 "五个维度全部正确才计通过，是刻意取的严格口径；"
                 "看整体效果请按下面的分项指标读。")
    if meta.get("mode") == "offline":
        lines.append("> **offline 档的局限**：未启用 embedding 与 LLM，"
                     "语义召回退化为关键词、意图退化为规则、答案质量不参与评分。"
                     "本档用于**回归对比**（同一口径下前后比），"
                     "端到端效果请用 `--mode online` 跑。")
    lines.append("")

    # ---- 核心指标 ----
    lines.append("## 核心指标")
    lines.append("")
    lines.append("| 指标 | 得分 | 通过/总数 | 未参与 |")
    lines.append("|---|---:|---:|---:|")
    for key, label in (
        ("skill_acc", "技能路由准确率"),
        ("tool_recall", "工具召回率（该带的带了吗）"),
        ("tool_precision", "工具精确率（带的多余吗）"),
        ("tool_f1", "工具 F1"),
        ("intent_acc", "意图分类准确率"),
        ("risk_acc", "风控判定准确率"),
        ("answer_quality", "答案质量（仅 online 档）"),
    ):
        m = s.get(key, {})
        if not m.get("total") and key == "answer_quality":
            continue
        lines.append(f"| {label} | {_pct(m.get('score', 0))} | "
                     f"{m.get('passed', 0)}/{m.get('total', 0)} | {m.get('errors', 0)} |")
    p = s.get("prompt", {})
    lines.append(f"| 提示词预算遵守率 | {_pct(p.get('budget_ok_rate', 0))} | "
                 f"— | — |")
    lines.append(f"| 平均 prompt token | {p.get('avg_tokens', 0)} | "
                 f"峰值 {p.get('max_tokens', 0)} | — |")
    lines.append("")

    # ---- 分组 ----
    by_tag = report.get("by_tag", {})
    if by_tag:
        lines.append("## 分组表现")
        lines.append("")
        lines.append("| 标签 | 用例 | 全通过率 | 技能 | 风控 | 意图 |")
        lines.append("|---|--:|--:|--:|--:|--:|")
        for tag, d in by_tag.items():
            lines.append(
                f"| {tag} | {d.get('cases', 0)} | {_pct(d.get('pass_rate', 0))} "
                f"| {_pct(d.get('skill_acc', 0))} | {_pct(d.get('risk_acc', 0))} "
                f"| {_pct(d.get('intent_acc', 0))} |"
            )
        lines.append("")

    # ---- 与基线对比 ----
    if diff:
        lines.append(f"## 与基线对比{'（' + baseline_name + '）' if baseline_name else ''}")
        lines.append("")
        lines.append("| 指标 | 基线 | 本次 | 变化 | 判定 |")
        lines.append("|---|---:|---:|---:|---|")
        for d in diff:
            icon = _STATUS_ICON.get(d.get("status", "same"), d.get("status", ""))
            base = "—" if d.get("base") is None else d.get("base")
            delta = "—" if d.get("delta") is None else d.get("delta")
            lines.append(f"| {d.get('metric')} | {base} | {d.get('cur')} | {delta} | {icon} |")
        lines.append("")

    # ---- 失败明细 ----
    failures = report.get("failures", [])
    if failures:
        lines.append(f"## 失败明细（最多 {len(failures)} 条）")
        lines.append("")
        for f in failures:
            lines.append(f"### `{f.get('id')}` — {f.get('query')}")
            if f.get("error"):
                lines.append(f"- 执行异常：`{f['error']}`")
            for dim, d in (f.get("dims") or {}).items():
                if d.get("ok") is True:
                    continue
                mark = "❌" if d.get("ok") is False else "⚠️"
                lines.append(f"- {mark} **{dim}**：期望 `{d.get('expected')}` "
                             f"／ 实际 `{d.get('actual')}`")
                if d.get("missing"):
                    lines.append(f"  - 漏召：{d['missing']}")
                if d.get("extra"):
                    lines.append(f"  - 多带：{d['extra']}")
            lines.append("")
    else:
        lines.append("## 失败明细")
        lines.append("")
        lines.append("全部用例通过 ✅")
        lines.append("")

    return "\n".join(lines)


def render_ci_summary(report: Dict, diff: Optional[List[Dict]] = None) -> str:
    """给 CI / 终端看的一屏摘要。"""
    meta = report.get("meta", {})
    s = report.get("summary", {})
    out = [
        f"评测通过率 {_pct(meta.get('pass_rate', 0))} "
        f"({meta.get('passed')}/{meta.get('cases')})  mode={meta.get('mode')}",
        f"  技能 {_pct(s.get('skill_acc', {}).get('score', 0))} | "
        f"工具F1 {_pct(s.get('tool_f1', {}).get('score', 0))} | "
        f"风控 {_pct(s.get('risk_acc', {}).get('score', 0))} | "
        f"意图 {_pct(s.get('intent_acc', {}).get('score', 0))}",
    ]
    if diff:
        bad = [d for d in diff if d.get("status") == "regressed"]
        if bad:
            out.append("  回归项: " + ", ".join(
                f"{d['metric']}({d['base']}→{d['cur']})" for d in bad))
        else:
            out.append("  无回归项")
    return "\n".join(out)
