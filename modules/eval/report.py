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
from __future__ import annotations  # 启用注解延迟求值：类型标注可引用后方才定义的名字

from typing import Dict, List, Optional  # 泛型类型注解工具

_STATUS_ICON = {  # 对比状态 → 人读图标 的映射表：报告里一眼看出变化方向
    "improved": "⬆ 改善",  # 改善：本次比基线好
    "regressed": "⬇ 回归",  # 回归：本次比基线差
    "same": "— 持平",  # 持平：变化在噪声阈值内
    "new": "🆕 新增",  # 新增：基线里还没有这个指标
}


# 【什么】小数转百分比字符串
# 【作用】0.875 → "87.5%"；报告里所有得分统一走这个出口
# 【为什么】格式化口径集中在唯一函数里：改精度/改样式只改一处，全报告一致
def _pct(x: float) -> str:  # 入参为 0~1 的小数
    return f"{x * 100:.1f}%"  # 乘 100、保留 1 位小数、加百分号


# 【什么】渲染 Markdown 报告
# 【作用】把 runner 产出的 report dict 渲染成完整 Markdown 文本
# 【为什么】渲染逻辑与数据生产分离：runner 不碰展示，报告改版式不用动评测逻辑
def render_markdown(report: Dict, diff: Optional[List[Dict]] = None,
                    baseline_name: str = "") -> str:
    """渲染 Markdown 报告。

    Args:
        report:        runner.run() 的返回值
        diff:          metrics.diff_summary() 的结果；传了才输出对比段
        baseline_name: 基线文件说明，写进对比段标题
    """
    meta = report.get("meta", {})  # 取元信息段；缺段给空 dict 防 KeyError
    s = report.get("summary", {})  # 取记分板汇总段
    lines: List[str] = []  # 逐行收集，最后 join 成完整文本

    lines.append("# AI Agent 评测报告")  # 报告标题
    lines.append("")  # 空行：Markdown 段落分隔
    lines.append(f"- 运行模式：**{meta.get('mode')}**"
                 f"（offline = 确定性链路，不含 embedding/LLM 外部依赖）")  # 档位行：口径解读的关键信息
    lines.append(f"- 用例数：{meta.get('cases')} ｜ 全维度通过：{meta.get('passed')}"
                 f" ｜ 通过率：**{_pct(meta.get('pass_rate', 0))}**")  # 总量行：用例数/通过数/通过率
    lines.append(f"- 生成时间：{meta.get('created_at')} ｜ Python {meta.get('python')}")  # 环境行：时间与解释器版本
    for n in meta.get("notes", []):  # 逐条输出装配降级说明
        lines.append(f"- 说明：{n}")  # 说明行（如"L2 向量路由已关闭"）
    lines.append("")  # 空行

    # ---- 口径说明 ----
    # 【为什么必须写这段】offline 档关掉了 embedding 与 LLM，语义类指标天然偏低。
    #   不写清楚，读者会把它当成端到端效果，得出"系统只有 10 分"的错误结论。
    lines.append("> **口径说明**：全维度通过率 = 技能、工具、风控、意图、提示词"
                 "五个维度全部正确才计通过，是刻意取的严格口径；"
                 "看整体效果请按下面的分项指标读。")  # 引用块：解释全维度通过率的严格口径
    if meta.get("mode") == "offline":  # 离线档追加局限性声明
        lines.append("> **offline 档的局限**：未启用 embedding 与 LLM，"
                     "语义召回退化为关键词、意图退化为规则、答案质量不参与评分。"
                     "本档用于**回归对比**（同一口径下前后比），"
                     "端到端效果请用 `--mode online` 跑。")  # 引用块：offline 档能证明什么、不能证明什么
    lines.append("")  # 空行

    # ---- 核心指标 ----
    lines.append("## 核心指标")  # 小节标题
    lines.append("")  # 空行
    lines.append("| 指标 | 得分 | 通过/总数 | 未参与 |")  # 表头
    lines.append("|---|---:|---:|---:|")  # 对齐行：数值列右对齐
    for key, label in (  # 逐指标渲染一行：key 是 summary 里的字段名，label 是给人看的名字
        ("skill_acc", "技能路由准确率"),  # 技能
        ("tool_recall", "工具召回率（该带的带了吗）"),  # 工具召回
        ("tool_precision", "工具精确率（带的多余吗）"),  # 工具精确
        ("tool_f1", "工具 F1"),  # 工具 F1
        ("intent_acc", "意图分类准确率"),  # 意图
        ("risk_acc", "风控判定准确率"),  # 风控
        ("answer_quality", "答案质量（仅 online 档）"),  # 答案质量
    ):
        m = s.get(key, {})  # 取该指标的 dict；缺省给空 dict
        if not m.get("total") and key == "answer_quality":  # 答案质量在 offline 档恒为 0 条：直接跳过这行，避免误读
            continue  # 跳过
        lines.append(f"| {label} | {_pct(m.get('score', 0))} | "
                     f"{m.get('passed', 0)}/{m.get('total', 0)} | {m.get('errors', 0)} |")  # 表格行：得分/通过数/未参与数
    p = s.get("prompt", {})  # 取 prompt 分布段
    lines.append(f"| 提示词预算遵守率 | {_pct(p.get('budget_ok_rate', 0))} | "
                 f"— | — |")  # 预算遵守率行：无通过/总数概念，用 — 占位
    lines.append(f"| 平均 prompt token | {p.get('avg_tokens', 0)} | "
                 f"峰值 {p.get('max_tokens', 0)} | — |")  # token 行：均值 + 峰值合在一行
    lines.append("")  # 空行

    # ---- 分组 ----
    by_tag = report.get("by_tag", {})  # 取分组统计段
    if by_tag:  # 有分组数据才渲染这一节
        lines.append("## 分组表现")  # 小节标题
        lines.append("")  # 空行
        lines.append("| 标签 | 用例 | 全通过率 | 技能 | 风控 | 意图 |")  # 表头
        lines.append("|---|--:|--:|--:|--:|--:|")  # 对齐行
        for tag, d in by_tag.items():  # 逐标签渲染一行
            lines.append(
                f"| {tag} | {d.get('cases', 0)} | {_pct(d.get('pass_rate', 0))} "
                f"| {_pct(d.get('skill_acc', 0))} | {_pct(d.get('risk_acc', 0))} "
                f"| {_pct(d.get('intent_acc', 0))} |"
            )  # 表格行：用例数与各维度通过率
        lines.append("")  # 空行

    # ---- 与基线对比 ----
    if diff:  # 传了 diff 才渲染这一节
        lines.append(f"## 与基线对比{'（' + baseline_name + '）' if baseline_name else ''}")  # 小节标题：有基线名就带上
        lines.append("")  # 空行
        lines.append("| 指标 | 基线 | 本次 | 变化 | 判定 |")  # 表头
        lines.append("|---|---:|---:|---:|---|")  # 对齐行
        for d in diff:  # 逐指标渲染一行
            icon = _STATUS_ICON.get(d.get("status", "same"), d.get("status", ""))  # 状态映射成图标；未知状态原样显示
            base = "—" if d.get("base") is None else d.get("base")  # 基线缺失（新增指标）显示 —
            delta = "—" if d.get("delta") is None else d.get("delta")  # 差值缺失同样显示 —
            lines.append(f"| {d.get('metric')} | {base} | {d.get('cur')} | {delta} | {icon} |")  # 表格行
        lines.append("")  # 空行

    # ---- 失败明细 ----
    failures = report.get("failures", [])  # 取失败明细段
    if failures:  # 有失败才逐条展开
        lines.append(f"## 失败明细（最多 {len(failures)} 条）")  # 小节标题：条数即截断后的长度
        lines.append("")  # 空行
        for f in failures:  # 逐条失败渲染
            lines.append(f"### `{f.get('id')}` — {f.get('query')}")  # 小标题：用例 id + 提问原文
            if f.get("error"):  # 执行异常优先展示：它是环境问题，不是效果问题
                lines.append(f"- 执行异常：`{f['error']}`")  # 异常行
            for dim, d in (f.get("dims") or {}).items():  # 逐维度渲染
                if d.get("ok") is True:  # 通过的维度不展示：失败明细只看问题
                    continue  # 跳过
                mark = "❌" if d.get("ok") is False else "⚠️"  # False=明确失败打 ❌；None=不适用打 ⚠️
                lines.append(f"- {mark} **{dim}**：期望 `{d.get('expected')}` "
                             f"／ 实际 `{d.get('actual')}`")  # 维度行：期望 vs 实际
                if d.get("missing"):  # 工具维度的漏召清单
                    lines.append(f"  - 漏召：{d['missing']}")  # 漏召行
                if d.get("extra"):  # 工具维度的多带清单
                    lines.append(f"  - 多带：{d['extra']}")  # 多带行
            lines.append("")  # 空行：两条失败之间留白
    else:  # 没有失败
        lines.append("## 失败明细")  # 小节标题
        lines.append("")  # 空行
        lines.append("全部用例通过 ✅")  # 全通过提示
        lines.append("")  # 空行

    return "\n".join(lines)  # 逐行 join 成完整 Markdown 文本返回


# 【什么】渲染 CI / 终端用的一屏摘要
# 【作用】CI 日志里一眼看到通过率与回归项，不用翻 Markdown
# 【为什么】CI 场景只关心两个问题的答案：过没过？有没有回归？
#           完整报告的信息密度在终端里反而是干扰
def render_ci_summary(report: Dict, diff: Optional[List[Dict]] = None) -> str:
    """给 CI / 终端看的一屏摘要。"""
    meta = report.get("meta", {})  # 取元信息段
    s = report.get("summary", {})  # 取记分板汇总段
    out = [  # 两行核心摘要
        f"评测通过率 {_pct(meta.get('pass_rate', 0))} "
        f"({meta.get('passed')}/{meta.get('cases')})  mode={meta.get('mode')}",  # 第一行：通过率 + 档位
        f"  技能 {_pct(s.get('skill_acc', {}).get('score', 0))} | "
        f"工具F1 {_pct(s.get('tool_f1', {}).get('score', 0))} | "
        f"风控 {_pct(s.get('risk_acc', {}).get('score', 0))} | "
        f"意图 {_pct(s.get('intent_acc', {}).get('score', 0))}",  # 第二行：四个核心分项
    ]
    if diff:  # 有基线对比时追加回归情况
        bad = [d for d in diff if d.get("status") == "regressed"]  # 筛出回归项
        if bad:  # 有回归
            out.append("  回归项: " + ", ".join(
                f"{d['metric']}({d['base']}→{d['cur']})" for d in bad))  # 逐条列出：指标名(基线→本次)
        else:  # 无回归
            out.append("  无回归项")  # 明确告知，CI 日志不用猜
    return "\n".join(out)  # join 成多行文本返回
