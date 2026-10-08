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
from __future__ import annotations  # 启用注解延迟求值：类型标注可引用后方才定义的名字

from dataclasses import dataclass, field  # dataclass 工具：定义 Metric / ScoreBoard 两个数据类
from typing import Dict, List, Optional, Set, Tuple  # 泛型类型注解工具


# 【什么】集合的精确率 / 召回率 / F1
# 【作用】工具名单的匹配度量；expected 为空集时精确率记为 1.0（没期待就不该怪它多带）
# 【为什么】名单型输出要区分"漏召"与"多带"两种错法，单一对错口径看不出该调哪个参数
def prf(expected: Set[str], actual: Set[str]) -> Tuple[float, float, float]:
    """返回 (precision, recall, f1)。

    【边界】expected 为空：召回率记 1.0（不期待任何东西，没漏），
            精确率记 1.0（名单里带什么都不算错，因为本条不校验工具）；
            两者都为空 → 返回 (1.0, 1.0, 1.0)。
    """
    if not expected and not actual:  # 边界①：双方皆空——本条不校验工具且确实没带，算满分
        return 1.0, 1.0, 1.0  # 直接返回满分
    if not expected:  # 边界②：期望为空但实际有输出——本条不校验工具，多带不算错
        return 1.0, 1.0, 1.0  # 按约定给满分
    if not actual:  # 边界③：期望非空但实际为空——全漏召，召回率 0
        return 1.0, 0.0, 0.0  # 精确率按约定给 1.0 避免除零歧义，召回率与 F1 为 0
    hit = len(expected & actual)  # 命中数 = 两集合交集大小
    precision = hit / len(actual)  # 精确率 = 命中 / 实际带了几个
    recall = hit / len(expected)  # 召回率 = 命中 / 该带几个
    if precision + recall == 0:  # 两者皆零的防御分支（理论上 hit=0 时会触发）→ F1 直接为 0，避免除零
        return 0.0, 0.0, 0.0  # 全错返回全零
    f1 = 2 * precision * recall / (precision + recall)  # 调和平均：对偏科敏感，任一项低则 F1 低
    return round(precision, 4), round(recall, 4), round(f1, 4)  # 统一保留 4 位小数：便于跨次运行比较


# 【什么】单个指标的累加器
# 【作用】按"通过数 / 总数 / 异常数"三元组持续累计判定结果，并即时刷新得分
# 【为什么】用例量可能上万，先存全量明细再统计会吃内存；累加器只要 O(1) 空间
@dataclass  # 自动生成 __init__/__repr__ 等样板代码
class Metric:
    """一个指标的汇总结果。

    Attributes:
        name:    指标名，如 skill_acc
        passed:  通过条数
        total:   参与统计的条数
        score:   passed/total，保留 4 位
        errors:  执行异常导致无法判分的条数（不计入 total）
    """

    name: str  # 指标名：如 skill_acc，报告里直接展示
    passed: int = 0  # 判定通过的条数
    total: int = 0  # 参与统计的条数（异常/不适用不计入）
    errors: int = 0  # 执行异常或不适用的条数（只展示，不进分母）
    score: float = 0.0  # 得分 = passed/total，随每次 add 即时刷新

    # 【什么】累加一次判定结果
    # 【作用】ok=None 表示本条不适用/执行异常，只记 errors 不进分母
    # 【为什么】None 若进分母，会把环境抖动算成效果下降，口径上必须排除
    def add(self, ok: Optional[bool]) -> None:  # 入参为三态布尔：True 通过 / False 未通过 / None 不适用
        if ok is None:  # 不适用 / 执行异常分支
            self.errors += 1  # 只记异常数
            return  # 不进分母，直接返回
        self.total += 1  # 正常判定：分母 +1
        if ok:  # 判定通过
            self.passed += 1  # 分子 +1
        self.score = round(self.passed / self.total, 4) if self.total else 0.0  # 即时刷新得分，保留 4 位小数

    # 【什么】序列化为 dict
    # 【作用】基线文件与报告渲染都消费这个结构
    # 【为什么】字段名与 summary() 的约定一致，下游渲染/基线对比才能对得上
    def to_dict(self) -> Dict:  # 转成可 JSON 序列化的字典
        return {  # 固定五字段
            "name": self.name,  # 指标名
            "passed": self.passed,  # 通过数
            "total": self.total,  # 总数（置信度判断的分母）
            "score": self.score,  # 得分
            "errors": self.errors,  # 异常数
        }


# 【什么】一块记分板：按维度累加所有用例的判定结果
# 【作用】runner 每跑完一条就往这里记一笔，跑完调 summary() 得到汇总
# 【为什么】对错型指标用 Metric 累加器（O(1) 内存），分布型指标（P/R/F1、token、耗时）
#           必须留原始序列才能算均值/峰值——两类口径分开存储
@dataclass  # 自动生成样板代码
class ScoreBoard:
    """一块记分板：按维度累加所有用例的判定结果。

    【为什么用累加器而不是先收集再统计】
        用例可能上万条，先存全量明细再统计会吃掉内存；
        累加器只需 O(维度数) 的内存，明细只保留失败项（报告要看的）。
    """

    skill: Metric = field(default_factory=lambda: Metric("skill_acc"))  # 技能路由准确率累加器
    intent: Metric = field(default_factory=lambda: Metric("intent_acc"))  # 意图分类准确率累加器
    risk: Metric = field(default_factory=lambda: Metric("risk_acc"))  # 风控判定准确率累加器
    branch: Metric = field(default_factory=lambda: Metric("branch_acc"))  # 流水线支路准确率累加器（预留维度）
    answer: Metric = field(default_factory=lambda: Metric("answer_quality"))  # 答案质量累加器（仅 online 档使用）

    _tool_p: List[float] = field(default_factory=list)      # 每条用例的精确率
    _tool_r: List[float] = field(default_factory=list)      # 每条用例的召回率
    _tool_f1: List[float] = field(default_factory=list)  # 每条用例的 F1：汇总时取平均
    _prompt_tokens: List[int] = field(default_factory=list)  # 每条用例的 prompt token 数
    _budget_ok: List[bool] = field(default_factory=list)  # 每条用例是否守住预算
    _latency_ms: List[float] = field(default_factory=list)  # 每条用例的端到端耗时

    # 【什么】记一次工具名单比对
    # 【作用】expected 为空的用例不参与统计（那条本来就不校验工具）
    # 【为什么】不校验工具的用例若进均值，会稀释/抬高指标，口径上必须排除
    def add_tools(self, expected: Set[str], actual: Set[str]) -> Tuple[float, float, float]:  # 入参为期望与实际两个集合
        if not expected:  # 本条不校验工具：直接返回满分约定值，不进统计
            return 1.0, 1.0, 1.0
        p, r, f = prf(expected, actual)  # 算本条 P/R/F1
        self._tool_p.append(p)  # 精确率入列
        self._tool_r.append(r)  # 召回率入列
        self._tool_f1.append(f)  # F1 入列
        return p, r, f  # 顺带返回本条结果，方便调用方记录

    # 【什么】记一次提示词组装结果
    # 【作用】累计 token 数、预算遵守、耗时三组数据
    # 【为什么】prompt 维度的指标不是单次对错而是分布特征（均值/峰值/遵守率），必须留原始序列
    def add_prompt(self, tokens: int, within_budget: bool, latency_ms: float = 0.0) -> None:  # 入参：token 数、预算是否守住、耗时
        self._prompt_tokens.append(tokens)  # token 数入列
        self._budget_ok.append(within_budget)  # 预算遵守入列
        if latency_ms > 0:  # 耗时为正才记（0 表示未测/无意义）
            self._latency_ms.append(latency_ms)  # 耗时入列

    # 【什么】汇总成可归档的 dict
    # 【作用】这份结构就是基线文件的内容，也是报告的数据源
    # 【为什么】结构一旦定死，基线对比与报告渲染才有稳定契约；改结构要三处同步改
    def summary(self) -> Dict:  # 汇总出口
        def avg(xs: List[float]) -> float:  # 局部小工具：列表均值，空列表给 0 防除零
            return round(sum(xs) / len(xs), 4) if xs else 0.0  # 均值保留 4 位小数

        return {  # 固定结构：五个对错型指标 + 三个均值型指标 + prompt 分布
            "skill_acc": self.skill.to_dict(),  # 技能准确率
            "intent_acc": self.intent.to_dict(),  # 意图准确率
            "risk_acc": self.risk.to_dict(),  # 风控准确率
            "branch_acc": self.branch.to_dict(),  # 支路准确率（预留维度）
            "answer_quality": self.answer.to_dict(),  # 答案质量（online 档）
            "tool_precision": {  # 工具精确率：按条取平均（宏平均）
                "name": "tool_precision", "score": avg(self._tool_p),  # 平均精确率
                "total": len(self._tool_p), "passed": 0, "errors": 0,  # 参与条数；对错型字段对均值型指标无意义，置 0
            },
            "tool_recall": {  # 工具召回率
                "name": "tool_recall", "score": avg(self._tool_r),  # 平均召回率
                "total": len(self._tool_r), "passed": 0, "errors": 0,  # 参与条数
            },
            "tool_f1": {  # 工具 F1
                "name": "tool_f1", "score": avg(self._tool_f1),  # 平均 F1
                "total": len(self._tool_f1), "passed": 0, "errors": 0,  # 参与条数
            },
            "prompt": {  # 提示词维度：分布型指标
                "avg_tokens": round(sum(self._prompt_tokens) / len(self._prompt_tokens), 1)
                if self._prompt_tokens else 0,  # 平均 token，空序列给 0 防除零
                "max_tokens": max(self._prompt_tokens) if self._prompt_tokens else 0,  # 峰值 token（最坏情况，最接近爆预算的那次）
                "budget_ok_rate": round(
                    sum(1 for b in self._budget_ok if b) / len(self._budget_ok), 4
                ) if self._budget_ok else 0.0,  # 预算遵守率 = 遵守条数 / 总条数
                "avg_latency_ms": round(avg(self._latency_ms), 1),  # 平均耗时，保留 1 位小数
            },
        }


# 【什么】两次运行结果的逐指标对比
# 【作用】标出改善 / 回归 / 持平，供回归门禁判定是否放行
# 【为什么】单看本次分数回答不了"这次改动是好是坏"，必须与基线同口径对比才有结论
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
            "answer_quality", "tool_precision", "tool_recall", "tool_f1"]  # 参与对比的对错/比值型指标清单
    out: List[Dict] = []  # 对比结果逐条追加
    for k in keys:  # 逐指标对比
        b = base.get(k, {}).get("score")  # 基线得分；缺该指标得 None
        c = cur.get(k, {}).get("score")  # 本次得分
        if b is None:  # 基线里没有这个指标 → 是新增指标
            out.append({"metric": k, "base": None, "cur": c, "delta": None, "status": "new"})  # 标 new，不参与回归判定
            continue  # 下一个指标
        delta = round((c or 0) - (b or 0), 4)  # 差值；None 兜底成 0 防 TypeError
        if delta > tolerance:  # 正向超阈值
            status = "improved"  # 改善
        elif delta < -tolerance:  # 负向超阈值
            status = "regressed"  # 回归
        else:  # 阈值以内
            status = "same"  # 持平（浮点噪声不算变化）
        out.append({"metric": k, "base": b, "cur": c, "delta": delta, "status": status})  # 记录本条对比

    # ---- token 与延迟：越高越差，且必须用相对阈值 ----
    # 【为什么这里不能用 tolerance 那个绝对差值】
    #   耗时受机器负载、CPU 调度、冷热缓存影响，同一份代码两次跑差 50% 很常见
    #   （实测 12.5ms → 19.5ms，纯噪声）。用绝对差值会把每次运行都判成回归，
    #   门禁天天红灯，最后没人看它 —— 那才是门禁真正的死法。
    #   所以这两项按"相对变化超过 X% 才算数"判定：token 10%、延迟 25%。
    bp = base.get("prompt", {})  # 基线 prompt 段
    cp = cur.get("prompt", {})  # 本次 prompt 段
    for field_name, rel_tol in (("avg_tokens", 0.10), ("avg_latency_ms", 0.25)):  # 两个"越高越差"的字段及其相对阈值：token 10%、延迟 25%
        b, c = bp.get(field_name, 0) or 0, cp.get(field_name, 0) or 0  # 取双方值；None/缺失兜底成 0
        delta = round(c - b, 2)  # 绝对差值（仅展示用，判定不用它）
        if not b:  # 基线为 0：相对变化无意义（除零），直接算持平
            status = "same"  # 持平
        elif abs(delta) / abs(b) < rel_tol:  # 相对变化在阈值内
            status = "same"  # 判持平：噪声不算回归
        else:  # 相对变化超阈值
            status = "regressed" if delta > 0 else "improved"  # 方向取反：涨 = 回归，降 = 改善
        out.append({"metric": field_name, "base": b, "cur": c, "delta": delta, "status": status})  # 记录本条对比

    b_rate = bp.get("budget_ok_rate", 0)  # 基线预算遵守率
    c_rate = cp.get("budget_ok_rate", 0)  # 本次预算遵守率
    d_rate = round((c_rate or 0) - (b_rate or 0), 4)  # 遵守率差值；None 兜底成 0
    out.append({  # 预算遵守率对比（它是"越高越好"，用回绝对阈值口径）
        "metric": "budget_ok_rate", "base": b_rate, "cur": c_rate, "delta": d_rate,
        "status": "regressed" if d_rate < -tolerance else ("improved" if d_rate > tolerance else "same"),  # 三态判定
    })
    return out  # 返回全部指标的对比列表
