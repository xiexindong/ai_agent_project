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
from __future__ import annotations  # 启用注解延迟求值：类型标注可引用后方才定义的名字

import platform  # 取 Python 版本号：写进报告 meta，便于复现运行环境
from dataclasses import dataclass, field  # dataclass 工具：定义 CaseResult 数据类
from datetime import datetime  # 生成报告创建时间戳
from typing import Any, Dict, List, Optional, Set  # 泛型类型注解工具

from modules.eval.dataset import EvalCase  # 用例定义（输入侧数据结构）
from modules.eval.harness import Sut  # 被测系统句柄（执行侧）
from modules.eval.metrics import ScoreBoard  # 记分板（汇总侧）


# 单条用例最多保留多少失败维度说明
_MAX_FAIL_DETAIL = 50  # 失败明细截断上限：防止报告体积爆炸，默认 50 条足够定位问题


# 【什么】一条用例的完整结果：执行输出 + 逐维度判定 + 异常状态
# 【作用】runner 内部流转与报告序列化的基本单元
# 【为什么】执行状态（error）与判定状态（ok）必须分开存放，才能在报告里区分
#           "环境炸了"与"效果不达标"（见模块 docstring 的设计说明）
@dataclass  # 自动生成 __init__/__repr__ 等样板代码
class CaseResult:
    """一条用例的执行与判定结果。"""

    id: str  # 用例唯一标识（如 calc-01）：基线对比按它对齐
    query: str  # 用户提问原文：报告里直接展示
    tags: List[str] = field(default_factory=list)  # 分类标签：by_tag 分组统计用
    dims: Dict[str, Dict] = field(default_factory=dict)   # {维度: {expected, actual, ok}}
    actual: Dict[str, Any] = field(default_factory=dict)  # 原始输出，排查用
    error: str = ""                                       # 执行异常；空 = 执行正常
    latency_ms: float = 0.0  # 本条用例各环节耗时合计（毫秒）

    # 【什么】本条用例整体是否通过（只读属性）
    # 【作用】把"所有维度的判定 + 执行状态"折叠成一个布尔值，供统计通过率
    # 【为什么】维度有三态（True 通过 / False 未通过 / None 不适用），口径必须是
    #           "没有明确 False 即通过"，否则不校验工具的闲聊用例会被误判失败（见 docstring 实测教训）
    @property  # 以属性形式暴露：调用方写 r.ok 而不是 r.ok()
    def ok(self) -> bool:
        """所有参与判定的维度都通过才算通过。

        【为什么写 `is not False` 而不是直接 `all(...)`】
            维度有三种状态：True 通过 / False 未通过 / None 本条不适用
            （如闲聊用例不校验工具名单）。
            `all(d["ok"] ...)` 会把 None 当假，于是"不校验工具"的闲聊用例
            永远判失败 —— 实测 4 条闲聊全红，而它们实际上全对。
            正确口径是：只要没有明确的 False 就算通过。
        """
        return all(d.get("ok") is not False for d in self.dims.values()) and not self.error  # 无维度判 False 且无执行异常才算通过

    # 【什么】序列化为普通 dict
    # 【作用】失败明细与 verbose 全量明细都靠它落进报告 JSON
    # 【为什么】dataclass 不能直接 JSON 序列化（含 property 与嵌套结构），显式列字段更可控
    def to_dict(self) -> Dict:  # 转成可 JSON 序列化的字典
        return {  # 逐字段组装；注意 ok 是 property，这里取值一次固化下来
            "id": self.id, "query": self.query, "tags": self.tags,  # 标识、原文与标签
            "dims": self.dims, "actual": self.actual, "error": self.error,  # 判定明细、原始输出与异常
            "latency_ms": self.latency_ms, "ok": self.ok,  # 耗时与整体结论
        }


# 【什么】评测运行器：驱动一套 Sut 跑完一批用例
# 【作用】对外只暴露 run_case（单条）与 run（整批）两个方法
# 【为什么】把"逐条执行 → 逐维判定 → 汇总记分"的流程封装在一处，
#           上层（脚本/测试）不需要知道判定细节
class EvalRunner:
    """评测运行器：拿一套 Sut 跑一批用例。

    Args:
        sut:            被测对象（见 harness.build_sut）
        check_prompt:   是否每条都做一次提示词组装评测（测预算与裁剪）
        max_context_tokens: 提示词预算上限，用于判定是否守住预算
    """

    # 【什么】初始化运行器
    # 【作用】绑定被测对象与提示词评测开关、预算红线
    # 【为什么】预算红线作为构造参数注入而不是写死：便于不同场景（如更严格预算）复用同一套逻辑
    def __init__(self, sut: Sut, check_prompt: bool = True,
                 max_context_tokens: int = 3000):
        self.sut = sut  # 被测对象句柄
        self.check_prompt = check_prompt  # 是否逐条做提示词组装评测
        self.max_context_tokens = max_context_tokens  # 提示词预算上限：prompt 维度的判定依据

    # ------------------------------------------------------------------
    # 【单条用例的判定流程】
    #   跑四个面 → 逐维度比对期望 → 记进记分板
    #   任一环节抛异常都记成 error，不中断整批评测。
    # ------------------------------------------------------------------
    # 【什么】跑一条用例并逐维度判定
    # 【作用】返回 CaseResult；执行异常短路返回，效果不符记录进 dims
    # 【为什么】四个被测面共用同一条 query，在一条方法里跑完才能保证
    #           "这条用例的各维度结果"天然对齐，不会出现跨用例串数据
    def run_case(self, case: EvalCase) -> CaseResult:  # 单条用例的执行 + 判定入口
        dims: Dict[str, Dict] = {}  # 逐维度判定结果：{维度名: {expected, actual, ok, ...}}
        actual: Dict[str, Any] = {}  # 原始输出收集：排查失败时要看"系统到底说了什么"
        total_ms = 0.0  # 各环节耗时累计

        # ---- ① 路由：技能 + 工具名单 ----
        r = self.sut.route(case.query)  # 执行路由被测面
        total_ms += float(r.get("latency_ms", 0))  # 累加路由耗时
        actual["skill"] = r.get("skill", "")  # 记录实际命中技能
        actual["skill_route"] = r.get("skill_route", "")  # 记录命中层级（L1/L2/L3）
        actual["tools"] = r.get("tools", [])  # 记录实际工具名单
        actual["source"] = r.get("source", "")  # 记录决策来源
        if r.get("error"):  # 路由执行炸了：短路返回——输入都没产出，后续维度跑了也没意义
            return CaseResult(case.id, case.query, list(case.tags), {}, actual,  # dims 传空：没有任何判定可做
                              error=f"route: {r['error']}", latency_ms=total_ms)  # error 带环节前缀，便于定位是哪一步炸的

        # 技能：期望 "" 表示应走通用流程，所以是相等判定而不是包含判定
        dims["skill"] = {  # 技能维度判定
            "expected": case.expect_skill or "（通用流程）",  # 期望为空串时显示成人话
            "actual": r.get("skill", "") or "（通用流程）",  # 实际为空串时同样显示成人话：报告两侧可读性一致
            "ok": (r.get("skill", "") == case.expect_skill),  # 严格相等：命中就是命中，没命中就是没命中
        }
        if case.expect_tools:  # 本条声明了期望工具名单 → 走子集判定
            exp, act = set(case.expect_tools), set(r.get("tools", []))  # 双方转集合：名单语义不关心顺序与重复
            dims["tools"] = {  # 工具维度判定
                "expected": sorted(exp), "actual": sorted(act),  # 排序后入报告：diff 友好
                "ok": exp.issubset(act),  # 判定口径：期望 ⊆ 实际（漏召即失败；多带不判死，多带由精确率指标反映）
                "missing": sorted(exp - act),  # 漏召清单：该带没带
                "extra": sorted(act - exp),  # 多带清单：带了不该带的
            }
        else:  # 本条不校验工具名单（如闲聊用例）
            dims["tools"] = {"expected": [], "actual": sorted(r.get("tools", [])),  # 仍记录实际输出供人参考
                             "ok": None, "missing": [], "extra": []}  # ok=None：三态中的"不适用"，不参与通过判定

        # ---- ② 意图 ----
        if case.expect_intent:  # 只有声明了期望意图的用例才跑这一面（省耗时）
            it = self.sut.intent(case.query)  # 执行意图被测面
            total_ms += float(it.get("latency_ms", 0))  # 累加意图耗时
            actual["intent"] = it.get("intent", "")  # 记录实际意图
            if it.get("error"):  # 意图执行炸了：维度记"不适用+错误"，不算通过也不算不通过
                dims["intent"] = {"expected": case.expect_intent, "actual": "",
                                  "ok": None, "error": it["error"]}  # ok=None：执行问题不该算成效果问题
            else:  # 正常执行：相等判定
                dims["intent"] = {"expected": case.expect_intent,
                                  "actual": it.get("intent", ""),
                                  "ok": it.get("intent", "") == case.expect_intent}  # 严格相等判定

        # ---- ③ 风控 ----
        rk = self.sut.risk(case.query)  # 执行风控被测面（每条都跑：任何输入都该过风控）
        actual["risk_passed"] = rk.get("passed")  # 记录实际放行与否
        actual["risk_level"] = rk.get("level", "")  # 记录风险等级
        if rk.get("error"):  # 风控执行炸了
            dims["risk"] = {"expected": case.expect_risk_pass, "actual": None,
                            "ok": None, "error": rk["error"]}  # 同样按"不适用+错误"处理
        else:  # 正常执行：布尔相等判定
            dims["risk"] = {"expected": case.expect_risk_pass,
                            "actual": rk.get("passed"),
                            "ok": bool(rk.get("passed")) == case.expect_risk_pass}  # 该放行的放了吗、该拦的拦了吗

        # ---- ④ 提示词组装（默认开）----
        if self.check_prompt:  # 由构造开关决定是否逐条评测 prompt 层
            bp = self.sut.build_prompt(case.query)  # 执行提示词组装被测面
            total_ms += float(bp.get("latency_ms", 0))  # 累加组装耗时
            actual["prompt_tokens"] = bp.get("prompt_tokens", 0)  # 记录完整提示词 token 数
            actual["trimmed"] = bp.get("trimmed", [])  # 记录被裁剪的区块
            if bp.get("error"):  # 组装炸了：这是实打实的失败（模板渲染都过不了），ok=False
                dims["prompt"] = {"expected": "渲染成功且不超预算", "actual": "",
                                  "ok": False, "error": bp["error"]}  # 渲染失败算效果失败——prompt 层是自家代码
            else:  # 渲染成功：判定是否守住预算
                within = bp.get("context_tokens", 0) <= self.max_context_tokens  # 预算红线比较
                dims["prompt"] = {  # prompt 维度判定
                    "expected": f"context <= {self.max_context_tokens}",  # 期望：不超红线
                    "actual": bp.get("context_tokens", 0),  # 实际：上下文 token 数
                    "ok": within,  # 守住则通过
                }

        return CaseResult(case.id, case.query, list(case.tags), dims, actual,  # 打包本条全部结果
                          latency_ms=round(total_ms, 1))  # 总耗时保留 1 位小数

    # ------------------------------------------------------------------
    # 【整批跑】
    #   返回可直接归档的报告 dict（也是基线文件格式）
    # ------------------------------------------------------------------
    # 【什么】跑完整批用例并汇总
    # 【作用】返回可直接归档的报告 dict（meta / summary / by_tag / failures 四段）
    # 【为什么】汇总口径集中在这一处：通过率、截断、verbose 明细的规则若散落在各处，
    #           不同入口跑出来的报告会互相矛盾
    def run(self, cases: List[EvalCase], verbose: bool = False) -> Dict:  # 整批入口
        board = ScoreBoard()  # 新建记分板：一次 run 一块，多次 run 互不污染
        results: List[CaseResult] = []  # 全量结果暂存（算通过率与挑失败明细用）

        for c in cases:  # 逐条跑
            res = self.run_case(c)  # 单条执行 + 判定
            results.append(res)  # 暂存结果

            # 执行异常：只记 errors，不进分母（否则环境抖动会被算成效果下降）
            if res.error:  # 本条执行炸了
                board.skill.add(None)  # 三个基础维度各记一次"异常"：只进 errors 不进分母
                board.risk.add(None)  # 同上
                board.intent.add(None)  # 同上
                continue  # 跳过后续正常累计

            board.skill.add(res.dims.get("skill", {}).get("ok"))  # 技能维度计入（route 成功才到这，skill 维度必有值）
            board.risk.add(res.dims.get("risk", {}).get("ok"))  # 风控维度计入；ok 可能是 None（风控异常），由 add 内部归到 errors
            if "intent" in res.dims:  # 本条跑了意图才计
                board.intent.add(res.dims["intent"].get("ok"))  # 意图维度计入
            if res.dims.get("tools", {}).get("ok") is not None:  # 本条校验了工具才计
                exp = set(c.expect_tools)  # 期望集合
                act = set(res.actual.get("tools", []))  # 实际集合
                board.add_tools(exp, act)  # 计入 P/R/F1 序列
            if self.check_prompt and "prompt" in res.dims:  # 开了 prompt 评测且本条有结果
                board.add_prompt(  # 计入 prompt 分布统计
                    int(res.actual.get("prompt_tokens", 0)),  # token 数
                    bool(res.dims["prompt"].get("ok")),  # 预算是否守住
                    res.latency_ms,  # 本条总耗时
                )

        passed = sum(1 for r in results if r.ok)  # 全维度通过条数
        failures = [r for r in results if not r.ok][:_MAX_FAIL_DETAIL]  # 失败明细截断到上限

        report: Dict[str, Any] = {  # 组装报告 dict（这份结构就是基线文件格式）
            "meta": {  # 元信息段：口径标注全在这
                "mode": self.sut.mode,  # 运行档位（offline/online）：解读报告可信度的关键
                "cases": len(cases),  # 用例总数
                "passed": passed,  # 全维度通过数
                "pass_rate": round(passed / len(cases), 4) if cases else 0.0,  # 通过率；空用例集防除零
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 生成时间戳
                "python": platform.python_version(),  # 解释器版本：复现环境参考
                "notes": list(self.sut.notes),  # 装配降级说明原样带上
            },
            "summary": board.summary(),  # 记分板汇总段
            "by_tag": _tag_stats(results),  # 分组统计段
            "failures": [r.to_dict() for r in failures],  # 失败明细段（已截断）
        }
        if verbose:  # 详细模式：附全量用例明细（体积大，排查专用）
            report["cases"] = [r.to_dict() for r in results]  # 全量序列化
        return report  # 返回报告 dict


# 【什么】按标签分组统计
# 【作用】报告里能看出"是计算类还行但知识库类崩了"，而不是只有一个总分
# 【为什么】总分掩盖结构问题：40 条里某一类全挂、其余全对时总分仍有 0.9，
#           分组统计让短板无处遁形
def _tag_stats(results: List[CaseResult]) -> Dict[str, Dict]:  # 入参全量结果，出参 {标签: 统计}
    agg: Dict[str, Dict[str, List]] = {}  # 中间累加结构：{标签: {维度: [布尔序列]}}
    for r in results:  # 逐条归入各标签桶
        for t in (r.tags or ["(untagged)"]):  # 无标签的进默认桶；一条多标签会同时进多个桶
            d = agg.setdefault(t, {"skill": [], "risk": [], "intent": [], "all": []})  # 取桶，不存在则建空桶
            if "skill" in r.dims and r.dims["skill"].get("ok") is not None:  # 技能维度有有效判定
                d["skill"].append(bool(r.dims["skill"]["ok"]))  # 入技能序列
            if "risk" in r.dims and r.dims["risk"].get("ok") is not None:  # 风控维度有有效判定
                d["risk"].append(bool(r.dims["risk"]["ok"]))  # 入风控序列
            if "intent" in r.dims and r.dims["intent"].get("ok") is not None:  # 意图维度有有效判定
                d["intent"].append(bool(r.dims["intent"]["ok"]))  # 入意图序列
            d["all"].append(bool(r.ok))  # 整体通过与否无条件入序列

    def rate(xs: List[bool]) -> float:  # 局部小工具：布尔序列求通过率，空序列给 0 防除零
        return round(sum(1 for x in xs if x) / len(xs), 4) if xs else 0.0  # 通过数 / 总数，保留 4 位

    return {  # 字典推导：每桶算出最终统计
        tag: {
            "cases": len(d["all"]),  # 该桶用例数
            "pass_rate": rate(d["all"]),  # 该桶全维度通过率
            "skill_acc": rate(d["skill"]),  # 该桶技能准确率
            "risk_acc": rate(d["risk"]),  # 该桶风控准确率
            "intent_acc": rate(d["intent"]),  # 该桶意图准确率
        }
        for tag, d in sorted(agg.items())  # 按标签名排序：报告输出稳定可 diff
    }
