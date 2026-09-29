"""
任务规划器（Plan-and-Execute）
==============================
给 ReAct 主链路补上"先规划、再执行"的能力：复杂流程先拆成有序步骤，
再逐步执行，每步结果入账，走偏或失败时只重排**剩下**的步骤。

【为什么 ReAct 之外还要这一层】
    ReAct 是"走一步看一步"：每一步只看当前状态决定下一步，没有全局视角。
    多步任务上它的典型病灶：
        - 偏航：中间某步结果把话题带跑，最后答非所问
        - 打转：换个参数反复调同一个工具（项目里靠去重/预算硬挡，属事后刹车）
        - 漏步：用户要三件事，做完两件就收尾
    Plan-and-Execute 的解法：先花一次 LLM 把"要做什么、分几步、谁先谁后"定下来，
    执行阶段每一步都带着"我在第几步、上一步得到什么"去问模型。

【完整数据流（规划怎么嵌进 ReAct 循环）】
    1. prepare  → 清洗、召回记忆/RAG/技能（与改造前一致）
    2. 判定     → should_plan：这条 query 是不是"复杂流程"？不是就直接 reason（零开销）
    3. plan 节点→ 调 LLM 拆步骤 → ExecutionPlan（id/instruction/tool_hint/depends_on）
                  按 depends_on 拓扑排序，保证被依赖的步骤先执行
    4. reason   → prompt 里注入"计划进度"（总思路 + 已完成 + 当前步 + 未做）
                  模型每一步只围绕当前步决定调什么工具
    5. act      → 执行工具，结果写回 observation（与改造前一致）
    6. 推进     → 当前步标记 done/failed，结果进 task_results，指针移到下一步
    7. 失败     → 步骤失败且开了重规划 → 只重排剩余步骤（已完成的成果保留）
    8. 收敛     → 所有步骤做完 → prompt 提示"计划已完成，请汇总作答"→ 进 answer

【三个设计目的】
    1. 只在复杂流程上启用：加一次规划就是加一次 LLM 调用，简单问题不该付这个钱
    2. 计划是数据不是文本：步骤是结构化对象（可序列化、可断点续跑、可观测），
       不是往 prompt 里塞一段"请你先做 A 再做 B"的自由文本
    3. 任何一环坏了都退回 ReAct：规划失败/解析失败/LLM 不可用 → 降级成"不拆步骤"，
       行为与改造前一致（新机制是增强，不是依赖）

【输入输出对照】
    planner.needs_planning("查一下北京天气，然后写一首关于它的诗")  → True（含"然后"）
    planner.needs_planning("3*7 等于多少")                          → False（单步）
    planner.create_plan(query, tools_desc="calculator: 计算...")
        → ExecutionPlan(rationale="先查天气再写诗", steps=[PlanStep(id="s1",...), ...])
    plan.progress_text()
        → "## 执行计划（共 2 步，已完成 1）\n- [x] s1: 查北京天气 → 晴 26℃\n- [ ] s2: 写诗（当前步骤）"
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from modules.common.logger import get_logger
from modules.common.validator import repair_json

_logger = get_logger("planner")

# 【步骤状态】pending 待做 / done 已完成 / failed 失败 / skipped 重规划时废弃
STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# 【复杂度信号词】命中即说明用户要的是"一串动作"，不是一个答案
_SEQ_WORDS = ("然后", "接着", "之后", "随后", "最后", "接下来", "其次", "再", "先")
_SUMMARY_WORDS = ("分别", "汇总", "对比", "比较", "统计", "整理", "总结", "报告", "调研",
                  "方案", "计划", "流程", "评估", "分析", "列出", "一步步", "逐步")
_STEP_MARKS = ("第一步", "第二步", "第三步", "1.", "2.", "3.", "①", "②", "③")
# 并列标点：出现得越多，说明一句话里塞了好几件事
_MULTI_MARKS = ("；", ";", "、", "，然后", "以及", "并且")

# 【给模型的输出契约】只保留"字段名→类型"，token 省一半以上（与 multiagent 同做法）
_PLAN_SCHEMA = ('{"rationale": "string", "steps": [{"id": "string", "instruction": "string", '
                '"tool_hint": "string", "depends_on": ["string"]}]}')


# 【什么】计划里的一步
# 【作用】结构化的一步：做什么、建议用什么工具、依赖谁、做完得到什么
@dataclass
class PlanStep:
    """执行计划中的一个步骤。

    【字段】
        id           步骤编号，如 "s1"；重规划时对不上号的旧步骤会被标 skipped
        instruction  这一步要做什么（给模型看的一句话指令）
        tool_hint    建议的工具名，只是提示，模型仍按工具清单自己决定
        depends_on   依赖的步骤 id，决定执行顺序（先做被依赖的）
        status       pending / done / failed / skipped
        result       本步产出的结果摘要（工具结果的前 200 字）
    """

    id: str = ""
    instruction: str = ""
    tool_hint: str = ""
    depends_on: List[str] = field(default_factory=list)
    status: str = STATUS_PENDING
    result: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "tool_hint": self.tool_hint,
            "depends_on": list(self.depends_on or []),
            "status": self.status,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanStep":
        d = d or {}
        return cls(
            id=str(d.get("id", "") or ""),
            instruction=str(d.get("instruction", "") or ""),
            tool_hint=str(d.get("tool_hint", "") or ""),
            depends_on=list(d.get("depends_on", []) or []),
            status=str(d.get("status", STATUS_PENDING) or STATUS_PENDING),
            result=str(d.get("result", "") or ""),
        )

    @property
    def is_open(self) -> bool:
        return self.status == STATUS_PENDING


# 【什么】一份执行计划
# 【作用】ReAct 循环全程读它：现在第几步、前面做完了什么、还剩什么
@dataclass
class ExecutionPlan:
    """一次运行的执行计划。

    【为什么计划要带状态而不是只带清单】
        计划不是"生成完就完事"的文档：执行过程中要能标记某步完成、某步失败、
        重规划后替换剩余步骤。带状态的计划才能支撑"走到一半改计划"。

    【输入输出对照】
        plan = ExecutionPlan(query="查天气再写诗", steps=[s1, s2])
        plan.current()            → s1（第一个 pending 的步骤）
        plan.mark_done("s1", "晴") → s1.status=done, s1.result="晴"
        plan.is_complete()        → False（s2 还没做）
        plan.progress_text()      → 给 prompt 用的进度文本
    """

    query: str = ""
    rationale: str = ""                              # 为什么这么拆（日志与 prompt 展示）
    steps: List[PlanStep] = field(default_factory=list)
    replan_count: int = 0                            # 已经重规划过几次

    # 【什么】当前该做的那一步
    # 【作用】reason 注入 prompt、act 推进计划都以它为准
    def current(self) -> Optional[PlanStep]:
        for step in self.steps:
            if step.is_open:
                return step
        return None

    # 【什么】按 id 找步骤
    # 【作用】标记完成/失败时定位
    def get(self, step_id: str) -> Optional[PlanStep]:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    # 【什么】标记一步的结果
    # 【作用】done 记结果摘要，failed 记失败原因；两者都不是"待做"
    def mark(self, step_id: str, status: str, result: str = "") -> None:
        step = self.get(step_id)
        if step is None:
            return
        step.status = status
        step.result = str(result or "")[:200]

    # 【什么】是否全部完成
    # 【作用】完成了就提示模型汇总作答，不再继续拆工具调用
    #   【为什么 failed 也算"不再做"】失败已经在 observation 里告诉模型了，
    #     算待办的话计划永远完不成，模型会一直停在"还有步骤没做"的状态里
    def is_complete(self) -> bool:
        closed = (STATUS_DONE, STATUS_SKIPPED, STATUS_FAILED)
        return bool(self.steps) and all(s.status in closed for s in self.steps)

    # 【什么】还剩几步没做
    # 【作用】观测与重规划判断用
    def pending(self) -> List[PlanStep]:
        return [s for s in self.steps if s.is_open]

    def done(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == STATUS_DONE]

    # 【什么】渲染成注入 prompt 的计划进度文本
    # 【作用】让模型每轮都看见"全局到哪了、我现在该干嘛、前面得到了什么"
    #   这是 Plan-and-Execute 相对裸 ReAct 的关键增量：模型不再只看上一步观察
    def progress_text(self) -> str:
        if not self.steps:
            return ""
        total = len(self.steps)
        done_n = len(self.done())
        box = {"done": "[x]", "failed": "[!]", "skipped": "[-]", "pending": "[ ]"}
        lines = []
        for s in self.steps:
            mark = box.get(s.status, "[ ]")
            tail = f" → {s.result}" if s.result else ""
            hint = f"（当前步骤）" if s.is_open and s is self.current() else ""
            lines.append(f"- {mark} {s.id}: {s.instruction}{tail}{hint}")
        cur = self.current()
        head = f"## 执行计划（共 {total} 步，已完成 {done_n}）\n总体思路：{self.rationale or '（未给出）'}\n"
        if cur is None:
            # 【为什么完成时要显式提示汇总】不提示的话模型可能继续找工具调用，
            #   表现为"做完了还在调工具"，最后一轮被 max_iterations 硬截断
            head += "状态：所有步骤已完成，请基于下面的结果汇总出最终答案，不要再调用新的工具。\n"
        else:
            head += f"当前步骤：{cur.id} —— 请只围绕这一步决定要不要调工具、调哪个工具。\n"
        return head + "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "rationale": self.rationale,
            "steps": [s.to_dict() for s in self.steps],
            "replan_count": self.replan_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionPlan":
        d = d or {}
        return cls(
            query=str(d.get("query", "") or ""),
            rationale=str(d.get("rationale", "") or ""),
            steps=[PlanStep.from_dict(s) for s in (d.get("steps", []) or [])],
            replan_count=int(d.get("replan_count", 0) or 0),
        )


# 【什么】任务规划器
# 【作用】判定要不要规划、生成计划、失败后重排剩余步骤
class TaskPlanner:
    """任务规划器：把复杂流程拆成有序步骤，并在走偏时重排剩余步骤。

    【三条降级路径】规划不是必经环节，任一步出问题都退回"不拆步骤"：
        LLM 不可用 / 返回非法 JSON / 步骤解析为空 → 返回单步计划（等同不规划）
        重规划次数用尽                            → 保留原计划的剩余步骤继续做
        开关关闭                                  → needs_planning 恒为 False
    """

    def __init__(self, llm: Any = None, enabled: bool = True, min_score: int = 2,
                 max_steps: int = 6, replan_enabled: bool = True, max_replans: int = 1):
        self.llm = llm
        self.enabled = bool(enabled)
        self.min_score = max(1, int(min_score))
        self.max_steps = max(1, int(max_steps))
        self.replan_enabled = bool(replan_enabled)
        self.max_replans = max(0, int(max_replans))

    # ------------------------------------------------------------------
    # ① 这条 query 值不值得规划？
    # ------------------------------------------------------------------
    # 【什么】复杂度打分：够复杂才规划
    # 【作用】避免"3*7 等于多少"也去拆一次计划——那是一次纯亏的 LLM 调用
    #   【打分规则（累计 ≥ min_score 才规划）】
    #     序列词（然后/接着/最后/先…再…）  +2   最直接的"多步"信号
    #     汇总词（对比/汇总/统计/报告/方案）+1   要综合多个结果
    #     步骤标记（第一步/1. /①）          +2   用户自己就写成了流程
    #     并列标点（；/、/以及）≥2 处        +1   一句话里塞了好几件事
    #     长度 ≥ 40 字                      +1   长问句通常是复合需求
    #     技能 SOP 本身是多步的（含 1. 2.）  +1   流程型技能天然要分步
    def needs_planning(self, query: str, skill_context: str = "") -> bool:
        if not self.enabled:
            return False
        q = str(query or "").strip()
        if len(q) < 8:                                   # 太短：问好、简单计算，不必规划
            return False

        score = 0
        reasons: List[str] = []
        if any(w in q for w in _SEQ_WORDS):
            score += 2
            reasons.append("含序列词")
        if any(w in q for w in _SUMMARY_WORDS):
            score += 1
            reasons.append("含汇总/对比词")
        if any(m in q for m in _STEP_MARKS):
            score += 2
            reasons.append("自带步骤标记")
        if sum(q.count(m) for m in _MULTI_MARKS) >= 2:
            score += 1
            reasons.append("并列项多")
        if len(q) >= 40:
            score += 1
            reasons.append("长问句")
        if skill_context and "1." in skill_context and "2." in skill_context:
            score += 1
            reasons.append("技能 SOP 是多步的")

        hit = score >= self.min_score
        _logger.info(f"规划判定: score={score}/{self.min_score} → {'规划' if hit else '不规划'} | {reasons}")
        return hit

    # ------------------------------------------------------------------
    # ② 生成计划
    # ------------------------------------------------------------------
    # 【什么】调一次 LLM 把 query 拆成有序步骤
    # 【作用】产出 ExecutionPlan；解析不出来就降级成"单步计划"（行为等同不规划）
    def create_plan(self, query: str, tools_desc: str = "", skill_context: str = "") -> ExecutionPlan:
        """生成执行计划。

        【输入输出对照】
            in   query="查北京天气，然后写一首关于它的诗", tools_desc="weather: 查天气"
            out  ExecutionPlan(rationale="先取事实再创作",
                               steps=[s1(查天气,tool_hint=weather), s2(写诗,depends_on=[s1])])
            in   LLM 返回乱码
            out  ExecutionPlan(steps=[单步 instruction=query])   ← 降级，不抛异常
        """
        plan = ExecutionPlan(query=str(query or ""))
        if self.llm is None:
            return self._fallback_plan(plan, "LLM 不可用")

        prompt = (
            "你是一个任务规划器。请把下面这个请求拆成有序的执行步骤。\n\n"
            f"## 用户请求\n{query}\n\n"
            f"## 可用工具\n{tools_desc or '（无可用工具）'}\n\n"
            f"## 作业流程（可选）\n{skill_context or '（无）'}\n\n"
            "## 输出要求\n"
            f"只输出一段 JSON，不要任何解释文字，格式如下：\n{_PLAN_SCHEMA}\n"
            f"约束：\n"
            f"1. steps 最多 {self.max_steps} 步，能一步做完就不要拆\n"
            "2. instruction 是一句话的具体动作，不要写'然后'这类连接词\n"
            "3. tool_hint 填工具名，用不上工具就填空字符串\n"
            "4. depends_on 填它依赖的步骤 id（如 s2 要用 s1 的结果就填 ['s1']），无依赖填空数组\n"
            "5. 必须按依赖顺序排列：被依赖的步骤排在前面\n"
            "6. rationale 用一句话说明为什么这么拆\n"
        )
        try:
            raw = self.llm.invoke(prompt, system_prompt="你是严谨的任务规划器，只输出 JSON。")
        except Exception as e:
            return self._fallback_plan(plan, f"规划调用失败: {e}")

        data = self._parse_json(raw)
        if not isinstance(data, dict) or not data.get("steps"):
            return self._fallback_plan(plan, "规划结果为空或不是对象")

        plan.rationale = str(data.get("rationale", "") or "")[:200]
        steps: List[PlanStep] = []
        for i, item in enumerate(list(data.get("steps") or [])[: self.max_steps], start=1):
            if not isinstance(item, dict):
                continue
            instruction = str(item.get("instruction", "") or "").strip()
            if not instruction:
                continue
            steps.append(PlanStep(
                id=str(item.get("id", "") or f"s{i}"),
                instruction=instruction,
                tool_hint=str(item.get("tool_hint", "") or ""),
                depends_on=[str(x) for x in (item.get("depends_on") or []) if x],
            ))
        if not steps:
            return self._fallback_plan(plan, "步骤解析为空")

        plan.steps = self._sort_steps(steps)          # 按 depends_on 排成可执行的先后顺序
        _logger.info(f"计划已生成: {len(plan.steps)} 步 | rationale={plan.rationale[:40]}")
        return plan

    # ------------------------------------------------------------------
    # ③ 重规划：只重排剩下的
    # ------------------------------------------------------------------
    # 【什么】某步做完/失败后，重排剩余步骤
    # 【作用】已完成的成果保留，只替换没做的部分——不是推倒重来
    def replan(self, plan: ExecutionPlan, observation: str = "", tools_desc: str = "") -> ExecutionPlan:
        """重排剩余步骤。

        【为什么不整份重生成】
            已完成步骤的结果是有成本的（调过工具、花过 token），
            推倒重来等于把已经拿到的信息丢掉，还可能把做过的副作用再做一遍。

        【什么时候不重排】
            开关关闭 / 次数已用尽 / LLM 不可用 → 原样返回，让模型继续按原计划做
        """
        if not self.replan_enabled or plan.replan_count >= self.max_replans:
            _logger.info(f"跳过重规划: enabled={self.replan_enabled} "
                         f"count={plan.replan_count}/{self.max_replans}")
            return plan
        if self.llm is None:
            return plan

        done_steps = plan.done()
        failed_steps = [s for s in plan.steps if s.status == STATUS_FAILED]
        # 【哪些要重排】没做的（pending）+ 做失败的（failed）都算"没达成"，
        #   只留下真正做成的那部分
        remaining = plan.pending() + failed_steps
        if not remaining:
            return plan

        prompt = (
            "原计划执行到一半需要调整。请只重新规划**尚未完成**的步骤。\n\n"
            f"## 原始请求\n{plan.query}\n\n"
            f"## 已完成步骤\n" + "\n".join(f"- {s.id}: {s.instruction} → {s.result}" for s in done_steps) + "\n\n"
            f"## 待重排的步骤\n" + "\n".join(f"- {s.id}: {s.instruction}" for s in remaining) + "\n\n"
            f"## 最新观察\n{observation or '（无）'}\n\n"
            f"## 可用工具\n{tools_desc or '（无可用工具）'}\n\n"
            f"## 输出要求\n只输出 JSON，格式：{_PLAN_SCHEMA}\n"
            f"约束：最多 {self.max_steps} 步；不要重复已完成步骤；id 用 r1、r2 依次编号。\n"
        )
        try:
            raw = self.llm.invoke(prompt, system_prompt="你是严谨的任务规划器，只输出 JSON。")
        except Exception as e:
            _logger.warning(f"重规划调用失败，沿用原计划: {e}")
            return plan

        data = self._parse_json(raw)
        if not isinstance(data, dict) or not data.get("steps"):
            _logger.warning("重规划结果不可用，沿用原计划")
            return plan

        new_steps: List[PlanStep] = []
        for i, item in enumerate(list(data.get("steps") or [])[: self.max_steps], start=1):
            if not isinstance(item, dict):
                continue
            instruction = str(item.get("instruction", "") or "").strip()
            if not instruction:
                continue
            new_steps.append(PlanStep(
                id=str(item.get("id", "") or f"r{plan.replan_count + 1}_{i}"),
                instruction=instruction,
                tool_hint=str(item.get("tool_hint", "") or ""),
            ))
        if not new_steps:
            return plan

        # 【为什么旧步骤标 skipped 而不是删掉】
        #   删掉会让"曾经规划过这一步但没做"这件事从记录里消失，
        #   排查时看到的就是一份干净但失真的计划（与多 Agent 重规划同一个原则）
        #   【failed 为什么不改成 skipped】它失败过这件事本身就是关键信息，
        #   保留 failed 状态，后面才能看出"重规划是因为哪一步挂了"
        for s in remaining:
            if s.status == STATUS_PENDING:
                s.status = STATUS_SKIPPED
        plan.steps = done_steps + failed_steps + new_steps
        plan.replan_count += 1
        _logger.info(f"已重规划: 保留 {len(done_steps)} 步完成项，重排 {len(new_steps)} 步 "
                     f"（第 {plan.replan_count} 次）")
        return plan

    # ------------------------------------------------------------------
    # 内部：解析与排序
    # ------------------------------------------------------------------
    # 【什么】把 LLM 的输出解析成 dict
    # 【作用】先直接 json.loads，失败再用 repair_json 兜"近似 JSON"，都失败返回 None
    @staticmethod
    def _parse_json(raw: Any) -> Optional[dict]:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        if "```" in text:                              # 模型爱用代码块包 JSON，先剥壳
            text = text.split("```")[1] if "```" in text else text
            text = text.replace("json", "", 1).strip()
        try:
            return repair_json(text)
        except Exception as e:
            _logger.warning(f"规划结果 JSON 解析失败: {str(e)[:100]} | raw={text[:120]}")
            return None

    # 【什么】按 depends_on 做拓扑排序
    # 【作用】保证被依赖的步骤排在前面；依赖成环或引用不存在的 id 时按原顺序兜底
    @staticmethod
    def _sort_steps(steps: List[PlanStep]) -> List[PlanStep]:
        by_id = {s.id: s for s in steps}
        ordered: List[PlanStep] = []
        placed = set()
        # 【为什么最多循环 len(steps) 轮】有环时 Kahn 算法排不完，
        #   用轮数兜底防止死循环；剩下的按原顺序补在后面
        for _ in range(len(steps)):
            progress = False
            for s in steps:
                if s.id in placed:
                    continue
                deps = [d for d in s.depends_on if d in by_id]      # 引用不存在的 id 直接忽略
                if all(d in placed for d in deps):
                    ordered.append(s)
                    placed.add(s.id)
                    progress = True
            if not progress:
                break
        for s in steps:                                              # 环里的/没排上的，补在最后
            if s.id not in placed:
                ordered.append(s)
        return ordered

    # 【什么】降级成"单步计划"
    # 【作用】没有计划时 ReAct 照常跑：current() 返回这一步，做完就算完成
    @staticmethod
    def _fallback_plan(plan: ExecutionPlan, reason: str) -> ExecutionPlan:
        _logger.warning(f"规划降级（按单步执行）: {reason}")
        plan.rationale = "规划不可用，按单步执行"
        plan.steps = [PlanStep(id="s1", instruction=plan.query or "回答用户的问题")]
        return plan
