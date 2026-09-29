"""
委派（Delegation）模块
======================
让执行者在接到任务后，能把"更适合别人做的那部分"正式转给别人：
真实执行、产出回传、并入委派方的结论，全程受治理约束。

【委派 vs 派活（dispatch）】
    派活：主管在规划阶段一次性把任务分下去 —— 事前、中心化、只有一层
    委派：执行者接到活之后发现"这块不该我干"，当场转出去
          —— 事中、横向、可递归，产出回传给委派方并入它的结论

    举例：主管派给 analyst_1「算 Q4 环比」，
          analyst_1 发现自己缺 Q3 原始数据 → 委派给 researcher_1「查 Q3/Q4 营收」
          → researcher_1 真去查了 → 结果回传 → analyst_1 拿数据算完再交差。
    没有委派能力时，analyst_1 只能硬着头皮编一个数，或者整单失败。

【与 AgentMessage(DELEGATE) 的分工】
    消息层那条 delegate 只是"说了一句这块你来做"（且指挥权归主管，worker 发会被拒），
    它不执行、不回传、不受治理；
    本模块的委派是一次真实调用：有输入、有产出、有记录、有边界。
    委派成功后会往总线补一条 FINDING，让队友看到"这块已经转给谁了"，避免重复做。

【完整数据流（一次委派从头到尾 8 步）】
    1. Agent.run 拿到任务，先看自己 can_delegate 且委派已启用
    2. 做一次结构化决策 → DelegationDecision（要不要转 / 转给谁 / 转什么）
    3. DelegationPolicy.can() 过五道闸：开关 → 资格 → 存在性 → 深度/次数 → 防环
    4. 放行后 ledger.open() 开一条 DelegationRecord（先记账，再执行）
    5. DelegationExecutor.execute() 构造子任务，真调受托 Agent 的 run()
    6. 受托方执行时自己也可以再委派（depth+1），直到深度上限
    7. ledger.close() 回填产出/失败原因与耗时
    8. 受托产出并入委派方 prompt 的「委派产出」段，委派方据此出自己的结论

【五道闸（任一不过 = 不委派，主流程照常往下走）】
    ① 开关    delegation_enabled
    ② 资格    委派方 spec.can_delegate；delegate_to 白名单非空时还要在名单里
    ③ 存在性  受托方必须在花名册里、且处于启用状态
    ④ 边界    depth < max_depth、总次数 < max_per_run、单人次数 < max_per_agent
    ⑤ 防环    受托方不能是自己，也不能已经在委派链上（A→B→A 直接拦掉）

【被拒绝的委派要不要留痕】
    要。ledger.denials 全部记下来：排查"为什么这块没人接"时，
    "根本没想转"和"想转但被治理拦了"是两件完全不同的事。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from modules.common.logger import get_logger
from modules.multiagent.models import (
    AgentOutput,
    AgentSpec,
    DelegationDecision,
    DelegationRecord,
    SubTask,
    TeamConfig,
    json_schema_hint,
)

_logger = get_logger("delegation")


# 【什么】委派治理策略
# 【作用】把四条边界（开关/深度/总次数/单人次数）装成一个可传可测的小对象
class DelegationPolicy(BaseModel):
    """委派的四条边界。默认关闭：不启用时 can() 一律返回 False。"""

    enabled: bool = Field(default=False, description="是否启用委派")
    max_depth: int = Field(default=1, ge=0, description="最多转派几层")
    max_per_run: int = Field(default=6, ge=0, description="单次运行最多委派几次")
    max_per_agent: int = Field(default=2, ge=0, description="单个 Agent 最多委派几次")

    # 【什么】从团队配置生成策略
    # 【作用】配置集中放在 TeamConfig 里，这里只做一次搬运，避免两处默认值打架
    @classmethod
    def from_config(cls, cfg: TeamConfig) -> "DelegationPolicy":
        return cls(
            enabled=bool(getattr(cfg, "delegation_enabled", False)),
            max_depth=int(getattr(cfg, "delegation_max_depth", 1) or 0),
            max_per_run=int(getattr(cfg, "delegation_max_per_run", 6) or 0),
            max_per_agent=int(getattr(cfg, "delegation_max_per_agent", 2) or 0),
        )


# 【什么】一次运行的委派台账
# 【作用】记账（成功/拒绝）+ 计数（总数/单人）+ 防环（委派链）
class DelegationLedger:
    """一次运行的委派台账。

    【为什么台账要按 run 建而不是全局】
        委派次数上限是"这一轮问题"的预算，不是这个 Agent 一生的预算。
        按 run 建，下一轮提问自动归零，配额不会被上一轮吃掉
        （与 harness 每轮 orchestrator.reset() 是同一个道理）。

    【输入输出对照】
        ledger.open(DelegationRecord(from_agent="a", to_agent="b", ...)) → 记账并返回同一条
        ledger.close(rec, AgentOutput(success=True, content="营收 1200 万"))
            → rec.success=True, rec.content="营收 1200 万", 耗时已填
        ledger.count("a") → 1（a 已经转出去 1 次）
    """

    def __init__(self, policy: DelegationPolicy, roster: Optional[dict[str, AgentSpec]] = None):
        self.policy = policy
        self.roster: dict[str, AgentSpec] = dict(roster or {})   # {Agent 名: spec}，校验存在性用
        self.records: list[DelegationRecord] = []
        self.denials: list[dict] = []                            # 被拒绝的尝试（含原因）

    # ------------------------------------------------------------------
    # 准入：这次转派能不能做
    # ------------------------------------------------------------------
    # 【什么】五道闸的准入判定
    # 【作用】返回 (能不能转, 原因)；不过也记进 denials，便于事后排查
    def can(self, delegator: str, delegatee: str, depth: int = 0,
            chain: Optional[list[str]] = None) -> tuple[bool, str]:
        """判断一次委派能否成立。返回 (是否放行, 中文原因)。

        【为什么返回中文原因而不是只返回 bool】
            委派被拒时要把原因写进 trace 和日志：
            "没转派"和"想转但被拦"在排查时是两件事，
            只留一个 False，事后只能靠猜。
        """
        p = self.policy
        chain = list(chain or [])

        def deny(reason: str) -> tuple[bool, str]:
            self.denials.append({"from": delegator, "to": delegatee, "depth": depth, "reason": reason})
            _logger.info(f"委派被拒: {delegator}→{delegatee} | {reason}")
            return False, reason

        if not p.enabled:
            return deny("委派未启用")
        if delegator == delegatee:
            return deny("不能委派给自己")
        if delegatee in chain:
            # 【为什么要防环】A 转给 B、B 又转回 A 是最典型的死循环，
            #   两个 Agent 会互相甩锅直到耗光步数预算，
            #   而且看起来"都在正常工作"，日志里全是正常执行
            return deny(f"委派成环（链上已有 {delegatee}）")
        if depth >= p.max_depth:
            return deny(f"已达委派深度上限 {p.max_depth}")
        if len(self.records) >= p.max_per_run:
            return deny(f"已达单次运行委派上限 {p.max_per_run}")
        if self.count(delegator) >= p.max_per_agent:
            return deny(f"{delegator} 已达单人委派上限 {p.max_per_agent}")

        spec = self.roster.get(delegator)
        if spec is not None:
            if not spec.can_delegate:
                return deny(f"{delegator} 未被授权委派（can_delegate=False）")
            if spec.delegate_to and delegatee not in spec.delegate_to:
                return deny(f"{delegatee} 不在 {delegator} 的委派白名单内")

        target = self.roster.get(delegatee)
        if target is None:
            return deny(f"受托方 {delegatee} 不在花名册")
        if not target.enabled:
            return deny(f"受托方 {delegatee} 已停用")
        return True, "放行"

    # 【什么】某个 Agent 已经转出去几次
    # 【作用】单人配额用；只数"真执行了"的记录，被拒的不算（没干活不该占额度）
    def count(self, agent: str) -> int:
        return sum(1 for r in self.records if r.from_agent == agent)

    # 【什么】可以转派给谁
    # 【作用】给委派决策当候选名单：排除自己、排除主管、排除停用的人
    #   【为什么要排除主管】主管是指挥者，把活转回主管等于"甩回给派活的人"，
    #     会形成 主管→A→主管 的无效回路；真需要主管重拆，走 replan 而不是委派
    def candidates(self, exclude: str = "") -> list[AgentSpec]:
        from modules.multiagent.models import AgentRole
        return [
            s for name, s in self.roster.items()
            if name != exclude and s.enabled and s.role != AgentRole.SUPERVISOR
        ]

    # 【什么】开一条委派记录
    # 【作用】先记账再执行：执行中途崩了也留得下"这次转派发生过"
    def open(self, rec: DelegationRecord) -> DelegationRecord:
        self.records.append(rec)
        return rec

    # 【什么】回填委派结果
    # 【作用】把受托方产出/失败原因与耗时写回记录，供 trace 与最终答案使用
    def close(self, rec: DelegationRecord, out: Optional[AgentOutput], started: float) -> DelegationRecord:
        rec.duration_ms = int((time.time() - started) * 1000)
        if out is None:
            rec.success, rec.error = False, "委派未产出结果"
            return rec
        rec.success = bool(out.success)
        rec.content = (out.content or "")[:2000]      # 台账里留摘要，不把全文塞进结果
        rec.error = "" if out.success else (out.error or "委派执行失败")
        return rec

    # 【什么】本次运行委派情况的汇总
    # 【作用】给日志与 RunResult 用：转了几次、成了几次、被拒几次
    def summary(self) -> dict:
        return {
            "total": len(self.records),
            "ok": sum(1 for r in self.records if r.success),
            "failed": sum(1 for r in self.records if not r.success),
            "denied": len(self.denials),
            "max_depth": max((r.depth for r in self.records), default=0),
        }


# 【什么】委派执行器
# 【作用】把"转出去的那部分活"真正跑一遍：造子任务 → 调受托 Agent → 回填台账
class DelegationExecutor:
    """委派执行器：受托方真跑一次任务，产出回传给委派方。

    【为什么执行器只拿一个 run_fn 回调，而不是直接依赖 harness】
        harness 依赖本模块（它要建台账），本模块再依赖 harness 就是循环导入。
        传一个"给定 Agent 名与 SubTask 就返回 AgentOutput"的回调进来，
        两边都只依赖 models，装配关系由 harness 决定。

    【输入输出对照】
        ex = DelegationExecutor(run_fn=..., ledger=..., roster=...)
        rec = ex.delegate(from_agent="analyst_1", to_agent="researcher_1",
                          instruction="查 Q3/Q4 营收", parent_task_id="t2", depth=0)
        → DelegationRecord(from_agent="analyst_1", to_agent="researcher_1",
                           success=True, content="Q3 1100 万，Q4 1200 万", duration_ms=812)
    """

    def __init__(self, run_fn: Callable[[str, SubTask, dict], AgentOutput],
                 ledger: DelegationLedger, roster: Optional[dict[str, AgentSpec]] = None,
                 on_event: Optional[Callable[[DelegationRecord, str], None]] = None,
                 context_factory: Optional[Callable[[int, list[str]], Any]] = None):
        self._run_fn = run_fn                     # (agent_name, task, extra) → AgentOutput
        self.ledger = ledger
        self.roster: dict[str, AgentSpec] = dict(roster or {})
        self.on_event = on_event                  # 委派发生时的回调（harness 用它写 trace）
        # 【什么】给受托方造一个"带委派深度的上下文"
        # 【作用】受托方自己还能不能再往下转，取决于 depth+1 是否还够，
        #         这个信息只有执行器知道，所以由它来构造
        self._context_factory = context_factory

    # 【什么】本执行器是否启用了委派
    # 【作用】Agent 靠它在"问模型要不要转派"之前先短路：
    #   未启用时连那次决策调用都不该发生（委派关闭 = 零额外开销）
    @property
    def enabled(self) -> bool:
        return bool(self.ledger.policy.enabled)

    # 【什么】可以转派给谁
    # 【作用】给委派决策当候选名单；Agent 侧通过它问"我能转给谁"
    def candidates(self, exclude: str = "") -> list[AgentSpec]:
        return self.ledger.candidates(exclude)

    # 【什么】把活转给别人并拿回结果
    # 【作用】委派的主入口：校验 → 记账 → 执行 → 回填；被拒时返回 None 让调用方继续自己干
    def delegate(self, from_agent: str, to_agent: str, instruction: str,
                 parent_task_id: str = "", depth: int = 0, reason: str = "",
                 chain: Optional[list[str]] = None) -> Optional[DelegationRecord]:
        """执行一次委派。被治理拒绝时返回 None（委派方照常自己干）。

        【为什么被拒返回 None 而不是抛异常】
            委派是"锦上添花"的能力：转不出去，原任务照样要做。
            抛异常会让整个子任务失败，等于为一个可选能力赔上主流程。
        """
        chain = list(chain or [])
        ok, why = self.ledger.can(from_agent, to_agent, depth=depth, chain=chain)
        if not ok:
            if self.on_event is not None:
                try:
                    self.on_event(None, why)     # 拒绝也要进 trace：排查"为什么没人接"靠它
                except Exception as e:
                    _logger.debug(f"委派事件回调异常（忽略）: {e}")
            return None

        rec = DelegationRecord(
            from_agent=from_agent, to_agent=to_agent, instruction=instruction,
            reason=reason, parent_task_id=parent_task_id, depth=depth,
            chain=chain + [from_agent],
        )
        self.ledger.open(rec)
        # 【子任务 id 为什么要带委派痕迹】
        #   委派产出的 task_id 形如 "t2→researcher_1"，
        #   trace 里一眼能看出它不是主管派的原任务，而是中途转出来的
        task = SubTask(id=f"{parent_task_id or 'task'}>{to_agent}", assignee=to_agent,
                       instruction=instruction)
        started = time.time()
        try:
            extra = {
                "delegated_by": from_agent,
                "delegation_depth": depth + 1,          # 受托方再往下转就是第 depth+1 层
                "delegation_chain": rec.chain,
            }
            ctx = self._context_factory(depth + 1, rec.chain) if self._context_factory else None
            out = self._run_fn(to_agent, task, {"extra": extra, "ctx": ctx})
        except Exception as e:
            _logger.error(f"委派执行异常: {from_agent}→{to_agent}: {type(e).__name__}: {e}")
            out = AgentOutput.failed(to_agent, task.id, f"{type(e).__name__}: {e}")
        self.ledger.close(rec, out, started)
        _logger.info(f"委派完成: {rec.label} | {'成功' if rec.success else '失败'} "
                     f"| depth={depth} | {rec.duration_ms}ms")
        if self.on_event is not None:
            try:
                self.on_event(rec, "委派完成" if rec.success else "委派失败")
            except Exception as e:
                _logger.debug(f"委派事件回调异常（忽略）: {e}")
        return rec


# 【什么】让 Agent 自己判断"这块要不要转给别人"
# 【作用】把委派决策做成结构化输出：模型只在格子里填空，填一半自动降级为不委派
def decide_delegation(llm: Any, instruction: str, roster: list[AgentSpec],
                      system_prompt: str = "") -> Optional[DelegationDecision]:
    """问模型一次"要不要转派"。解析失败返回 None（= 不委派）。

    【为什么这是一次额外 LLM 调用，还要不要做】
        成本确实存在，所以默认关闭、且只有 can_delegate=True 的 Agent 才会问。
        收益是：analyst 缺数据时能当场找 researcher 要，
        而不是编一个数或者整单失败——后者要么污染答案，要么白跑一轮团队。
    """
    if llm is None or not roster:
        return None
    roster_text = "\n".join(
        f"- {a.name}（{a.role.value}）: {a.description or '通用执行者'}" for a in roster
    )
    prompt = (
        "## 你手上的任务\n{instruction}\n\n"
        "## 可选的同伴\n{roster}\n\n"
        "## 判断规则\n"
        "- 只有当你确实缺少信息/能力、且某位同伴更适合做其中一部分时，才转派\n"
        "- 转派的 instruction 必须具体到能直接执行，不要写'帮我看看'\n"
        "- 自己能做就自己做完，不要为了省事转派\n\n"
        "## 输出结构（严格遵守）\n{schema}\n\n"
        "只输出一个 JSON 对象。"
    ).format(instruction=instruction, roster=roster_text,
             schema=json_schema_hint(DelegationDecision))
    try:
        data = llm.invoke_structured(prompt, system_prompt=system_prompt)
        return DelegationDecision.model_validate(data)
    except Exception as e:
        _logger.warning(f"委派决策解析失败（按不委派处理）: {e}")
        return None
