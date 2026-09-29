"""
多 Agent 委派（Delegation）演示
================================
执行者接到活之后，发现"这块更适合别人"，当场把这部分转出去：
真实执行 → 产出回传 → 并入自己的结论 → 全程受治理约束。

跑法：
    python examples/multiagent_delegation_demo.py

【委派 vs 派活】
    派活（dispatch）：主管在规划阶段一次性分下去 —— 事前、中心化、一层
    委派（delegation）：执行者中途转出去           —— 事中、横向、可递归

    典型场景：主管派 analyst_1「算 Q4 环比」，
              analyst_1 手上没有原始数据 → 委派 researcher_1「查 Q3/Q4 营收」
              → 数据回来 → analyst_1 算完交差。
    没有委派时，analyst_1 只能编一个数（污染答案）或者整单失败（白跑一轮）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.multiagent import (  # noqa: E402
    AgentHarness, AgentRole, AgentSpec, DelegationPolicy, DelegationRecord,
    SubTask, TeamConfig,
)

SEP = "=" * 68


# 【什么】脚本化的假 LLM
# 【作用】按 prompt 关键字应答：规划 / 委派决策 / 普通产出；不依赖真模型
class ScriptedLLM:
    def __init__(self, delegate: bool = True):
        self.delegate = delegate

    def invoke_structured(self, prompt: str, system_prompt: str = "", temperature: float = 0.2) -> dict:
        if "should_delegate" in prompt:                     # 委派决策
            return {
                "should_delegate": self.delegate,
                "to_agent": "researcher_1",
                "instruction": "查 2024 年 Q3 与 Q4 的营收原始数据",
                "reason": "我手上只有 Q4 数据，缺 Q3 无法算环比",
            }
        return {                                            # 主管规划
            "thought": "先算环比，缺的数据由分析员自己转派",
            "direct_answer": "", "needs_review": False,
            "subtasks": [{"id": "t1", "assignee": "analyst_1",
                          "instruction": "计算 Q4 相对 Q3 的环比增长率",
                          "depends_on": [], "priority": 1}],
        }

    def invoke(self, prompt: str, system_prompt: str = "", temperature: float = 0.3,
               max_tokens: int = 2048) -> str:
        if "你委派出去的工作产出" in prompt:                # 委派方拿到了数据
            return "Q3 营收 1100 万，Q4 营收 1200 万，环比增长 9.1%"
        if "汇总" in prompt:                                # 主管汇总
            return "2024 Q4 营收 1200 万，环比 Q3（1100 万）增长 9.1%。"
        return "Q3 营收 1100 万，Q4 营收 1200 万"          # 受托方产出

    def invoke_with_tools(self, messages, tools, temperature: float = 0.3) -> dict:
        return {"content": "（无工具）", "tool_calls": None}


def _team(**kw) -> TeamConfig:
    """主管 + 分析（有委派资格）+ 检索。委派开关由 kw 控制。"""
    return TeamConfig(
        agents=[
            AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR,
                      description="拆任务、派活、汇总"),
            AgentSpec(name="analyst_1", role=AgentRole.ANALYST,
                      description="做计算、对比、归因", can_delegate=True),
            AgentSpec(name="researcher_1", role=AgentRole.RESEARCHER,
                      description="查资料、找事实依据"),
        ],
        review_enabled=False, consensus_enabled=False, share_findings=False,
        **kw,
    )


def demo_why_delegation():
    print(SEP)
    print("第 1 段  为什么需要委派：同一件事，开与不开的差别")
    print(SEP)
    for enabled in (False, True):
        h = AgentHarness(config=_team(delegation_enabled=enabled), llm=ScriptedLLM())
        r = h.run("2024 Q4 营收环比增长多少？")
        got = [o for o in r.outputs if o.agent == "analyst_1"]
        print(f"\n委派开关={enabled}:")
        print(f"  委派次数     : {len(r.delegations)}")
        print(f"   analyst_1 结论: {(got[0].content if got else '（无产出）')[:60]}")
        print(f"   最终答案     : {r.final_answer[:60]}")
    print("\n开关关闭时 analyst_1 手上没有 Q3 数据，只能凭 Q4 硬答；")
    print("打开后它把查数据这块转给 researcher_1，拿到真实数据才算得出环比。")


def demo_full_chain():
    print()
    print(SEP)
    print("第 2 段  委派全过程：谁转给谁、带回什么、怎么并入结论")
    print(SEP)
    h = AgentHarness(config=_team(delegation_enabled=True), llm=ScriptedLLM())
    r = h.run("2024 Q4 营收环比增长多少？")

    for rec in r.delegations:
        print(f"  委派 {rec.label}: {rec.instruction}")
        print(f"    理由   : {rec.reason}")
        print(f"    深度   : {rec.depth} | 链: {'→'.join(rec.chain + [rec.to_agent])}")
        print(f"    结果   : {'成功' if rec.success else '失败'} | {rec.content or rec.error}")
    print("\n  trace 里的委派事件：")
    for e in r.trace:
        if e.kind == "delegate":
            print(f"    - {e.message} (from={e.agent}, to={e.payload.get('to')})")
    print(f"\n  analyst_1 产出的委派痕迹: "
          f"{len([o for o in r.outputs if o.agent == 'analyst_1'][0].delegations)} 条")


def demo_governance():
    print()
    print(SEP)
    print("第 3 段  治理五道闸：什么时候不让转")
    print(SEP)
    from modules.multiagent import DelegationLedger

    roster = {
        "analyst_1": AgentSpec(name="analyst_1", role=AgentRole.ANALYST, can_delegate=True),
        "researcher_1": AgentSpec(name="researcher_1", role=AgentRole.RESEARCHER),
        "intern": AgentSpec(name="intern", role=AgentRole.RESEARCHER, enabled=False),
    }
    cases = [
        ("未启用", DelegationPolicy(enabled=False), "analyst_1", "researcher_1", 0, []),
        ("转给自己", DelegationPolicy(enabled=True), "analyst_1", "analyst_1", 0, []),
        ("受托方停用", DelegationPolicy(enabled=True), "analyst_1", "intern", 0, []),
        ("受托方不存在", DelegationPolicy(enabled=True), "analyst_1", "ghost", 0, []),
        ("成环（回传给上游）", DelegationPolicy(enabled=True), "researcher_1", "analyst_1", 1, ["analyst_1"]),
        ("超过深度上限", DelegationPolicy(enabled=True, max_depth=1), "researcher_1", "analyst_1", 1, []),
    ]
    for title, policy, a, b, depth, chain in cases:
        lg = DelegationLedger(policy, roster=roster)
        ok, why = lg.can(a, b, depth=depth, chain=chain)
        print(f"  {title:<18} → {'放行' if ok else '拒绝'} | {why}")

    # 次数上限
    lg = DelegationLedger(DelegationPolicy(enabled=True, max_per_agent=1), roster=roster)
    lg.open(DelegationRecord(from_agent="analyst_1", to_agent="researcher_1"))
    ok, why = lg.can("analyst_1", "researcher_1")
    print(f"  {'超出单人上限':<18} → {'放行' if ok else '拒绝'} | {why}")


def demo_explicit_api():
    print()
    print(SEP)
    print("第 4 段  显式委派 API：代码/主管主动转派（治理与自动委派完全一致）")
    print(SEP)
    h = AgentHarness(config=_team(delegation_enabled=True), llm=ScriptedLLM(delegate=False))
    rec = h.delegate("analyst_1", "researcher_1",
                     "查 2024 年 Q3 与 Q4 的营收原始数据",
                     query="环比多少", reason="主管补数据")
    print(f"  显式委派: {rec.label if rec else '（被拒）'} | "
          f"{'成功' if rec and rec.success else '失败'}")
    if rec:
        print(f"  带回内容: {rec.content}")
        print(f"  耗时     : {rec.duration_ms}ms | 深度 {rec.depth}")
    print(f"  转给不存在的人: {h.delegate('analyst_1', 'ghost', '查数据', query='x')}")


def demo_vs_dispatch():
    print()
    print(SEP)
    print("第 5 段  委派 / 派活 / 转派消息 三者怎么区分")
    print(SEP)
    rows = [
        ("派活 dispatch", "主管、规划阶段", "一层，事前定好", "PlanResult.subtasks"),
        ("委派 delegation", "执行者、执行过程中", "可递归，有深度上限", "RunResult.delegations"),
        ("转派消息 DELEGATE", "只是一条消息", "不执行、不回传、不记账", "MessageBus"),
    ]
    print(f"  {'机制':<20}{'发起方/时机':<18}{'层数':<16}{'落在哪'}")
    for a, b, c, d in rows:
        print(f"  {a:<20}{b:<18}{c:<16}{d}")
    print("\n  记忆点：想让活真的被干掉，用委派；只想说一句'这块你来做'，用消息。")


if __name__ == "__main__":
    demo_why_delegation()
    demo_full_chain()
    demo_governance()
    demo_explicit_api()
    demo_vs_dispatch()
    print()
    print(SEP)
    print("演示结束。开关：MULTIAGENT_DELEGATION_ENABLED=true")
    print(SEP)
