"""
Human-in-the-Loop 与 多 Agent 协调 演示
========================================
展示「Agent 跑到一半停下来等人」和「多个 Agent 怎么收敛」两套机制。

全程离线：LLM 用脚本化的 ScriptedLLM 顶替，不连任何外部服务。
想接真模型，把 ScriptedLLM() 换成 LLMClient() 即可（需配 DEEPSEEK_API_KEY）。

运行：
    python examples/hitl_coordination_demo.py

五幕：
    第一幕  HITL 基础：敏感工具被拦下（人点拒绝 / 批准 / 改参数）
    第二幕  HITL 超时：人一直不来，按默认动作兜底（不是挂死）
    第三幕  消息总线：成员之间横向对话，异议进收件箱
    第四幕  一致性 + 仲裁：两份矛盾产出，审核者出裁决
    第五幕  重规划：有人失败，主管只针对未完成的部分重新派活
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.hitl import (  # noqa: E402
    HumanAction,
    HumanReply,
    InterruptKind,
    auto_resolver,
    build_gate,
    scripted_resolver,
)
from modules.multiagent import (  # noqa: E402
    AgentOutput,
    AgentRole,
    AgentSpec,
    Consensus,
    MessageBus,
    MessageKind,
    ReplanController,
    TeamConfig,
    text_overlap,
)
from modules.multiagent.harness import AgentHarness  # noqa: E402


def line(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


# ---------------------------------------------------------------------------
# 脚本化 LLM：按 prompt 里的关键词返回预设内容，代替真实模型
# ---------------------------------------------------------------------------
class ScriptedLLM:
    """按关键词返回固定内容的假模型。让演示可复现、零成本。"""

    def invoke(self, prompt: str, system_prompt: str = "", temperature: float = 0.3,
               max_tokens: int = 2048) -> str:
        if "裁决" in prompt:
            return "裁决：采用 1200 万这一口径（有订单系统数据支撑），800 万缺少来源说明。"
        if "团队成员产出" in prompt:
            return "汇总：营收 1200 万，同比 +8%。【摘要】营收 1200 万，同比增长 8%。"
        if "查营收" in prompt:
            return "营收 1200 万，来自订单系统。"
        if "算增长" in prompt:
            return "营收 800 万，同比 -5%。"          # 与上面矛盾，用来制造冲突
        if "查库存" in prompt:
            raise RuntimeError("库存接口 503")        # 用来演示失败与重规划
        return "已完成。"

    def invoke_structured(self, prompt: str, system_prompt: str = "",
                          temperature: float = 0.2) -> dict:
        # 终审请求：返回 ReviewVerdict 的结构，别把规划结构给它
        if "待审草稿" in prompt:
            return {"passed": True, "score": 0.9, "issues": [], "suggestion": "可发布"}
        # 重规划请求里会出现"执行现状"，此时只补派未完成的任务
        if "执行现状" in prompt:
            return {
                "thought": "只补派失败的那一项",
                "subtasks": [
                    {"id": "t2retry", "assignee": "analyst_1", "instruction": "算增长"},
                ],
                "direct_answer": "",
                "needs_review": True,
            }
        return {
            "thought": "拆成两个并行任务",
            "subtasks": [
                {"id": "t1", "assignee": "researcher_1", "instruction": "查营收"},
                {"id": "t2", "assignee": "analyst_1", "instruction": "查库存"},
            ],
            "direct_answer": "",
            "needs_review": True,
        }

    def invoke_with_tools(self, messages: list, tools: list, temperature: float = 0.3) -> dict:
        return {"content": "（脚本模型不调工具）", "tool_calls": []}


# ---------------------------------------------------------------------------
# 第一幕：敏感工具被人拦下 / 放行 / 改参数
# ---------------------------------------------------------------------------
def act1_basic_interrupt():
    line("第一幕  HITL：敏感工具调用前的人工批准")

    # 人的三种不同答复，分别看结果
    cases = [
        ("人点拒绝", auto_resolver(HumanAction.REJECT)),
        ("人点批准", auto_resolver(HumanAction.APPROVE)),
        ("人改参数", auto_resolver(HumanAction.MODIFY, edited=json.dumps(
            {"table": "orders", "limit": 10}, ensure_ascii=False))),
    ]
    for label, resolver in cases:
        gate = build_gate(
            enabled=True, sensitive_tools=["data_query", "mysql_*"], resolver=resolver,
        )
        go, args = gate.ask_tool("data_query", {"table": "users", "limit": 1000})
        print(f"  {label:6s} → 放行={go!s:5s} 最终参数={args}")

    # 非敏感工具：根本不会打断（零开销）
    gate = build_gate(enabled=True, sensitive_tools=["data_query"],
                      resolver=auto_resolver(HumanAction.REJECT))
    go, args = gate.ask_tool("calculator", {"expr": "3*7"})
    print(f"  非敏感工具 calculator → 放行={go}（未产生任何中断请求，"
          f"累计请求数={len(gate.pending()) + len(gate.records())}）")


# ---------------------------------------------------------------------------
# 第二幕：人一直不来 —— 超时兜底，而不是挂死
# ---------------------------------------------------------------------------
def act2_timeout_fallback():
    line("第二幕  HITL：无人响应时按默认动作兜底")

    # 不注入 resolver，等 1 秒必定超时
    gate = build_gate(enabled=True, sensitive_tools=["data_query"], timeout_seconds=1.0)
    # 工具类默认兜底是 reject（宁可不做，也不能做错）
    go, _ = gate.ask_tool("data_query", {"table": "users"})
    print(f"  BEFORE_TOOL  超时 → 放行={go}（默认 reject）")

    # 终稿类默认兜底是 approve（答案已经过汇总与终审，卡住比放行更糟）
    go, text = gate.ask_final("这是最终答案草稿")
    print(f"  BEFORE_FINAL 超时 → 放行={go}（默认 approve）")
    print(f"  归档记录：{[(r.final_action.value, r.request.kind.value) for r in gate.records()]}")


# ---------------------------------------------------------------------------
# 第三幕：消息总线
# ---------------------------------------------------------------------------
def act3_message_bus():
    line("第三幕  协调：成员之间的横向对话")

    bus = MessageBus()
    bus.post("researcher_1", "营收原始数据已确认：1200 万", kind=MessageKind.FINDING)
    bus.post("analyst_1", "你的口径是含税还是不含税？", receiver="researcher_1",
             kind=MessageKind.CLARIFY)
    bus.post("consensus", "你的结论与多数派不一致", receiver="analyst_1",
             kind=MessageKind.OBJECTION)

    print("  researcher_1 的收件箱：")
    for m in bus.inbox("researcher_1"):
        print(f"    [{m.kind.value}] 来自 {m.sender}: {m.content}")
    print("  analyst_1 的收件箱：")
    for m in bus.inbox("analyst_1"):
        print(f"    [{m.kind.value}] 来自 {m.sender}: {m.content}")
    print(f"  异议条数={len(bus.objections())}（只作为仲裁输入，不触发新的执行轮次）")


# ---------------------------------------------------------------------------
# 第四幕：一致性检测 + 仲裁
# ---------------------------------------------------------------------------
def act4_consensus_and_arbitration():
    line("第四幕  协调：产出打架时的检测与仲裁")

    outputs = [
        AgentOutput(agent="researcher_1", role=AgentRole.RESEARCHER, task_id="t1",
                    content="营收 1200 万，来自订单系统"),
        AgentOutput(agent="analyst_1", role=AgentRole.ANALYST, task_id="t2",
                    content="营收 800 万，同比 -5%"),
    ]
    cons = Consensus(threshold=0.35).detect(outputs)
    print(f"  两份产出相似度 = {text_overlap(outputs[0].content, outputs[1].content)}")
    print(f"  判定: has_conflict={cons.has_conflict} needs_arbitration={cons.needs_arbitration}")
    print(f"  说明: {cons.note}")

    # 一致性产出（换个说法但结论一致）
    same = [
        AgentOutput(agent="a", content="营收 1200 万，同比 +8%"),
        AgentOutput(agent="b", content="营收 1200 万，同比 +8%，来自订单系统"),
    ]
    c2 = Consensus(threshold=0.35).detect(same)
    print(f"  结论一致时: has_conflict={c2.has_conflict} 多数派={c2.majority}")

    # 仲裁：用脚本模型顶替审核者
    from modules.multiagent.agents import ReviewerAgent
    reviewer = ReviewerAgent(AgentSpec(name="reviewer_1", role=AgentRole.REVIEWER),
                             llm=ScriptedLLM())
    from modules.multiagent import Arbitrator
    arb = Arbitrator(judge=reviewer).arbitrate("公司今年营收是多少？", outputs, cons)
    print(f"  仲裁结论: {arb.content if arb else '（无仲裁者）'}")


# ---------------------------------------------------------------------------
# 第五幕：完整跑一次 —— 失败、重规划、冲突、人工介入一起发生
# ---------------------------------------------------------------------------
def act5_full_run():
    line("第五幕  端到端：一次运行里同时出现失败、重规划、冲突与人工裁决")

    # 人只在"冲突裁决"这一处介入，并且直接给出最终答案；其余环节自动放行
    gate = build_gate(
        enabled=True,
        sensitive_tools=["data_query"],
        resolver=scripted_resolver(
            {"on_dispute": HumanAction.MODIFY},
            fallback=HumanAction.APPROVE,
        ),
    )
    # MODIFY 需要带 edited 内容，这里用自动答复器补上"人改后的答案"
    base_resolver = gate.resolver

    def resolver_with_edit(req):
        reply = base_resolver(req)
        if reply.action == HumanAction.MODIFY:
            reply.edited = "人工裁决：以 1200 万为准，800 万缺少来源，已标注待核实。"
        return reply

    gate.resolver = resolver_with_edit

    cfg = TeamConfig(
        agents=[
            AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR, description="拆任务与汇总"),
            AgentSpec(name="researcher_1", role=AgentRole.RESEARCHER, description="查资料"),
            AgentSpec(name="analyst_1", role=AgentRole.ANALYST, description="算数据"),
            AgentSpec(name="reviewer_1", role=AgentRole.REVIEWER, description="审核"),
        ],
        max_workers=2, timeout_seconds=30,
        replan_enabled=True,      # 打开重规划（默认关闭，因为要多调一次 LLM）
        max_replans=1,
        hitl_enabled=True,
    )
    harness = AgentHarness(config=cfg, llm=ScriptedLLM(), gate=gate)

    started = time.time()
    result = harness.run("公司今年营收是多少？", session_id="demo")
    print(f"  状态={result.status.value} 耗时={int((time.time()-started)*1000)}ms "
          f"重规划={result.replan_count} 次 人工介入={result.interrupt_count} 次")
    print(f"  产出数={len(result.outputs)} 失败者={result.failed_agents}")
    if result.consensus:
        print(f"  一致性: conflict={result.consensus.has_conflict} "
              f"similarity={result.consensus.similarity}")
    print(f"  最终答案: {result.final_answer}")
    print("  关键 trace：")
    for e in result.trace:
        if e.kind in ("replan", "consensus", "arbitration", "hitl", "done"):
            print(f"    [{e.kind}] {e.message[:70]}")


def main() -> None:
    print("Human-in-the-Loop 与 多 Agent 协调 演示（全程离线，LLM 由脚本顶替）")
    act1_basic_interrupt()
    act2_timeout_fallback()
    act3_message_bus()
    act4_consensus_and_arbitration()
    act5_full_run()
    print("\n" + "=" * 62)
    print("说明：接真模型把 ScriptedLLM() 换成 LLMClient()；")
    print("      生产环境把 gate 换成共享存储实现即可跨实例审批。")
    print("=" * 62)


if __name__ == "__main__":
    main()
