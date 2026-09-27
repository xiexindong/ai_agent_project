"""
中心化编排 + 标准消息协议 + 统一观测链路 演示
==============================================
展示多 Agent 系统怎么做到「可控、可观测、可管控、可承载」四件事。

全程离线：LLM 用脚本化的 ScriptedLLM 顶替，不连任何外部服务。
想接真模型，把 ScriptedLLM() 换成 LLMClient() 即可（需配 DEEPSEEK_API_KEY）。

运行：
    python examples/orchestration_demo.py

六幕：
    第一幕  标准协议：信封长什么样、回复怎么自动 +1 跳
    第二幕  权限与安全：转派权只归主管、敏感词打码/拦截、参数上限
    第三幕  负载控制：令牌桶限流、单轮配额、指数退避重试
    第四幕  防死循环：A、B 踢皮球被升级给主管（不是无限来回）
    第五幕  统一观测：一次运行的完整 span 树 + 统计 + JSON 导出
    第六幕  编排器大盘：消息投递 / 升级 / 配额 / 限流一览
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.multiagent import (  # noqa: E402
    AgentRole, AgentSpec, MessageBus, MessageKind, TeamConfig,
)
from modules.multiagent.harness import AgentHarness  # noqa: E402
from modules.orchestration import (  # noqa: E402
    DeliveryStatus, Envelope, GuardPolicy, Orchestrator,
    OrchestratorConfig, PolicyGuard, Quota, RateLimiter, RejectCode,
    RouteAction, TokenBucket, Tracer, retry_with_backoff,
)


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
        if "团队成员产出" in prompt or "汇总" in prompt:
            return "汇总：营收 1200 万，同比 +8%。"
        if "查营收" in prompt:
            return "营收 1200 万，来自订单系统。"          # 与下面矛盾，用来制造冲突
        if "算增长" in prompt:
            return "营收 800 万，同比 -5%，参考内部代号 A-7。"   # 含敏感词，用来演示打码
        return "已完成。"

    def invoke_structured(self, prompt: str, system_prompt: str = "",
                          temperature: float = 0.2) -> dict:
        if "待审草稿" in prompt:
            return {"passed": True, "score": 0.9, "issues": [], "suggestion": "可发布"}
        return {
            "thought": "拆成两个并行任务",
            "subtasks": [
                {"id": "t1", "assignee": "researcher_1", "instruction": "查营收"},
                {"id": "t2", "assignee": "analyst_1", "instruction": "算增长"},
            ],
            "direct_answer": "",
            "needs_review": True,
        }

    def invoke_with_tools(self, messages: list, tools: list, temperature: float = 0.3) -> dict:
        return {"content": "（脚本模型不调工具）", "tool_calls": []}


# ---------------------------------------------------------------------------
# 第一幕：标准协议
# ---------------------------------------------------------------------------
def act1_protocol():
    line("第一幕  标准协议：每条消息都是一个信封")

    q = Envelope(sender="analyst_1", receiver="researcher_1",
                 content="你的口径是含税还是不含税？", kind=MessageKind.CLARIFY,
                 sender_role="worker", trace_id="trace_demo_1")
    print(f"  发起: {q.sender} → {q.receiver} [{q.kind.value}] hop={q.hop} "
          f"size_bytes={q.size_bytes} delivery={q.delivery.value}")

    a = q.reply("researcher_1", "是不含税口径")
    print(f"  回复: {a.sender} → {a.receiver} hop={a.hop} parent_id={a.parent_id[:12]}…")
    print(f"        跳数由协议自动 +1（{q.hop} → {a.hop}），调用方填不了也漏不掉")

    # 广播与点对点的可见性
    bus = MessageBus()
    bus.post_envelope(q)
    bus.post_envelope(a)
    print(f"  researcher_1 的收件箱: {bus.inbox_text('researcher_1') or '（空，自己发的看不到）'}")
    print(f"  analyst_1 的收件箱:    {bus.inbox_text('analyst_1')}")


# ---------------------------------------------------------------------------
# 第二幕：权限与安全
# ---------------------------------------------------------------------------
def act2_guard():
    line("第二幕  权限与安全：能不能发，由编排器说了算")

    bus = MessageBus()
    guard = PolicyGuard(GuardPolicy(known_agents=["researcher_1", "analyst_1", "supervisor"],
                                    sensitive_words=["内部代号"], on_sensitive="redact"))
    orch = Orchestrator(bus=bus, guard=guard, supervisor="supervisor")

    cases = [
        ("worker 转派任务", dict(sender="analyst_1", content="这块你来做",
                                 receiver="researcher_1", kind=MessageKind.DELEGATE,
                                 sender_role="worker")),
        ("主管转派任务", dict(sender="supervisor", content="这块你来做",
                             receiver="researcher_1", kind=MessageKind.DELEGATE,
                             sender_role="supervisor")),
        ("发给不存在的人", dict(sender="analyst_1", content="在吗", receiver="ghost",
                               kind=MessageKind.CLARIFY, sender_role="worker")),
        ("正文含敏感词", dict(sender="analyst_1", content="结果参考内部代号 A-7",
                             receiver="researcher_1", kind=MessageKind.FINDING,
                             sender_role="worker")),
    ]
    for label, kw in cases:
        d = orch.send(**kw)
        flag = "放行" if d.delivered else f"拦下（{d.code.value}）"
        shown = d.envelope.content[:32]
        print(f"  {label:14s} → {flag:22s} 正文: {shown}")

    # 同一个策略换成 reject：敏感词不再打码，而是整条拒绝
    strict = Orchestrator(
        bus=MessageBus(),
        guard=PolicyGuard(GuardPolicy(known_agents=["researcher_1", "analyst_1"],
                                      sensitive_words=["内部代号"], on_sensitive="reject")),
    )
    d = strict.send("analyst_1", "结果参考内部代号 A-7", receiver="researcher_1",
                    sender_role="worker")
    print(f"  策略改 reject     → {'放行' if d.delivered else '拦下（' + d.code.value + '）'}")


# ---------------------------------------------------------------------------
# 第三幕：负载控制
# ---------------------------------------------------------------------------
def act3_control():
    line("第三幕  负载控制：限流、配额、重试都在编排层")

    # 限流：桶空了就拒
    bucket = TokenBucket(capacity=3, refill_per_sec=1.0)
    results = [bucket.try_consume() for _ in range(4)]
    print(f"  令牌桶(capacity=3) 连取 4 次 → {results}")
    print("        第 4 次拿不到令牌：这就是「总量可控」")

    # 双层限流：单发送者不会霸占全局
    limiter = RateLimiter(capacity=100, refill_per_sec=100,
                          per_sender_capacity=2, per_sender_refill=0.0)
    print(f"  双层限流 同一人连发 3 次 → "
          f"{[limiter.acquire('a')[0] for _ in range(3)]}（第 3 次被发送者桶拦下）")
    print(f"  换个人再发             → {limiter.acquire('b')[0]}（全局桶还有令牌）")

    # 配额：单轮单人上限
    quota = Quota(per_run=100, per_round=30, per_sender_round=2)
    for i in range(3):
        ok, why = quota.check("analyst_1", round_no=0, hop=0)
        print(f"  配额 第 {i + 1} 次 → {'放行' if ok else '丢弃：' + why}")
        if ok:
            quota.consume("analyst_1", 0)

    # 重试：前两次失败第三次成功
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("下游 503")
        return "拿到结果"

    print(f"  指数退避重试 → {retry_with_backoff(flaky, attempts=3, base_delay=0.01)!r}"
          f"（共调用 {state['n']} 次）")


# ---------------------------------------------------------------------------
# 第四幕：防死循环
# ---------------------------------------------------------------------------
def act4_loop_guard():
    line("第四幕  防死循环：A、B 来回踢皮球会被升级给主管")

    bus = MessageBus()
    orch = Orchestrator(
        bus=bus,
        guard=PolicyGuard(GuardPolicy(known_agents=["a", "b", "supervisor"])),
        config=OrchestratorConfig(max_hops=2, pingpong_limit=2),
        supervisor="supervisor",
    )
    print("  模拟：a 催 → b 拖 → a 再催 → b 再拖 ……（跳数上限 2）")
    chain = [
        ("a", "数据好了吗", "b"),
        ("b", "还差一点", "a"),
        ("a", "到底好了没", "b"),
        ("b", "马上", "a"),
    ]
    parent = ""
    for sender, content, receiver in chain:
        d = orch.send(sender, content, receiver=receiver, sender_role="worker",
                      parent_id=parent)
        parent = d.envelope.id
        tag = "投递" if d.delivered else f"{d.action.value}（{d.code.value}）"
        print(f"    {sender} → {receiver}: {content:8s} hop={d.envelope.hop} "
              f"实际收件人={d.envelope.receiver:10s} 结果={tag}")

    print(f"\n  升级记录 {len(orch.escalations)} 条：")
    for e in orch.escalations:
        print(f"    - {e['sender']}→{e['receiver']} 跳数={e['hop']} 原因={e['reason']}")


# ---------------------------------------------------------------------------
# 第五幕：统一观测
# ---------------------------------------------------------------------------
def act5_observability():
    line("第五幕  统一观测：一次运行的完整链路")

    team = TeamConfig(
        agents=[
            AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR),
            AgentSpec(name="researcher_1", role=AgentRole.RESEARCHER),
            AgentSpec(name="analyst_1", role=AgentRole.ANALYST),
            AgentSpec(name="reviewer", role=AgentRole.REVIEWER),
        ],
        orchestration_sensitive_words="内部代号",   # 命中即打码
        consensus_enabled=True,
        review_enabled=True,
    )
    h = AgentHarness(config=team, llm=ScriptedLLM())
    r = h.run("我司上季度营收是多少，同比增长如何？")

    print(f"  状态={r.status.value} 耗时={r.duration_ms}ms trace_id={r.trace_id}")
    print(f"  最终答案: {r.final_answer[:60]}…")
    print(f"  一致性判定: {r.consensus.note if r.consensus else '（未触发）'}")
    print("\n  --- span 树（线上按 trace_id 拉出来的就是这棵）---")
    for ln in h.last_trace.summary().splitlines():
        print("    " + ln)

    print("\n  --- 聚合统计 ---")
    stats = h.last_trace.stats()
    print(f"    span 总数={stats['span_count']} 按状态={stats['by_status']}")
    for kind, v in stats["by_kind"].items():
        print(f"    {kind:10s} 次数={v['count']} 总耗时={v['total_ms']}ms 最慢={v['max_ms']}ms")

    print("\n  --- 导出 JSON（可上报外部平台，前 240 字）---")
    print("    " + h.last_trace.to_json().replace("\n", " ")[:240] + "…")
    print(f"\n  RunResult 自带链路: trace_id={r.trace_id[:12]}… spans={len(r.spans)} 条")
    return h


# ---------------------------------------------------------------------------
# 第六幕：编排器大盘
# ---------------------------------------------------------------------------
def act6_stats(h: AgentHarness):
    line("第六幕  编排器大盘：消息投递、升级、配额、限流")

    s = h.orchestrator.stats()
    print(f"  当前轮次       = {s['round']}")
    print(f"  已投递消息     = {s['delivered']}")
    print(f"  被升级的消息   = {s['escalated']}")
    print(f"  配额使用       = {s['quota']['total']} / {s['quota']['limits']['per_run']}")
    print(f"  限流剩余令牌   = 全局 {s['limiter']['global_available']}")
    for name, left in s["limiter"]["senders"].items():
        print(f"                   {name}: {left}")

    print("\n  总线里每条消息的最终状态：")
    for m in h.bus.all(limit=20):
        print(f"    [{m.delivery.value:9s}] {m.sender} → {m.receiver} "
              f"[{m.kind.value}] hop={m.hop} {m.content[:34]}")


def main():
    print("中心化编排演示（全程离线，LLM 由脚本顶替）")
    act1_protocol()
    act2_guard()
    act3_control()
    act4_loop_guard()
    h = act5_observability()
    act6_stats(h)
    line("收尾")
    print("  四件事各自的落点：")
    print("    可控   → Orchestrator（跳数上限 + 重复检测 + 乒乓检测 → 升级主管）")
    print("    可观测 → Tracer（span 树 + 耗时 + 输入输出 + JSON 导出）")
    print("    安全   → PolicyGuard（权限矩阵 + 敏感词 + 参数上限）")
    print("    负载   → RateLimiter / Quota / retry_with_backoff（都在编排层）")


if __name__ == "__main__":
    main()
