"""
中心化编排测试：协议 / 权限 / 安全 / 负载 / 防环 / 观测
========================================================
这六项对应「多 Agent 能不能上生产」的治理要求：
    1. 可控     —— 防 A、B 无限来回（跳数上限 + 重复检测 + 乒乓检测）
    2. 可观测   —— 每次运行一条 trace，每个环节一个 span，可导出可断言
    3. 安全     —— 权限矩阵、敏感词、参数上限统一在编排器执行
    4. 负载     —— 限流、配额、重试、超时都在编排层

全部用例不依赖 LLM、不依赖数据库。
"""
import pytest

from modules.multiagent.harness import AgentHarness
from modules.multiagent.models import (
    AgentRole, AgentSpec, MessageKind, PlanResult, SubTask, TeamConfig,
)
from modules.multiagent.coordination import MessageBus
from modules.orchestration import (
    DeliveryStatus,
    Envelope,
    GuardPolicy,
    Orchestrator,
    OrchestratorConfig,
    PolicyGuard,
    Quota,
    RateLimiter,
    RejectCode,
    RouteAction,
    TokenBucket,
    Tracer,
    retry_with_backoff,
)


# 【什么】造一个编排器（含总线、权限、限流、配额）
# 【作用】用例只关心要验的那一层，其余用宽松默认值，避免互相干扰
def _orch(roster=("a", "b"), supervisor="supervisor", **orch_cfg) -> tuple:
    bus = MessageBus()
    guard = PolicyGuard(GuardPolicy(known_agents=list(roster) + [supervisor]))
    # 【为什么限流参数给这么大】
    #   本文件多数用例验的是权限与防环，不希望被限流"顺手"拦下，
    #   那会让断言的原因码变成 RATE_LIMITED，掩盖真正要验的逻辑。
    limiter = RateLimiter(capacity=1000, refill_per_sec=1000,
                          per_sender_capacity=1000, per_sender_refill=1000)
    quota = Quota(per_run=1000, per_round=1000, per_sender_round=1000,
                  max_hops=orch_cfg.get("max_hops", 10))
    orch = Orchestrator(bus=bus, guard=guard, limiter=limiter, quota=quota,
                        config=OrchestratorConfig(**orch_cfg), supervisor=supervisor)
    return bus, orch


# ---------------------------------------------------------------------------
# 1. 标准化协议
# ---------------------------------------------------------------------------
# 【什么】回复会自动 +1 跳并把接收方换回原发送者
# 【作用】hop 由协议负责累加，不让调用方自己填，否则防环上限形同虚设
def test_envelope_reply_increments_hop():
    q = Envelope(sender="a", receiver="b", content="口径是含税吗",
                 kind=MessageKind.CLARIFY, hop=0)
    a = q.reply("b", "是不含税口径")

    assert a.hop == 1                 # 回复 = 父消息 hop + 1
    assert a.receiver == "a"          # 默认回给发信方
    assert a.parent_id == q.id
    assert a.size_bytes == len("是不含税口径")


# 【什么】信封继承 AgentMessage，旧代码按老字段读依然成立
# 【作用】保证改造不破坏既有读写（bus.inbox / inbox_text）
def test_envelope_is_backward_compatible():
    env = Envelope(sender="a", receiver="b", content="hello")
    bus = MessageBus()
    bus.post_envelope(env)

    assert env.is_delivered                       # 直接入总线的补盖 DELIVERED
    assert [m.content for m in bus.inbox("b")] == ["hello"]


# ---------------------------------------------------------------------------
# 2. 权限与安全校验
# ---------------------------------------------------------------------------
# 【什么】转派权只归主管：worker 发 delegate 被拒
# 【作用】指挥权是权限矩阵里唯一收窄的一项，其余类型全员可发
def test_guard_rejects_delegate_from_worker():
    _, orch = _orch()
    d = orch.send("a", "你来做这块", receiver="b",
                  kind=MessageKind.DELEGATE, sender_role="worker")

    assert d.action == RouteAction.REJECT
    assert d.code == RejectCode.KIND_FORBIDDEN
    assert d.envelope.delivery == DeliveryStatus.REJECTED


# 【什么】同一条 delegate 换主管身份就放行
# 【作用】证明拦的是"角色"而不是"这条消息"
def test_guard_allows_delegate_from_supervisor():
    _, orch = _orch()
    d = orch.send("supervisor", "你来做这块", receiver="b",
                  kind=MessageKind.DELEGATE, sender_role="supervisor")

    assert d.delivered


# 【什么】敏感词默认打码放行，正文被改写但消息仍可用
# 【作用】既不外传敏感片段，也不因为一个词整条打断协作
def test_guard_redacts_sensitive_by_default():
    bus = MessageBus()
    guard = PolicyGuard(GuardPolicy(known_agents=["a", "b"], sensitive_words=["内部代号"],
                                    on_sensitive="redact"))
    orch = Orchestrator(bus=bus, guard=guard)
    d = orch.send("a", "参考内部代号 A-7 的数据", receiver="b", sender_role="general")

    assert d.delivered
    assert "内部代号" not in d.envelope.content
    assert "***" in d.envelope.content
    assert d.envelope.meta["redacted"] == ["内部代号"]


# 【什么】策略改成 reject 时命中敏感词直接拒绝
# 【作用】给"宁可断也不能漏"的合规场景一个硬开关
def test_guard_rejects_sensitive_when_policy_is_reject():
    bus = MessageBus()
    guard = PolicyGuard(GuardPolicy(known_agents=["a", "b"], sensitive_words=["内部代号"],
                                    on_sensitive="reject"))
    orch = Orchestrator(bus=bus, guard=guard)
    d = orch.send("a", "参考内部代号 A-7", receiver="b", sender_role="general")

    assert d.action == RouteAction.REJECT
    assert d.code == RejectCode.SENSITIVE


# 【什么】接收者不在花名册 / 正文超长 都会被拦
# 【作用】这两类是最常见的参数类问题，原因码要能区分
def test_guard_rejects_unknown_receiver_and_oversize():
    _, orch = _orch()
    d1 = orch.send("a", "在吗", receiver="ghost", sender_role="general")
    d2 = orch.send("a", "x" * 5001, receiver="b", sender_role="general")

    assert d1.code == RejectCode.RECEIVER_UNKNOWN
    assert d2.code == RejectCode.TOO_LARGE


# ---------------------------------------------------------------------------
# 3. 负载控制
# ---------------------------------------------------------------------------
# 【什么】令牌桶：容量内放行，超出拒绝
# 【作用】限流是"总量可控"的兜底，靠它防住下游被冲垮
def test_token_bucket_capacity():
    b = TokenBucket(capacity=2, refill_per_sec=0.0)

    assert [b.try_consume() for _ in range(3)] == [True, True, False]


# 【什么】双层限流：单体桶拦下时不会把全局令牌还回去
# 【作用】否则被限流的发送者可以无限重试，直到把全局桶也耗光
def test_rate_limiter_per_sender():
    limiter = RateLimiter(capacity=100, refill_per_sec=100,
                          per_sender_capacity=1, per_sender_refill=0.0)

    assert limiter.acquire("a")[0] is True
    ok, by = limiter.acquire("a")
    assert ok is False and by == "sender:a"
    assert limiter.acquire("b")[0] is True      # 换个人还有令牌


# 【什么】同一发送者单轮超配额被丢弃
# 【作用】防某个 Agent 在一轮里刷屏把上下文撑爆
def test_quota_per_sender_round():
    q = Quota(per_sender_round=1)
    assert q.check("a", 0, 0)[0] is True
    q.consume("a", 0)

    ok, why = q.check("a", 0, 0)
    assert ok is False and "上限" in why


# 【什么】指数退避重试：前两次失败第三次成功
# 【作用】重试是编排层的职责，Agent 内部自己重试会吃掉超时预算
def test_retry_with_backoff_succeeds_on_third():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    assert retry_with_backoff(fn, attempts=3, base_delay=0.001) == "ok"
    assert calls["n"] == 3


# 【什么】重试次数用尽后抛出最后一次异常
# 【作用】调用方拿得到真实原因，而不是被吞成 None
def test_retry_with_backoff_raises_last_error():
    with pytest.raises(RuntimeError):
        retry_with_backoff(lambda: (_ for _ in ()).throw(RuntimeError("always")),
                           attempts=2, base_delay=0.001)


# ---------------------------------------------------------------------------
# 4. 防死循环（可控的核心）
# ---------------------------------------------------------------------------
# 【什么】跳数超过上限：消息改投主管，不再回到原对手
# 【作用】A→B→A→B 到第 3 跳就被收口，token 不会一直烧
def test_max_hops_escalates_to_supervisor():
    bus, orch = _orch(max_hops=1)
    d1 = orch.send("a", "你的口径？", receiver="b", sender_role="general")
    d2 = orch.send("b", "不含税", receiver="a", sender_role="general",
                   parent_id=d1.envelope.id)
    d3 = orch.send("a", "那再确认一次", receiver="b", sender_role="general",
                   parent_id=d2.envelope.id)

    assert d1.delivered and d2.delivered
    assert d3.action == RouteAction.ESCALATE
    assert d3.code == RejectCode.MAX_HOPS
    assert d3.envelope.receiver == "supervisor"      # 换了个收口的人
    assert len(orch.escalations) == 1


# 【什么】互相踢皮球：双向各 2 次后升级
# 【作用】这不是"错"而是"卡住了"，必须有第三方收口
def test_pingpong_escalates():
    bus, orch = _orch(max_hops=10, pingpong_limit=2, repeat_limit=99)
    for i in range(2):
        assert orch.send("a", f"催一下 {i}", receiver="b", sender_role="general").delivered
        orch.send("b", f"等一下 {i}", receiver="a", sender_role="general")

    d = orch.send("a", "再催", receiver="b", sender_role="general")
    assert d.action == RouteAction.ESCALATE
    assert d.code == RejectCode.LOOP_DETECTED


# 【什么】相同内容重复发送被判定为无效交互
# 【作用】模型最容易出现的退化就是反复输出同一句话
def test_repeat_content_escalates():
    bus, orch = _orch(max_hops=10, repeat_limit=2)
    assert orch.send("a", "同一句话", receiver="b", sender_role="general").delivered
    assert orch.send("a", "同一句话", receiver="b", sender_role="general").delivered

    d = orch.send("a", "同一句话", receiver="b", sender_role="general")
    assert d.action == RouteAction.ESCALATE
    assert d.code == RejectCode.LOOP_DETECTED


# ---------------------------------------------------------------------------
# 5. 统一观测链路
# ---------------------------------------------------------------------------
# 【什么】span 树：子 span 挂在父 span 下，可导出、可统计
# 【作用】线上出问题按 trace_id 拉整棵树，哪一环慢哪一环错一眼可见
def test_tracer_builds_span_tree():
    tr = Tracer()
    trace = tr.new_trace("run", run_id="run_1")

    with tr.span(trace, "plan", kind="plan") as sp:
        sp.set_io(inputs={"query": "营收多少"}, outputs={"subtasks": 2})
        child = tr.start_span(trace, "agent.w1", kind="agent", parent_id=sp.span_id)
        tr.end_span(child, outputs={"success": True})

    tree = trace.to_tree()
    assert tree["name"] == "plan"
    assert tree["children"][0]["name"] == "agent.w1"
    assert trace.stats()["by_kind"]["plan"]["count"] == 1
    assert "agent.w1" in trace.summary()
    assert trace.to_dict()["span_count"] == 2


# 【什么】span 里抛异常：记 ERROR 后原样向上抛
# 【作用】观测层只负责记录，不负责兜底；吞掉异常会让上层降级永不触发
def test_tracer_marks_error_and_reraises():
    tr = Tracer()
    trace = tr.new_trace("run")
    with pytest.raises(ValueError):
        with tr.span(trace, "boom", kind="x"):
            raise ValueError("炸了")

    assert list(trace.spans.values())[-1].status.value == "error"


# 【什么】trace 结束时把未关闭的 span 收口为 TIMEOUT
# 【作用】运行被打断时，导出的树里不会永远挂着 running
def test_tracer_finish_closes_open_spans():
    tr = Tracer()
    trace = tr.new_trace("run")
    tr.start_span(trace, "hanging", kind="x")
    tr.finish(trace)

    assert all(sp.status.value != "running" for sp in trace.spans.values())


# ---------------------------------------------------------------------------
# 6. 接入 harness：真实跑一次，验证链路与编排
# ---------------------------------------------------------------------------
# 【什么】跑一次多 Agent，检查 trace / span / 消息统计
# 【作用】这是"线上能回溯完整链路"的最小证明
def test_harness_produces_trace_and_spans(fake_team, task_factory):
    inject = fake_team        # fixture 直接给的是注入器，不是工厂
    inject(
        plan=PlanResult(subtasks=[task_factory("t1", "w1")], needs_review=False),
        synth="团队汇总答案",
        workers={"w1": "ok"},
    )
    h = AgentHarness(config=_team_cfg(("w1", "w2")))
    r = h.run("营收是多少")

    # 【断言口径】trace_id 与 spans 都要有，否则线上无从回溯
    assert r.trace_id
    assert r.spans
    kinds = {sp["kind"] for sp in r.spans}
    assert {"run", "plan", "dispatch", "agent"} <= kinds

    # 共享发现走了编排器：总线里能查到一条已投递的 finding
    stats = h.orchestrator.stats()
    assert stats["delivered"] >= 1
    assert stats["escalated"] == 0


# 【什么】harness 的消息被敏感词策略打码
# 【作用】证明"自动发的系统消息"同样受编排器管控，不是旁路
def test_harness_messages_are_guarded(fake_team, task_factory):
    inject = fake_team        # fixture 直接给的是注入器，不是工厂
    inject(
        plan=PlanResult(subtasks=[task_factory("t1", "w1")], needs_review=False),
        synth="汇总",
        workers={"w1": lambda task: __import__(
            "modules.multiagent.models", fromlist=["AgentOutput"]
        ).AgentOutput(agent="w1", task_id=task.id, content="参考内部代号 A-7", steps=1)},
    )
    cfg = _team_cfg(("w1",), orchestration_sensitive_words="内部代号")
    h = AgentHarness(config=cfg)
    r = h.run("查一下")

    delivered = [m for m in h.bus.all(limit=50)
                 if m.delivery == DeliveryStatus.DELIVERED]
    assert delivered
    assert all("内部代号" not in m.content for m in delivered)


# 【什么】造团队配置的辅助函数
# 【作用】固定 1 主管 + N 执行者，其余参数透传给 TeamConfig
def _team_cfg(workers=("w1",), **cfg) -> TeamConfig:
    agents = [AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR)]
    agents += [AgentSpec(name=w, role=AgentRole.RESEARCHER) for w in workers]
    return TeamConfig(agents=agents, **cfg)
