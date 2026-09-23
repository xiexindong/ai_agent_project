"""
Harness 运行时测试：编排 / 超时 / 步数 / 降级 / 评测
====================================================
这五项是「多 Agent 能不能上生产」的全部治理点，也是面试最容易被追问的地方：
QPS 之外，面试官真正想听的是「你靠什么防止它跑飞」。

全部用例不依赖 LLM、不依赖数据库：
Agent 由 conftest 的 fake_team 注入，supervisor 返回固定计划，worker 返回固定产出。
"""
import pytest

from modules.multiagent.harness import AgentHarness
from modules.multiagent.models import (
    AgentOutput, AgentRole, AgentSpec, EvalCase, PlanResult, RunStatus,
    SubTask, TeamConfig,
)


# 【什么】造团队配置的辅助函数
# 【作用】固定 1 个主管 + N 个执行者，其余参数由 **cfg 透传给 TeamConfig
def _team(workers=("w1",), **cfg) -> TeamConfig:
    """造一个「1 主管 + N 执行者」的团队配置。"""
    agents = [AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR)]
    agents += [AgentSpec(name=w, role=AgentRole.RESEARCHER) for w in workers]
    return TeamConfig(agents=agents, **cfg)


# 【什么】造带子任务计划的辅助函数
# 【作用】关掉 needs_review，让用例只断言编排结果不牵扯终审
def _plan(subtasks) -> PlanResult:
    return PlanResult(subtasks=subtasks, needs_review=False)


# ------------------------------------------------------------------
# 【1. DAG 分批】同层并行、跨层串行，这是并行度的来源
# ------------------------------------------------------------------
# 【什么】依赖分层的用例
# 【作用】验：t1 → (t2,t3) → t4 分成 3 批，第 2 批两个可并行，无环
def test_topo_batches_levels(task_factory):
    t1 = task_factory("t1", "w1")
    t2 = task_factory("t2", "w1", depends_on=["t1"])
    t3 = task_factory("t3", "w2", depends_on=["t1"])
    t4 = task_factory("t4", "w1", depends_on=["t2", "t3"])

    batches, cyclic = AgentHarness.topo_batches([t1, t2, t3, t4])

    # 输入：t1 → (t2,t3) → t4
    # 输出：3 批，第 2 批可并行 2 个
    assert [len(b) for b in batches] == [1, 2, 1]
    assert cyclic == []
    assert [t.id for t in batches[1]] == ["t2", "t3"]


# 【什么】批内按优先级排序的用例
# 【作用】验：同批里 priority=5 的 high 排在 priority=1 的 low 前面
def test_topo_batches_priority_first(task_factory):
    """同批内 priority 大的排前面（先跑重要的，超时被砍时先砍不重要的）。"""
    low = task_factory("low", "w1", priority=1)
    high = task_factory("high", "w2", priority=5)
    batches, _ = AgentHarness.topo_batches([low, high])
    assert [t.id for t in batches[0]] == ["high", "low"]


# 【什么】环形依赖的用例
# 【作用】验：a↔b 互相依赖时 batches 为空、两个 id 都进了 cyclic
def test_topo_batches_detects_cycle(task_factory):
    """环必须被剔除并上报，否则 while 永远排不完 —— 是卡死，不是报错。"""
    a = task_factory("a", "w1", depends_on=["b"])
    b = task_factory("b", "w2", depends_on=["a"])
    batches, cyclic = AgentHarness.topo_batches([a, b])
    assert batches == []
    assert sorted(cyclic) == ["a", "b"]


# 【什么】依赖了不存在 id 的用例
# 【作用】验：该任务同样被剔除进 cyclic，不会留在队列里无限等待
def test_topo_batches_unknown_dependency(task_factory):
    """依赖了不存在的 id：同样剔除，不能让它无限等待。"""
    t = task_factory("t1", "w1", depends_on=["not_exist"])
    batches, cyclic = AgentHarness.topo_batches([t])
    assert batches == [] and cyclic == ["t1"]


# ------------------------------------------------------------------
# 【2. 无主管直接失败】没有编排者就不能假装成功
# ------------------------------------------------------------------
# 【什么】没有主管的用例
# 【作用】验：状态为 FAILED、error 里点名 supervisor、trace 末尾是 error 事件
def test_run_without_supervisor_fails(fake_team):
    h = AgentHarness(config=_team())
    h.supervisor = None                      # 模拟团队没注册上主管
    r = h.run("问题")
    assert r.status == RunStatus.FAILED
    assert "supervisor" in r.error
    assert r.trace[-1].kind == "error"


# ------------------------------------------------------------------
# 【3. 直接作答】不值得开团队时不派任务，省一次 LLM 往返
# ------------------------------------------------------------------
# 【什么】直接答路径的用例
# 【作用】验：计划是 direct_answer 时状态 SUCCESS、一个子任务都不派
def test_run_direct_answer_skips_dispatch(fake_team):
    fake_team(plan=PlanResult(direct_answer="你好，请问需要什么帮助？"))
    h = AgentHarness(config=_team())
    r = h.run("你好")
    assert r.status == RunStatus.SUCCESS
    assert r.final_answer == "你好，请问需要什么帮助？"
    assert r.outputs == []                   # 一个任务都没派


# ------------------------------------------------------------------
# 【4. 降级】单个 Agent 失败只降级，不拖垮整次运行
# ------------------------------------------------------------------
# 【什么】单个 Agent 失败的用例
# 【作用】验：一成一败时状态是 PARTIAL 而非 FAILED，failed_agents 只列 w2
def test_partial_when_one_agent_fails(fake_team, task_factory):
    fake_team(
        plan=_plan([task_factory("t1", "w1"), task_factory("t2", "w2")]),
        workers={"w1": "ok", "w2": "fail"},
        synth="汇总：w1 有结果，w2 失败",
    )
    h = AgentHarness(config=_team(("w1", "w2")))
    r = h.run("对比两件事")
    assert r.status == RunStatus.PARTIAL     # 不是 FAILED：有可用产出
    assert r.ok is True
    assert r.failed_agents == ["w2"]
    assert "w2" in r.error


# 【什么】全部 Agent 失败的用例
# 【作用】验：状态 FAILED、ok 为 False，但 final_answer 仍有兜底话术不是空串
def test_failed_when_all_agents_fail(fake_team, task_factory):
    """全军覆没时必须明确失败 + 给兜底文案，不能返回空答案当成功。"""
    fake_team(
        plan=_plan([task_factory("t1", "w1")]),
        workers={"w1": "fail"},
    )
    h = AgentHarness(config=_team(("w1",)))
    r = h.run("查一下")
    assert r.status == RunStatus.FAILED
    assert r.final_answer                     # 有兜底话术，不是空串
    assert r.ok is False


# ------------------------------------------------------------------
# 【5. 超时】总 deadline 与单任务等待，两层都要有
# ------------------------------------------------------------------
# 【什么】总超时的用例
# 【作用】验：规划耗时吃掉预算后状态 TIMEOUT，未执行的 2 个任务仍被记账为失败
def test_global_timeout_marks_batch_unexecuted(fake_team, task_factory):
    """总超时后，还没跑的批次必须被标记「未执行」而不是悄悄跳过。
    不写这段的后果：用户以为任务跑完了，其实有一半根本没启动，且 trace 里看不出来。"""
    fake_team(
        plan=_plan([task_factory("t1", "w1"), task_factory("t2", "w2")]),
        plan_delay=0.05,                      # 让规划先耗掉时间
    )
    h = AgentHarness(config=_team(("w1", "w2"), timeout_seconds=0.01))
    r = h.run("慢问题")
    assert r.status == RunStatus.TIMEOUT
    assert len(r.outputs) == 2                # 未执行也要记账
    assert all(not o.success for o in r.outputs)
    assert all("总超时" in o.error for o in r.outputs)


# 【什么】单任务超时的用例
# 【作用】验：卡住的 w1 被判"任务执行超时"，同批 w2 照常成功，整体 PARTIAL
def test_single_task_timeout_does_not_kill_batch(fake_team, task_factory):
    """一个任务卡住只让它自己失败，同批其他任务照常返回。"""
    fake_team(
        plan=_plan([task_factory("t1", "w1"), task_factory("t2", "w2")]),
        workers={"w1": 1.2, "w2": "ok"},     # w1 要 1.2 秒，超过等待上限
        synth="w2 的结果",
    )
    h = AgentHarness(config=_team(("w1", "w2"), timeout_seconds=0.5))
    r = h.run("一个问题")
    failed = {o.agent: o.error for o in r.outputs if not o.success}
    assert failed.get("w1") == "任务执行超时"     # 卡住的被砍
    assert any(o.agent == "w2" and o.success for o in r.outputs)  # 同伴活着
    assert r.status == RunStatus.PARTIAL


# ------------------------------------------------------------------
# 【6. 步数预算】防多个 Agent 各自循环叠加成天价账单
# ------------------------------------------------------------------
# 【什么】步数预算耗尽的用例
# 【作用】验：第一批花掉 10 步后，第二批 t2 被显式记账为失败并留下 trace
def test_step_budget_stops_later_batches(fake_team, task_factory):
    t1 = task_factory("t1", "w1")
    t2 = task_factory("t2", "w2", depends_on=["t1"])   # 两批
    fake_team(
        plan=_plan([t1, t2]),
        workers={"w1": lambda task: AgentOutput(agent="w1", task_id=task.id,
                                                content="完成", steps=10)},
        synth="第一批的结论",
    )
    h = AgentHarness(config=_team(("w1", "w2"), max_total_steps=5))
    r = h.run("分两步的任务")
    t2_out = [o for o in r.outputs if o.task_id == "t2"]
    # 第二批不是"消失"，而是被显式记账为失败：没跑的任务必须留痕
    assert len(t2_out) == 1 and t2_out[0].success is False
    assert "步数预算耗尽" in t2_out[0].error
    assert any("步数预算耗尽" in e.message for e in r.trace)


# ------------------------------------------------------------------
# 【7. trace】排查全靠它：plan → dispatch → agent_end → done 必须齐全
# ------------------------------------------------------------------
# 【什么】trace 覆盖完整生命周期的用例
# 【作用】验：事件序列首为 plan、尾为 done，中间含 dispatch/agent_start/agent_end
def test_trace_events_cover_full_lifecycle(fake_team, task_factory):
    fake_team(plan=_plan([task_factory("t1", "w1")]), synth="答案")
    h = AgentHarness(config=_team(("w1",)))
    r = h.run("一个问题")
    kinds = [e.kind for e in r.trace]
    assert kinds[0] == "plan" and kinds[-1] == "done"
    assert "dispatch" in kinds and "agent_start" in kinds and "agent_end" in kinds


# 【什么】外部事件钩子的用例
# 【作用】验：钩子能收到事件，且钩子自己抛异常不影响主流程（仍 SUCCESS）
def test_on_event_callback_receives_events(fake_team, task_factory):
    """外部钩子（SSE / 审计）能收到事件，且钩子自己抛异常不能影响主流程。"""
    seen = []
    fake_team(plan=_plan([task_factory("t1", "w1")]), synth="答案")

    # 【什么】故意抛异常的钩子
    # 【作用】验：钩子内部报错被吞掉，不打断 harness 主流程
    def bad_hook(event):
        seen.append(event.kind)
        raise RuntimeError("钩子炸了")

    h = AgentHarness(config=_team(("w1",)), on_event=bad_hook)
    r = h.run("一个问题")
    assert seen and r.status == RunStatus.SUCCESS   # 钩子异常被吞，主流程不受影响


# ------------------------------------------------------------------
# 【8. 评测】关键词 + 完成率 + 速度，能进 CI 的确定性指标
# ------------------------------------------------------------------
# 【什么】关键词命中的评测用例
# 【作用】验：答案含"营收"与"12%"时该条通过，得分大于 0.6
def test_evaluate_scores_keyword_hit(fake_team, task_factory):
    fake_team(
        plan=_plan([task_factory("t1", "w1")]),
        synth="营收增长了 12%",
    )
    h = AgentHarness(config=_team(("w1",)))
    report = h.evaluate([EvalCase(name="营收", query="营收如何",
                                  expected_keywords=["营收", "12%"],
                                  max_duration_ms=20000)])
    assert report.total == 1 and report.passed == 1
    assert report.results[0].score > 0.6


# 【什么】require_all_agents 的评测用例
# 【作用】验：有 Agent 失败时直接 0 分、passed 为 0，reason 里点名失败的 w2
def test_evaluate_zero_on_required_agent_failure(fake_team, task_factory):
    """require_all_agents 的用例：任一 Agent 失败直接 0 分（关键路径不允许缺人）。"""
    fake_team(
        plan=_plan([task_factory("t1", "w1"), task_factory("t2", "w2")]),
        workers={"w1": "ok", "w2": "fail"},
        synth="部分结论",
    )
    h = AgentHarness(config=_team(("w1", "w2")))
    report = h.evaluate([EvalCase(name="全量", query="两件事",
                                  expected_keywords=["结论"],
                                  require_all_agents=True)])
    assert report.passed == 0
    assert report.results[0].score == 0.0
    assert "w2" in report.results[0].reason


# 【什么】空用例集的评测用例
# 【作用】验：传空列表不报错，score 为 0.0、pass_rate 渲染成 "0/0"
def test_evaluate_empty_cases_is_safe():
    h = AgentHarness(config=_team())
    report = h.evaluate([])
    assert report.score == 0.0 and report.pass_rate == "0/0"
