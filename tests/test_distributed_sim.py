"""
本地模拟分布式多 Agent 测试
============================
覆盖：消息代理收发往返、四件套（超时降级 / 死信 / 冲突多数决 / 委派深度闸）、
真实进程隔离（多进程模式）。

【怎么跑】
    python -m pytest tests/test_distributed_sim.py -q

【为什么多数用线程模式】线程模式（use_processes=False）没有 spawn 开销，
毫秒级起节点；进程隔离单独一个测试验证。两种模式走同一套 _node_main 循环，
代理语义完全一致。
"""
from __future__ import annotations

import os

from modules.multiagent.distributed import DistributedCoordinator


def _make(nodes=("n1", "n2"), topic="tasks", **kw) -> DistributedCoordinator:
    """线程模式协调器 + 两个节点（默认 topic）。"""
    c = DistributedCoordinator(use_processes=False, **kw)
    for n in nodes:
        c.add_node(n, topic)
    return c


def test_pubsub_roundtrip():
    """基础往返：2 任务发布 → 路由到节点 → 执行 → 结果回传 → ack 销账。"""
    c = _make()
    try:
        out = c.run([
            {"task_id": "t1", "instruction": "查营收"},
            {"task_id": "t2", "instruction": "算环比"},
        ])
        assert out["answers"] == {"t1": "已执行: 查营收", "t2": "已执行: 算环比"}
        s = out["broker_stats"]
        assert s["published"] == 2 and s["acked"] == 2
        assert s["redelivered"] == 0 and s["dead_lettered"] == 0
        assert out["degraded"] == [] and out["conflicts"] == []
    finally:
        c.stop()


def test_slow_task_timeout_degrades():
    """四件套②③：slow 任务超时 → 进降级名单；正常任务不受影响。"""
    c = _make(task_timeout=1.0)
    try:
        out = c.run([
            {"task_id": "t1", "instruction": "快任务"},
            {"task_id": "t2", "instruction": "慢任务", "mode": "slow", "seconds": 3.0},
        ])
        assert out["answers"]["t1"] == "已执行: 快任务"
        assert "t2" not in out["answers"]
        assert [d["task_id"] for d in out["degraded"]] == ["t2"]
        assert "超时" in out["degraded"][0]["reason"]
    finally:
        c.stop()


def test_fail_message_goes_to_dlq():
    """代理语义：fail 消息 nack → 重投 → 超过重试上限进死信。"""
    c = _make(task_timeout=3.0, visibility_timeout=0.2, max_retries=2)
    try:
        out = c.run([{"task_id": "t1", "instruction": "毒消息", "mode": "fail"}])
        s = out["broker_stats"]
        assert s["redelivered"] >= 2            # 至少重投了 max_retries 次
        assert s["dead_lettered"] >= 1          # 最终进死信
        assert len(c.broker.dlq) >= 1
        assert [d["task_id"] for d in out["degraded"]] == ["t1"]   # 死信任务进降级名单
    finally:
        c.stop()


def test_conflict_majority_vote():
    """四件套④：冗余 3 份（种子 A/B/A）→ 多数决取 A，少数派留痕。"""
    c = _make(nodes=("n1", "n2", "n3"))
    try:
        out = c.run([{
            "task_id": "t1", "instruction": "估营收", "mode": "biased",
            "redundancy": 3, "seeds": ["A", "B", "A"],
        }])
        assert out["answers"]["t1"] == "已执行: 估营收 | 结论 A"
        assert len(out["conflicts"]) == 1
        conflict = out["conflicts"][0]
        assert conflict["rule"] == "多数决"
        assert sorted(conflict["variants"]) == sorted([
            "已执行: 估营收 | 结论 A", "已执行: 估营收 | 结论 B", "已执行: 估营收 | 结论 A",
        ])
    finally:
        c.stop()


def test_delegation_depth_gate():
    """四件套①：节点冒出委派请求 → 深度闸放行/拒绝。

    max_depth=2：第 1 层任务请求 → 子任务 depth=2 放行并拿到结果。
    max_depth=1：同样请求 → depth=2 > 1 → 拒绝并记录 denied。
    """
    c = _make(nodes=("n1",), task_timeout=3.0, max_depth=2)
    try:
        out = c.run([{"task_id": "t1", "instruction": "要委派", "mode": "delegate",
                      "delegate_topic": "tasks"}])
        assert out["denied"] == []
        assert "t1>d2" in out["answers"]                     # 子任务结果入账
        assert "承接委派的子任务" in out["answers"]["t1>d2"]
    finally:
        c.stop()

    c2 = _make(nodes=("n1",), task_timeout=2.0, max_depth=1)
    try:
        out2 = c2.run([{"task_id": "t1", "instruction": "要委派", "mode": "delegate",
                        "delegate_topic": "tasks"}])
        assert len(out2["denied"]) == 1
        assert out2["denied"][0]["to_depth"] == 2
        assert "深度上限" in out2["denied"][0]["reason"]
        assert "t1>d2" not in out2["answers"]                # 被拒 → 没有子任务结果
    finally:
        c2.stop()


def test_process_isolation():
    """进程模式：节点真的跑在独立进程里（pid 不同于主进程）。"""
    c = DistributedCoordinator(use_processes=True, task_timeout=8.0)
    try:
        c.add_node("p1", "tasks")
        c.add_node("p2", "tasks")
        out = c.run([
            {"task_id": "t1", "instruction": "进程A"},
            {"task_id": "t2", "instruction": "进程B"},
        ])
        assert out["answers"] == {"t1": "已执行: 进程A", "t2": "已执行: 进程B"}
        pids = {r["pid"] for r in out["results"]}
        assert os.getpid() not in pids                        # 节点不在主进程跑
        assert len(pids) >= 1
        assert out["broker_stats"]["acked"] == 2
    finally:
        c.stop()
