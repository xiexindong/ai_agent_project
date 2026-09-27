"""
Human-in-the-Loop 单元测试
==========================
全部离线：不连数据库、不连网络、不调真实模型。

【为什么每个用例都用 auto_resolver 或短超时】
    真等人意味着测试要挂起几十秒。注入自动答复器后，
    被测的是"策略判定 → 请求登记 → 答复生效"这条逻辑本身，
    而定时器只在一个用例里用 0.3 秒验证超时分支。
"""
import os

# 【为什么在 import 项目模块前压日志】logger 首次 get_logger() 时才装配，
#   默认 INFO 会打印大量执行日志，把 pytest 结果淹没。
os.environ.setdefault("LOG_LEVEL", "ERROR")

import pytest  # noqa: E402

from modules.hitl import (  # noqa: E402
    HumanAction,
    HumanReply,
    InterruptKind,
    auto_resolver,
    build_gate,
    scripted_resolver,
)
from modules.hitl.models import InterruptRequest, InterruptStatus  # noqa: E402
from modules.hitl.policy import HITLPolicy  # noqa: E402
from modules.hitl.store import InMemoryInterruptStore  # noqa: E402


def test_未启用时所有中断自动放行():
    """策略关闭时 ask 立即返回 approve，且不产生任何请求。"""
    gate = build_gate(enabled=False)
    reply = gate.ask(kind=InterruptKind.BEFORE_TOOL, title="调用 xx")
    assert reply.action == HumanAction.APPROVE
    assert reply.request_id == ""          # 没有登记请求
    assert gate.pending() == []


def test_敏感工具被拦下_非敏感工具不打断():
    """工具名匹配到才打断，没匹配到零开销放行。"""
    gate = build_gate(enabled=True, sensitive_tools=["data_query", "mysql_*"],
                      resolver=auto_resolver(HumanAction.REJECT))
    go, args = gate.ask_tool("data_query", {"id": "1"})
    assert go is False
    assert args == {"id": "1"}             # 拒绝时参数原样返回
    go2, _ = gate.ask_tool("calculator", {"expr": "1+1"})
    assert go2 is True
    # 前缀规则也要生效
    go3, _ = gate.ask_tool("mysql_execute", {"sql": "delete"})
    assert go3 is False


def test_人工改参数后按新参数执行():
    """modify 带 edited（JSON 字符串）时替换参数；非 JSON 则沿用原参数。"""
    new_args = {"table": "orders", "limit": 10}
    gate = build_gate(
        enabled=True, sensitive_tools=["data_query"],
        resolver=auto_resolver(HumanAction.MODIFY, edited='{"table": "orders", "limit": 10}'),
    )
    go, args = gate.ask_tool("data_query", {"table": "users", "limit": 1000})
    assert go is True
    assert args == new_args

    bad = build_gate(
        enabled=True, sensitive_tools=["data_query"],
        resolver=auto_resolver(HumanAction.MODIFY, edited="这不是JSON"),
    )
    go2, args2 = bad.ask_tool("data_query", {"table": "users"})
    assert go2 is True
    assert args2 == {"table": "users"}     # 格式不对就沿用原参数，不搞挂调用


def test_超时按默认动作兜底而不是挂死():
    """无人响应时：工具类默认 reject、终稿类默认 approve。"""
    gate = build_gate(enabled=True, sensitive_tools=["data_query"], timeout_seconds=0.3)
    go, _ = gate.ask_tool("data_query", {"id": "1"})
    assert go is False                      # tool_action 默认 reject

    go2, text = gate.ask_final("草稿内容")
    assert go2 is True                      # final_action 默认 approve
    assert text == "草稿内容"

    records = gate.records()
    assert len(records) == 2
    assert all(r.request.status == InterruptStatus.EXPIRED for r in records)
    # 兜底动作要能在归档里查到，否则审计只剩"超时"不知道最终怎么执行的
    assert records[0].final_action == HumanAction.REJECT


def test_按中断点编排不同答复():
    """scripted_resolver 能对不同 kind 给不同动作，未配置的走 fallback。"""
    gate = build_gate(
        enabled=True,
        resolver=scripted_resolver({"before_task": HumanAction.REJECT},
                                   fallback=HumanAction.APPROVE),
    )
    r1 = gate.ask(kind=InterruptKind.BEFORE_TASK, title="执行 t1")
    r2 = gate.ask(kind=InterruptKind.BEFORE_FINAL, title="确认终稿")
    assert r1.action == HumanAction.REJECT
    assert r2.action == HumanAction.APPROVE


def test_重复提交被拒绝():
    """同一次请求提交两遍，第二遍不生效（网络重试场景）。"""
    store = InMemoryInterruptStore()
    gate = build_gate(enabled=True, sensitive_tools=["data_query"])
    gate.store = store
    req = store.save(InterruptRequest(
        kind=InterruptKind.BEFORE_TOOL, title="调用 x", default_action=HumanAction.APPROVE
    ))
    assert store.submit(HumanReply(request_id=req.id, action=HumanAction.REJECT)) is True
    assert store.submit(HumanReply(request_id=req.id, action=HumanAction.APPROVE)) is False
    assert store.get(req.id).status == InterruptStatus.RESOLVED


def test_pending_只返回未处理的():
    """超时结束的请求不该再出现在审批待办里，否则审批页会一直挂着旧卡片。"""
    gate = build_gate(enabled=True, timeout_seconds=0.2)
    gate.ask(kind=InterruptKind.BEFORE_PLAN, title="确认计划")
    assert gate.pending() == []


def test_退出时作废在等的请求():
    """cancel_waiting 把还在等的请求标记作废并唤醒等待方。"""
    gate = build_gate(enabled=True, timeout_seconds=30)
    # 起一个线程发起中断，主线程随后作废它
    import threading
    holder = {}

    def _ask():
        holder["reply"] = gate.ask(kind=InterruptKind.ON_DISPUTE, title="等裁决")

    t = threading.Thread(target=_ask, daemon=True)
    t.start()
    import time
    time.sleep(0.2)
    n = gate.cancel_waiting()
    t.join(timeout=2)
    assert n >= 1
    assert holder.get("reply") is not None
    assert holder["reply"].action == HumanAction.APPROVE   # ON_DISPUTE 默认 approve


def test_策略的中断点白名单():
    """kinds 留空=全启用；填了只认填的。"""
    p_all = HITLPolicy(enabled=True)
    assert p_all.kind_enabled(InterruptKind.BEFORE_TOOL) is True

    p_some = HITLPolicy(enabled=True, kinds=["before_final"])
    assert p_some.kind_enabled(InterruptKind.BEFORE_FINAL) is True
    assert p_some.kind_enabled(InterruptKind.BEFORE_TOOL) is False

    p_off = HITLPolicy(enabled=False)
    assert p_off.kind_enabled(InterruptKind.BEFORE_FINAL) is False


def test_默认动作按中断点分别配置():
    """同一条策略下，工具默认拒绝、终稿默认放行。"""
    p = HITLPolicy(enabled=True)
    assert p.default_action_for(InterruptKind.BEFORE_TOOL) == HumanAction.REJECT
    assert p.default_action_for(InterruptKind.BEFORE_FINAL) == HumanAction.APPROVE


def test_content_or_取最终内容():
    """modify 且有编辑内容时用编辑后的，否则用原文。"""
    r1 = HumanReply(request_id="x", action=HumanAction.MODIFY, edited="改后")
    r2 = HumanReply(request_id="x", action=HumanAction.APPROVE)
    assert r1.content_or("原文") == "改后"
    assert r2.content_or("原文") == "原文"
