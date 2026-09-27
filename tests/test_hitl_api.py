"""
HITL 审批端点单元测试
======================
不依赖数据库与网络：用 TestClient 挂一个只有 HITL 路由的干净应用。

【为什么能这么测】
    hitl_routes 的 register 函数把 gate 和鉴权依赖都当参数传入，
    测试里传一个假鉴权（直接返回固定用户）和一个带自动答复器的 gate，
    就能验证"待办列表 / 提交决策 / 留痕"三件事，而不牵扯真实鉴权与存储。
"""
import os

os.environ.setdefault("LOG_LEVEL", "ERROR")

import pytest  # noqa: E402

fastapi = pytest.importorskip("fastapi")   # 没装 fastapi 就整组跳过，符合"可选依赖"约定
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from modules.api.hitl_routes import register_hitl_routes  # noqa: E402
from modules.hitl import HumanAction, InterruptKind, InterruptNotifier, build_gate  # noqa: E402
from modules.hitl.models import InterruptRequest  # noqa: E402


def _fake_user() -> dict:
    """假鉴权依赖：直接返回一个固定的用户字典。"""
    return {"username": "alice", "user_id": "u1"}


def _make_client(gate, notifier=None) -> TestClient:
    app = FastAPI()
    register_hitl_routes(app, gate, _fake_user, notifier)
    return TestClient(app)


def _sse_frame(event: str, data: dict) -> bytes:
    """直接调端点的组帧函数，避开对长连接做端到端读流（会把测试挂死）。"""
    from modules.api.hitl_routes import _sse
    return _sse(event, data)


def test_待办列表能列出等待中的请求():
    gate = build_gate(enabled=True, timeout_seconds=30)
    # 起一个线程发起中断（会阻塞等待），主线程看待办
    import threading
    t = threading.Thread(
        target=lambda: gate.ask(kind=InterruptKind.BEFORE_TOOL, title="调用 mysql_execute",
                                reason="敏感工具"),
        daemon=True,
    )
    t.start()
    client = _make_client(gate)
    for _ in range(50):                 # 等线程把请求登记进来
        r = client.get("/hitl/pending")
        if r.json()["count"] > 0:
            break
        import time
        time.sleep(0.05)
    body = r.json()
    assert body["count"] == 1
    item = body["items"][0]
    assert item["title"] == "调用 mysql_execute"
    assert item["default_action"] == "reject"      # 工具类默认拒绝
    assert "approve" in item["options"]
    gate.cancel_waiting()
    t.join(timeout=2)


def test_提交决策后请求被关闭():
    gate = build_gate(enabled=True, timeout_seconds=30)
    req = gate.store.save(InterruptRequest(
        kind=InterruptKind.BEFORE_FINAL, title="确认终稿", default_action=HumanAction.APPROVE,
    ))
    client = _make_client(gate)
    r = client.post("/hitl/reply", json={
        "request_id": req.id, "action": "reject", "note": "措辞要改",
    })
    assert r.status_code == 200
    assert r.json()["action"] == "reject"
    # 已处理的不该再出现在待办里
    assert client.get("/hitl/pending").json()["count"] == 0


def test_重复提交返回409():
    gate = build_gate(enabled=True)
    req = gate.store.save(InterruptRequest(
        kind=InterruptKind.ON_DISPUTE, title="裁决冲突", default_action=HumanAction.APPROVE,
    ))
    client = _make_client(gate)
    assert client.post("/hitl/reply", json={"request_id": req.id, "action": "approve"}).status_code == 200
    second = client.post("/hitl/reply", json={"request_id": req.id, "action": "reject"})
    assert second.status_code == 409


def test_非法action返回422():
    gate = build_gate(enabled=True)
    req = gate.store.save(InterruptRequest(
        kind=InterruptKind.BEFORE_PLAN, title="确认计划", default_action=HumanAction.APPROVE,
    ))
    client = _make_client(gate)
    r = client.post("/hitl/reply", json={"request_id": req.id, "action": "maybe"})
    assert r.status_code == 422


def test_留痕记录最终动作与操作人():
    gate = build_gate(enabled=True, timeout_seconds=0.2)
    gate.ask(kind=InterruptKind.BEFORE_TOOL, title="调用 data_query")   # 超时兜底
    client = _make_client(gate)
    body = client.get("/hitl/records").json()
    assert body["count"] == 1
    item = body["items"][0]
    assert item["final_action"] == "reject"          # 工具类的超时兜底
    assert item["operator"] == "system:timeout"      # 没真人处理，标成系统兜底
    assert item["status"] == "expired"


def test_未启用时端点返回503():
    app = FastAPI()
    register_hitl_routes(app, None, _fake_user)
    client = TestClient(app)
    assert client.get("/hitl/pending").status_code == 503
    assert client.post("/hitl/reply", json={"request_id": "x", "action": "approve"}).status_code == 503


def test_SSE端点已注册且组帧正确():
    """/hitl/stream 挂上了，且帧是标准 SSE 格式、中文不转义。"""
    gate = build_gate(enabled=True, timeout_seconds=0.2)
    notifier = InterruptNotifier(console=False)
    app = FastAPI()
    register_hitl_routes(app, gate, _fake_user, notifier)

    assert "/hitl/stream" in [getattr(r, "path", "") for r in app.routes]

    frame = _sse_frame("interrupt", {"title": "调用 send_notice"})
    assert frame.startswith(b"event: interrupt\ndata: ")
    assert frame.endswith(b"\n\n")
    assert "调用 send_notice".encode("utf-8") in frame        # ensure_ascii=False


def test_订阅与退订在连接生命周期内配平():
    """SSE 靠订阅队列收事件；连上加、断开摘，队列不泄漏。"""
    notifier = InterruptNotifier(console=False)
    q = notifier.subscribe()
    notifier.notify(InterruptRequest(kind=InterruptKind.BEFORE_TOOL, title="调用 data_query"))
    assert q.get_nowait().title == "调用 data_query"

    notifier.unsubscribe(q)
    assert notifier.subscriber_count == 0
