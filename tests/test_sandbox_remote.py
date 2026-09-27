"""
远程沙箱测试
============
用项目自带的模拟远程服务（modules/harness/remote_sandbox_server.py）
把 RemoteSandboxRunner 整条链路跑一遍：正常执行 / 工具报错 / 服务不可用降级 / 鉴权。

【为什么单独一个文件】
    要真起一个 HTTP 服务（uvicorn 线程），不能塞进纯单元测试里；
    fastapi / uvicorn 没装就整文件跳过，不影响其余用例。
"""
import socket
import threading
import time

import pytest

from modules.harness.sandbox import MODE_REMOTE, RemoteSandboxRunner, build_sandbox
from modules.tools.calculator import CalculatorTool

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

import uvicorn                                        # noqa: E402  （必须在 importorskip 之后）
from fastapi.testclient import TestClient             # noqa: E402
from modules.harness.remote_sandbox_server import app  # noqa: E402


# 【什么】找一个空闲端口
# 【作用】测试起服务不能写死端口，否则并发跑测试会撞车
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def remote_url():
    """在后台线程起一个真实的模拟远程沙箱服务，yield 它的 /run 地址。"""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(200):                              # 等服务真的监听起来（最多 10s）
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "模拟远程沙箱服务没起来"
    yield f"http://127.0.0.1:{port}/run"
    server.should_exit = True
    t.join(timeout=5)


# ------------------------------------------------------------------
# ① 服务本身（模拟远程沙箱）
# ------------------------------------------------------------------
def test_mock_server_runs_tool():
    """模拟服务直接收类坐标：算 3*7 得 21。"""
    client = TestClient(app)
    resp = client.post("/run", json={
        "module": "modules.tools.calculator", "class": "CalculatorTool",
        "params": {"expression": "3*7"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert float(data["result"]) == 21


def test_mock_server_rejects_bad_payload():
    """缺 class 字段 → 400（这是服务层的错误，不是工具执行失败）。"""
    client = TestClient(app)
    resp = client.post("/run", json={"module": "modules.tools.calculator"})
    assert resp.status_code == 400


def test_mock_server_requires_token_when_configured(monkeypatch):
    """服务端设了 REMOTE_SANDBOX_TOKEN 后，不带令牌的请求 401。"""
    client = TestClient(app)
    monkeypatch.setenv("REMOTE_SANDBOX_TOKEN", "dev-token")
    assert client.post("/run", json={
        "module": "modules.tools.calculator", "class": "CalculatorTool",
        "params": {"expression": "1+1"},
    }).status_code == 401
    assert client.post(
        "/run", json={"module": "modules.tools.calculator", "class": "CalculatorTool",
                      "params": {"expression": "1+1"}},
        headers={"X-Sandbox-Token": "dev-token"},
    ).status_code == 200


# ------------------------------------------------------------------
# ② 远程执行器
# ------------------------------------------------------------------
def test_build_sandbox_remote():
    """按模式名构造：remote → RemoteSandboxRunner。"""
    runner = build_sandbox(MODE_REMOTE)
    assert isinstance(runner, RemoteSandboxRunner)
    assert runner.mode == MODE_REMOTE


def test_remote_run_ok(remote_url):
    """正常路径：工具在"远端"跑，结果原样回传。"""
    runner = RemoteSandboxRunner(endpoint=remote_url, timeout_seconds=30.0)
    assert runner.available() is True                 # 健康检查通
    result = runner.run(CalculatorTool(), {"expression": "3*7"})
    assert result.success is True
    assert float(result.result) == 21


def test_remote_tool_error_not_degraded(remote_url):
    """工具在远端跑失败：返回失败结果，不算"服务不可用"，不降级。"""
    runner = RemoteSandboxRunner(endpoint=remote_url, timeout_seconds=30.0)
    result = runner.run(CalculatorTool(), {"expression": "__import__('os')"})
    assert result.success is False
    assert result.error                                # 错误信息要能回传


def test_remote_down_degrades_to_subprocess():
    """远端连不上 → 自动降级本机子进程，工具照样跑成功（沙箱不该成为可用性瓶颈）。"""
    runner = RemoteSandboxRunner(endpoint="http://127.0.0.1:9/run", timeout_seconds=30.0)
    assert runner.available() is False
    result = runner.run(CalculatorTool(), {"expression": "21*2"})
    assert result.success is True
    assert float(result.result) == 42


def test_remote_down_without_fallback_returns_error():
    """关掉降级后，连不上就直接失败（需要强隔离时用：宁可不跑，也不能在本机跑）。"""
    runner = RemoteSandboxRunner(endpoint="http://127.0.0.1:9/run", timeout_seconds=5.0,
                                 fallback=False)
    result = runner.run(CalculatorTool(), {"expression": "1+1"})
    assert result.success is False
    assert "远程沙箱" in result.error


def test_remote_wrong_token(remote_url, monkeypatch):
    """令牌不对 → 401 → 不允许降级时直接失败。"""
    monkeypatch.setenv("REMOTE_SANDBOX_TOKEN", "dev-token")
    bad = RemoteSandboxRunner(endpoint=remote_url, token="wrong", timeout_seconds=30.0,
                              fallback=False)
    assert bad.run(CalculatorTool(), {"expression": "1+1"}).success is False

    good = RemoteSandboxRunner(endpoint=remote_url, token="dev-token", timeout_seconds=30.0)
    assert good.run(CalculatorTool(), {"expression": "1+1"}).success is True
