"""
远程 MCP 工具测试（HTTP 传输）
==============================
用项目自带的模拟远程 MCP Server（modules/mcp/mcp_http_server.py）
把「Agent 当 MCP Client → 调远程机器上的工具」整条链路跑一遍：
握手 / 发现工具 / 调用 / 适配器注册 / 网关纳管 / 连不上降级 / 令牌鉴权。

【为什么单独一个文件】
    要真起一个 HTTP 服务（uvicorn 线程），不能塞进纯单元测试；
    fastapi / uvicorn 没装就整文件跳过，不影响其余用例。
"""
import socket
import threading
import time

import pytest

from modules.mcp.gateway import TRANSPORT_HTTP, TRANSPORT_STDIO, MCPGateway, MCPServerConfig
from modules.mcp.mcp_http_client import MCPHttpClient
from modules.tools.tool_registry import ToolRegistry

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

import uvicorn                                       # noqa: E402  （必须在 importorskip 之后）
from fastapi.testclient import TestClient            # noqa: E402
from modules.mcp.mcp_client import MCPToolAdapter    # noqa: E402
from modules.mcp.mcp_http_server import app          # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def mcp_url():
    """后台线程起一个真实的模拟远程 MCP Server，yield 它的 /mcp 地址。"""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "模拟远程 MCP Server 没起来"
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    t.join(timeout=5)


# ------------------------------------------------------------------
# ① 服务端（模拟远程 MCP Server）
# ------------------------------------------------------------------
def test_mock_server_health_and_tools():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    names = [t["name"] for t in client.get("/tools").json()["tools"]]
    assert {"web_search", "get_current_time", "knowledge_stats"} <= set(names)


def test_mock_server_jsonrpc_roundtrip():
    """直接按 JSON-RPC 打一帧：initialize → 应回 serverInfo。"""
    client = TestClient(app)
    resp = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == 1
    assert "serverInfo" in body["result"]


# ------------------------------------------------------------------
# ② 客户端：连远程 Server
# ------------------------------------------------------------------
def test_http_client_connect_and_discover(mcp_url):
    client = MCPHttpClient(endpoint=mcp_url, name="remote")
    assert client.connect() is True
    assert client.is_connected is True
    tools = client.discover_tools()
    names = [t["name"] for t in tools]
    assert "knowledge_stats" in names
    # 远程工具的元数据（inputSchema）是协议带回来的，不是本地硬编码
    schema = next(t for t in tools if t["name"] == "web_search")["inputSchema"]
    assert schema["required"] == ["query"]


def test_http_client_call_tool(mcp_url):
    client = MCPHttpClient(endpoint=mcp_url, name="remote")
    client.connect()
    result = client.call_tool("knowledge_stats", {})
    text = result["content"][0]["text"]
    assert "knowledge_base" in text


def test_http_client_unknown_tool(mcp_url):
    """远程不存在的工具：客户端不炸，返回 None（由上层转成失败结果）。"""
    client = MCPHttpClient(endpoint=mcp_url, name="remote")
    client.connect()
    assert client.call_tool("no_such_tool", {}) is None


def test_http_client_down_raises():
    """远端没起：connect 抛异常（网关接住后按"该 server 不可用"降级，不是崩）。"""
    client = MCPHttpClient(endpoint="http://127.0.0.1:9/mcp", timeout_seconds=3.0)
    with pytest.raises(Exception):
        client.connect()


def test_http_client_requires_token(mcp_url, monkeypatch):
    """服务端设了 MCP_SERVER_TOKEN 后，不带令牌 401，带了对的才通。"""
    monkeypatch.setenv("MCP_SERVER_TOKEN", "dev-token")
    with pytest.raises(Exception):
        MCPHttpClient(endpoint=mcp_url, timeout_seconds=10.0).connect()
    ok = MCPHttpClient(endpoint=mcp_url, token="dev-token", timeout_seconds=10.0)
    assert ok.connect() is True


# ------------------------------------------------------------------
# ③ 适配器：远程工具变成 BaseTool，进 ToolRegistry
# ------------------------------------------------------------------
def test_adapter_runs_remote_tool(mcp_url):
    client = MCPHttpClient(endpoint=mcp_url, name="remote")
    client.connect()
    info = next(t for t in client.discover_tools() if t["name"] == "knowledge_stats")
    tool = MCPToolAdapter(client, info)             # 同一个适配器，换个 client 就变远程
    assert tool.tool_type == "mcp"
    result = tool.run({})
    assert result.success is True
    assert "knowledge_base" in result.result


def test_adapter_and_registry(mcp_url):
    client = MCPHttpClient(endpoint=mcp_url, name="remote")
    client.connect()
    gw_info = next(t for t in client.discover_tools() if t["name"] == "get_current_time")
    registry = ToolRegistry()
    adapter = MCPToolAdapter(client, gw_info)
    registry.register(adapter)
    assert registry.get_tool("get_current_time") is adapter
    assert adapter.to_openai_format()["function"]["name"] == "get_current_time"
    assert registry.get_tool("get_current_time").run({}).success is True


# ------------------------------------------------------------------
# ④ 网关：远程 server 与本地子进程 server 混管
# ------------------------------------------------------------------
def test_server_config_transport_infer():
    """老配置（只有 command）仍是 stdio；只给 endpoint 自动判成 http。"""
    assert MCPServerConfig.from_dict({"name": "a", "command": ["python"]}).transport == TRANSPORT_STDIO
    cfg = MCPServerConfig.from_dict({"name": "r", "endpoint": "http://10.0.0.7:8800/mcp"})
    assert cfg.transport == TRANSPORT_HTTP
    assert cfg.endpoint == "http://10.0.0.7:8800/mcp"


def test_gateway_http_catalog_and_call(mcp_url):
    gw = MCPGateway(servers=[MCPServerConfig(name="remote", transport=TRANSPORT_HTTP,
                                             endpoint=mcp_url)],
                    catalog_cache="")
    tools = gw.catalog(refresh=True)
    assert "knowledge_stats" in [t["name"] for t in tools]
    result = gw.call_tool("knowledge_stats", {})
    assert "knowledge_base" in result["content"][0]["text"]
    assert gw.health()["servers"]["remote"] is not None


def test_gateway_http_ensure_registered(mcp_url):
    gw = MCPGateway(servers=[MCPServerConfig(name="remote", transport=TRANSPORT_HTTP,
                                             endpoint=mcp_url)],
                    catalog_cache="")
    gw.catalog(refresh=True)
    registry = ToolRegistry()
    newly = gw.ensure_registered(registry, ["knowledge_stats"])
    assert newly == ["knowledge_stats"]
    tool = registry.get_tool("knowledge_stats")
    assert tool is not None and tool.run({}).success is True


def test_gateway_http_down_degrades():
    """远程 server 挂了：目录建不出来但不抛异常——局部故障局部降级。"""
    gw = MCPGateway(servers=[MCPServerConfig(name="dead", transport=TRANSPORT_HTTP,
                                             endpoint="http://127.0.0.1:9/mcp")],
                    connect_retry=1, catalog_cache="")
    assert gw.catalog(refresh=True) == []
    with pytest.raises(Exception):
        gw.call_tool("knowledge_stats", {})
