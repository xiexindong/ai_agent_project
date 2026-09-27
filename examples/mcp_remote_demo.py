"""
远程 MCP 工具演示（Agent 当 Client，工具跑在"另一台机器"上）
==============================================================
用子进程起一个常驻 MCP Server（modules/mcp/mcp_http_server.py）当"远端"，
Agent 侧用 MCPHttpClient 通过 JSON-RPC over HTTP 连它：发现工具 → 注册进
ToolRegistry → 像本地工具一样调用。

和 stdio 版（examples/mcp_multi_server_demo.py）的区别：
    stdio  本机拉子进程 —— 要的是进程隔离
    http   远端常驻服务 —— 要的是远程执行（本 demo 用本机子进程模拟那台机器）

运行：
    python examples/mcp_remote_demo.py
"""
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.mcp.gateway import TRANSPORT_HTTP, MCPGateway, MCPServerConfig   # noqa: E402
from modules.mcp.mcp_http_client import MCPHttpClient                        # noqa: E402
from modules.tools.calculator import CalculatorTool                          # noqa: E402
from modules.tools.tool_registry import ToolRegistry                         # noqa: E402

PORT = int(os.getenv("MCP_DEMO_PORT", "8801") or 8801)


def _wait_healthy(url: str, seconds: float = 15.0) -> bool:
    """轮询健康检查，等"远端"真的监听起来。"""
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url.replace("/mcp", "/health"), timeout=1.0) as resp:
                if 200 <= int(getattr(resp, "status", 200)) < 300:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)
    return False


def main():
    # ------------------------------------------------------------------
    # ① 起"远程机器"上的 MCP Server（真实部署时这一步在另一台机器上做）
    # ------------------------------------------------------------------
    env = dict(os.environ, MCP_SERVER_PORT=str(PORT))
    proc = subprocess.Popen([sys.executable, "-m", "modules.mcp.mcp_http_server"],
                            cwd=str(ROOT), env=env)
    url = f"http://127.0.0.1:{PORT}/mcp"
    if not _wait_healthy(url):
        proc.terminate()
        print("远程 MCP Server 启动失败（可先手动跑：python -m modules.mcp.mcp_http_server）")
        return
    print(f"① 远程 MCP Server 已就绪: {url}\n")

    try:
        # ------------------------------------------------------------------
        # ② 客户端握手 + 发现远程工具
        # ------------------------------------------------------------------
        client = MCPHttpClient(endpoint=url, name="remote-mcp")
        print(f"② 握手: {client.connect()} | server={client.server_info}")
        tools = client.discover_tools()
        print(f"   远程工具: {[t['name'] for t in tools]}\n")

        # ------------------------------------------------------------------
        # ③ 网关纳管远程 server，按需注册进 ToolRegistry（与本地工具同接口）
        # ------------------------------------------------------------------
        registry = ToolRegistry()
        registry.register(CalculatorTool())                 # 本地工具
        gw = MCPGateway(servers=[MCPServerConfig(name="remote-mcp", transport=TRANSPORT_HTTP,
                                                 endpoint=url)], catalog_cache="")
        gw.catalog(refresh=True)
        newly = gw.ensure_registered(registry, ["knowledge_stats", "get_current_time"])
        print(f"③ 按需注册远程工具: {newly}")
        print(f"   通讯录: {[t.name for t in registry.list_tools()]}\n")

        # ------------------------------------------------------------------
        # ④ 调用：对 ReAct 来说，远程工具与本地工具完全等价
        # ------------------------------------------------------------------
        print("④ 调用远程工具 knowledge_stats:")
        print(f"   {registry.get_tool('knowledge_stats').run({})}")
        print("   调用本地工具 calculator:")
        print(f"   {registry.get_tool('calculator').run({'expression': '3*7'})}\n")
        print(f"   网关健康: {gw.health()}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\n远程 MCP Server 已关闭")


if __name__ == "__main__":
    main()
