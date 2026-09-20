"""
演示：同一个 MCPClient，接两个互不相识的 Server，Client 侧代码一行没改。

对比项：
  Server A = 项目自带的 modules/mcp/mcp_server.py（类 + protocol 模块 + loguru）
  Server B = examples/wild_mcp_server.py（纯标准库，平铺 for 循环，零项目依赖）

两者实现毫无共同点，但对 MCPClient 来说完全一样。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"

from modules.mcp.mcp_client import MCPClient  # noqa: E402

SERVERS = {
    "Server A (self-built)": [sys.executable, "-m", "modules.mcp.mcp_server"],
    "Server B (wild)": [sys.executable, str(ROOT / "examples" / "wild_mcp_server.py")],
}

for label, server_command in SERVERS.items():
    print(f"\n{'=' * 62}\n  {label}   <-  {' '.join(server_command[1:])}\n{'=' * 62}")

    client = MCPClient(server_command, name=label)
    print("connect()       ->", client.connect())

    names = [t["name"] for t in client.discover_tools()]
    print("discover_tools  ->", names)

    result = client.call_tool(
        "string_reverse" if "string_reverse" in names else "knowledge_stats",
        {"text": "Hello MCP"} if "string_reverse" in names else {},
    )
    print("call_tool       ->", result["content"][0]["text"][:70])

    client.disconnect()

print(f"\n{'=' * 62}")
print("Client code unchanged, yet it talks to two totally different servers.")
print(f"{'=' * 62}")
