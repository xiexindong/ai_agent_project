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

ROOT = Path(__file__).resolve().parent.parent       # examples/ 的上一层 = 项目根
sys.path.insert(0, str(ROOT))                       # 否则 import modules.xxx 失败

os.environ["PYTHONIOENCODING"] = "utf-8"            # 子进程 stdout 强制 UTF-8，防中文乱码

from modules.mcp.mcp_client import MCPClient  # noqa: E402

# 【什么】两个实现风格完全不同的 MCP Server
# 【作用】值统一是启动命令数组，MCPClient 拿到就能起进程，不关心对方怎么写
SERVERS = {
    "Server A (self-built)": [sys.executable, "-m", "modules.mcp.mcp_server"],                 # 项目自研：类 + protocol 模块
    "Server B (wild)": [sys.executable, str(ROOT / "examples" / "wild_mcp_server.py")],        # 野生：纯标准库平铺
}

# 【什么】逐个 server 跑同一段 Client 代码
# 【作用】打印 connect / discover_tools / call_tool 三步结果，证明客户端无需改动
for label, server_command in SERVERS.items():
    print(f"\n{'=' * 62}\n  {label}   <-  {' '.join(server_command[1:])}\n{'=' * 62}")

    client = MCPClient(server_command, name=label)
    print("connect()       ->", client.connect())

    # 【什么】拉工具清单并只取名字
    # 【作用】两个 server 暴露的工具不同，名字打印出来就能看出差异
    names = [t["name"] for t in client.discover_tools()]
    print("discover_tools  ->", names)

    # 【什么】按名字挑一个工具来调
    # 【作用】有 string_reverse 就传 text 参数，否则调无参的 knowledge_stats
    result = client.call_tool(
        "string_reverse" if "string_reverse" in names else "knowledge_stats",
        {"text": "Hello MCP"} if "string_reverse" in names else {},
    )
    print("call_tool       ->", result["content"][0]["text"][:70])

    client.disconnect()

print(f"\n{'=' * 62}")
print("Client code unchanged, yet it talks to two totally different servers.")
print(f"{'=' * 62}")
