"""
多 MCP Server 实战验证（真子进程 + 假数据）
=============================================
回答一个问题：多 server 纳管到底是"没实现"还是"没配置"？

答案是后者：机制在 modules/mcp/gateway.py 里，但 .env 没配 MCP_SERVERS_JSON，
所以运行时只登记了 1 个 enterprise-mcp。本脚本用 examples/mock_mcp_servers/
下的假部门 server 补齐配置，让下面这几件事全部真实发生：

    1. 一个网关同时纳管 5 个 server（4 个假部门 + 1 个故意坏掉的）
    2. 拉目录时真起 5 个子进程，工具按 server 归属合并进同一份目录
    3. 坏 server 只贡献 0 个工具并被熔断，其余 4 个照常可用
    4. 按需加载：本轮只召回 1 个工具时，只为它所属的那个 server 再起 1 次进程

【运行】
    python examples/mcp_multi_server_demo.py

【和 examples/mcp_skill_gateway_demo.py 的区别】
    那个是"假数据 + 不起进程"（把目录写进缓存文件假装拉过）；
    这个 是"真起子进程 + 真 JSON-RPC 通信"，只有工具的业务数据是假的。
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.common.circuit_breaker import CircuitBreaker
from modules.common.logger import get_logger
from modules.mcp.gateway import MCPGateway, MCPServerConfig
from modules.tools.tool_rag import ToolRAGIndex
from modules.tools.tool_registry import ToolRegistry

_logger = get_logger("multi_server_demo")

# 假 server 脚本：同一份代码，靠 --preset 变成不同部门
MOCK = str(Path(__file__).resolve().parent / "mock_mcp_servers" / "mock_mcp_server.py")
PRESETS = ["finance", "hr", "it", "data"]


# ----------------------------------------------------------------------
# 【装配多个 server】等价于在 .env 里配 MCP_SERVERS_JSON 的效果
#   command 用 sys.executable（当前解释器的绝对路径）：
#   写死 "python" 时，若子进程落到另一个没有项目依赖的 Python 上就会 import 失败。
# ----------------------------------------------------------------------
def build_servers() -> list:
    servers = [
        MCPServerConfig(name=f"{p}-mcp", command=[sys.executable, MOCK, "--preset", p])
        for p in PRESETS
    ]
    # 故意加一个起不来的：验证「局部故障局部降级」——它挂掉不影响其余 4 个
    servers.append(
        MCPServerConfig(name="broken-mcp", command=[sys.executable, "-m", "modules.mcp.not_exist_xyz"])
    )
    return servers


def main():
    print("####################################################################")
    print("#  多 MCP Server 纳管：真子进程 · 假业务数据                        #")
    print("####################################################################")

    tmpdir = Path(tempfile.mkdtemp(prefix="mcp_multi_"))
    gw = MCPGateway(
        servers=build_servers(),
        lazy_connect=True,
        catalog_cache=str(tmpdir / "multi_catalog.json"),   # 用临时文件，不污染 data/mcp_catalog.json
        connect_retry=2,
        breaker=CircuitBreaker(3, 30),
    )

    # ------------------------------------------------------------------
    # 第一幕：拉目录 —— 每个 server 各起一次子进程，工具按 server 归属合并
    # ------------------------------------------------------------------
    print("\n【第一幕】拉目录：5 个 server 各起一次子进程")
    print("=" * 66)
    catalog = gw.catalog(refresh=True)
    by_server = {}
    for t in catalog:
        by_server.setdefault(t["server"], []).append(t["name"])
    print(f"登记 server        {len(gw.server_names)} 个: {gw.server_names}")
    print(f"目录工具总数       {len(catalog)} 个")
    for s in gw.server_names:
        names = by_server.get(s, [])
        print(f"  - {s:<12} 贡献 {len(names)} 个: {names}")
    print("\n  broken-mcp 贡献 0 个，但其余 4 个照常 —— 这就是「局部故障局部降级」。")

    # ------------------------------------------------------------------
    # 第二幕：语义召回 —— 目录来自多个 server，召回时按语义跨 server 排序
    # ------------------------------------------------------------------
    print("\n【第二幕】Tool-RAG 召回：跨 4 个 server 的 12 个工具里挑本轮用得上的")
    print("=" * 66)
    idx = ToolRAGIndex(top_k=3, min_score=0.20)
    idx.index_tools(catalog)
    print(f"索引状态           {idx.stats()}")
    queries = ["我的报销单到哪一步了", "上海今天天气怎么样", "服务器负载高帮我建个工单"]
    picked = []
    for q in queries:
        hits = idx.recall(q)
        detail = ", ".join(f"{h.name}={h.score}" for h in hits) or "（无达标结果）"
        print(f"\n  提问: {q}")
        print(f"  召回: {detail}")
        for h in hits[:1]:
            picked.append((q, h.name))

    # ------------------------------------------------------------------
    # 第三幕：按需加载 —— 此刻才为被召回的工具所属 server 重新起进程
    # ------------------------------------------------------------------
    print("\n【第三幕】按需加载：只有被召回的工具所属 server 才再起一次进程")
    print("=" * 66)
    registry = ToolRegistry()
    registry.register_batch(registry.get_default_tools())
    names = [n for _, n in picked]
    newly = gw.ensure_registered(registry, names)
    print(f"请求注册           {names}")
    print(f"实际新注册         {newly}")
    for q, n in picked:
        entry_server = next((t["server"] for t in catalog if t["name"] == n), "?")
        args = {}
        if "reimbursement" in n:
            args = {"bill_id": "BX-2026-0918"}
        elif "weather" in n:
            args = {"city": "上海"}
        elif "health" in n:
            args = {"host": "web-01"}
        elif "ticket" in n:
            args = {"title": "服务器负载过高", "priority": "高"}
        elif "leave" in n or "attendance" in n:
            args = {"user_code": "E1002"}
        res = registry.execute_tool(n, args, session_id="multi-demo")
        print(f"\n  提问: {q}")
        print(f"  执行: {n}（来自 {entry_server}）")
        print(f"  结果: {str(res.result)[:90]}")

    # ------------------------------------------------------------------
    # 第四幕：健康快照 —— connects 说明到底起了几次进程
    # ------------------------------------------------------------------
    print("\n【第四幕】健康快照")
    print("=" * 66)
    health = gw.health()
    print(f"  {health}")
    print("\n  connects 的读法：第一次是拉目录（每个 server 各一次），")
    print("  之后只为「本轮真正用到的工具所属 server」再起进程。")

    gw.shutdown()

    # ------------------------------------------------------------------
    # 附：想让 main.py / API 服务也纳管这些 server，把这段粘进 .env
    # ------------------------------------------------------------------
    print("\n【附】想让真实链路（chat / serve）也纳管这 4 个，把下面这行加到 .env：")
    print("=" * 66)
    env_servers = [
        {"name": f"{p}-mcp", "command": [sys.executable, MOCK, "--preset", p]}
        for p in PRESETS
    ]
    print("MCP_SERVERS_JSON=" + json.dumps(env_servers, ensure_ascii=False))
    print("\n  加完重启，日志里会是：MCP 工具目录已建立: 工具数=12 | servers=[...]")
    print(f"  临时目录（可删）: {tmpdir}")


if __name__ == "__main__":
    main()
