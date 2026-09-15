"""
一个"野生"的 MCP Server —— 故意不用项目里任何一行代码。

【这个文件的意义】
modules/mcp/mcp_server.py 是"项目自研的 Server"：用类、用 get_logger、用 modules.mcp.protocol。
而这个文件完全相反：
    - 不 import 项目里任何一个模块，只用 Python 标准库（sys / json）
    - 没有 class、没有 _define_tools、没有 handle_request
    - 实现风格跟官方那个 Server 完全不同（就是个平铺的 for 循环）

但只要它吐出来的 JSON 符合 JSON-RPC 2.0，标准的 MCPClient 就能原样接住。
这就是"分两端"的核心收益：Client 不关心 Server 怎么写的，只认报文。
"""
import sys
import json

# 这个 Server 只提供 1 个工具，跟项目自带的 3 个完全不同
TOOLS = [{
    "name": "string_reverse",
    "description": "把字符串倒过来写。",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "要反转的字符串"}},
        "required": ["text"],
    },
}]


def reply(req_id, result):
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}, ensure_ascii=False)


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id", 0)

    if method == "initialize":
        out = reply(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "wild-server", "version": "0.1"},
        })
    elif method == "notifications/initialized":
        continue  # 通知不需要响应
    elif method == "tools/list":
        out = reply(req_id, {"tools": TOOLS})
    elif method == "tools/call":
        args = msg.get("params", {}).get("arguments", {})
        out = reply(req_id, {
            "content": [{"type": "text", "text": args.get("text", "")[::-1]}],
            "isError": False,
        })
    else:
        out = json.dumps({"jsonrpc": "2.0", "id": req_id,
                          "error": {"code": -32601, "message": f"未知方法: {method}"}},
                         ensure_ascii=False)

    print(out, flush=True)
