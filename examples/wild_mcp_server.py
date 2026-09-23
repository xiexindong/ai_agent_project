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

# 【什么】本 Server 暴露的唯一一个工具
# 【作用】tools/list 时把这份描述原样返回，Client 据此知道能调什么、传什么
TOOLS = [{
    "name": "string_reverse",                                        # 工具名
    "description": "把字符串倒过来写。",                              # 给模型看的说明
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "要反转的字符串"}},   # 唯一入参
        "required": ["text"],                                        # text 必填
    },
}]


# 【什么】按 JSON-RPC 2.0 拼一条成功响应
# 【作用】ensure_ascii=False 保证中文不被转成 \uXXXX
def reply(req_id, result):
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}, ensure_ascii=False)


# 【什么】主循环：一行一条请求，一行一条响应
# 【作用】只读 stdin、只写 stdout，靠报文格式与任何标准 MCPClient 对接
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id", 0)

    # 【什么】握手：报协议版本、能力与自身信息
    # 【作用】Client 靠这段确认对面是能聊的 MCP Server
    if method == "initialize":
        out = reply(req_id, {
            "protocolVersion": "2024-11-05",                     # 协议版本
            "capabilities": {"tools": {}},                        # 只声明 tools 能力
            "serverInfo": {"name": "wild-server", "version": "0.1"},
        })
    # 【什么】初始化完成通知
    # 【作用】通知类消息没有 id，不需要回包，直接跳过
    elif method == "notifications/initialized":
        continue
    # 【什么】返回工具清单
    # 【作用】把 TOOLS 原样回给 Client
    elif method == "tools/list":
        out = reply(req_id, {"tools": TOOLS})
    # 【什么】执行工具调用
    # 【作用】取 arguments.text 反转后作为 text 内容返回，isError=False 表示成功
    elif method == "tools/call":
        args = msg.get("params", {}).get("arguments", {})
        out = reply(req_id, {
            "content": [{"type": "text", "text": args.get("text", "")[::-1]}],
            "isError": False,
        })
    # 【什么】未知方法的兜底分支
    # 【作用】按协议返回 -32601（方法未找到），而不是静默不回
    else:
        out = json.dumps({"jsonrpc": "2.0", "id": req_id,
                          "error": {"code": -32601, "message": f"未知方法: {method}"}},
                         ensure_ascii=False)

    print(out, flush=True)
