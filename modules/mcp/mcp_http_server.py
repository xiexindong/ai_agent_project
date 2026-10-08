"""
远程 MCP Server（模拟实现：把项目自带的 MCPServer 挂到 HTTP 上）
================================================================
给 MCPHttpClient 当对端用，解决"想试远程 MCP 但没有第二台机器"的问题。
它就是「远程机器上部署的那个 MCP Server」，协议与 stdio 版完全一致（JSON-RPC 2.0），
只是传输层从 stdin/stdout 换成了 HTTP。

用法（一般不手工执行，正常由项目命令拉起）：

    python main.py mcp-http-server --host 127.0.0.1 --port 8800
    # 等价于
    uvicorn modules.mcp.mcp_http_server:app --host 0.0.0.0 --port 8800
    python -m modules.mcp.mcp_http_server

【完整数据流（Agent 客户端 ↔ 本服务）】
    1. Agent 侧：MCPHttpClient.connect() → POST /mcp {"method":"initialize"}
    2. 本服务：交给 MCPServer.handle_request → 回 serverInfo（与 stdio 版同一份逻辑）
    3. Agent 侧：POST /mcp {"method":"notifications/initialized"} → 本服务按规范不回包（202）
    4. Agent 侧：POST /mcp {"method":"tools/list"} → 回工具清单（名字 + 描述 + inputSchema）
    5. 网关把清单建成工具目录 → Tool-RAG 索引 → 每轮语义召回 top-k → 按需注册
    6. Agent 侧：POST /mcp {"method":"tools/call","params":{"name":..,"arguments":{..}}}
    7. 本服务：MCPServer 分派到具体工具方法 → 结果包成 content:[{type:"text"}] 回传
    8. Agent 侧：MCPToolAdapter 提取文本 → ToolResult → 回喂 LLM
    9. 连不上 / 401 / 500 → 网关接住，按"该 server 不可用"降级，本地工具不受影响

【四个设计目的】
    1. 远程执行：工具跑在远端进程里，Agent 只发一个 HTTP 请求 —— 天然隔离，
       不像远程沙箱那样还要自研接口：MCP 自带协议，工具元数据/参数 schema/错误码都是规定好的
    2. 换传输不改业务：本服务复用的就是 modules/mcp/mcp_server.py 的 MCPServer，
       stdio 版与 HTTP 版共用同一份工具清单与协议处理，改工具只改一处
    3. 跨语言可接：对面只要是符合 MCP 的 Server（Node/Go 写的都行），本项目的
       MCPHttpClient 不用改一行；反过来本项目这一端也能被别的 Agent 客户端接
    4. 可降级：远程 server 挂了只是"这一类工具不可用"，目录建不出来也不抛异常，
       本地工具与该 server 之外的远程工具照常可用

【为什么工具执行失败也返回 200】
    200 表示"Server 受理并按协议回了包"；工具业务性失败放在 result.isError 里，
    只有"协议层错误 / 鉴权不过"才用非 2xx —— 客户端据此区分要不要熔断降级。

【输入输出对照】
    POST /mcp in  {"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
             out  {"jsonrpc":"2.0","id":1,"result":{"serverInfo":{...},"capabilities":{"tools":{}}}}
    POST /mcp in  {"jsonrpc":"2.0","id":2,"method":"tools/call",
                   "params":{"name":"knowledge_stats","arguments":{}}}
             out  {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{...}"}],
                   "isError":false}}
    POST /mcp in  {"jsonrpc":"2.0","id":3,"method":"notifications/initialized","params":{}}
             out  202 空体（通知类按 JSON-RPC 规范不该有响应）
    GET  /health → {"status":"ok","protocol":"jsonrpc-2.0","endpoint":"/mcp"}
"""
from __future__ import annotations                                # 启用注解前向引用，类型标注里的类名可延迟解析

import json                                                       # 标准库：JSON 编解码（响应字符串 ↔ dict 互转）
import os                                                         # 标准库：读环境变量（MCP_SERVER_TOKEN / MCP_SERVER_PORT）
from typing import Any, Dict, Optional                            # 类型标注工具

from fastapi import FastAPI, Header, HTTPException                # Web 框架：应用类、请求头注入、HTTP 异常
from fastapi.responses import JSONResponse                        # Web 框架：JSON 响应封装

from modules.common.logger import get_logger                      # 项目统一日志入口
from modules.mcp.mcp_server import MCPServer                      # 复用 stdio 版服务端：同一份工具清单与协议处理
from modules.mcp.protocol import INVALID_REQUEST, create_error, parse_message  # 协议错误码与帧工具

_logger = get_logger("mcp_http_server")                           # 模块级日志器，名字 "mcp_http_server" 用于日志过滤

# 【什么】真正干活的是项目自带的 MCPServer（同一份工具清单与实现）
# 【作用】stdio 版（python -m modules.mcp.mcp_server）与这里的 HTTP 版共用它：
#   换传输不改业务，协议处理只有一份，新增工具只改 mcp_server.py 一处
# 【为什么】模块级单例：进程生命周期内只有一个实例，避免每请求重建工具清单
_server = MCPServer()                                             # 模块级服务端单例

app = FastAPI(title="远程 MCP Server（模拟）", version="1.0")     # FastAPI 应用实例（uvicorn 的挂载点）


# 【什么】健康检查
# 【作用】确认服务活着；curl 一下就知道起没起来，客户端握手前也能先探一把
# 【为什么】负载均衡/容器编排需要存活探针；本地调试也需要一个最简单的确认手段
@app.get("/health")                                               # 注册 GET /health 路由
def health() -> dict:                                             # 返回健康状态字典
    return {"status": "ok", "protocol": "jsonrpc-2.0", "endpoint": "/mcp"}  # 状态 + 协议标识 + 协议端点路径


# 【什么】调试用的工具清单（不走 JSON-RPC，直接 curl 就能看）
# 【作用】curl http://127.0.0.1:8800/tools 即可确认远端到底暴露了哪些工具，
#   排查"模型看不到某工具"时比走一遍握手省事
# 【为什么】调试端点绕过协议层，直达内部清单，所见即所得
@app.get("/tools")                                                # 注册 GET /tools 路由
def list_tools() -> dict:                                         # 返回工具清单字典
    return {"tools": _server._tools}                              # 直读服务端实例的工具清单（私有字段，仅限调试端点）


# 【什么】MCP 协议端点：一问一答，一次 POST 一个 JSON-RPC 帧
# 【作用】解析请求 → 交给 MCPServer.handle_request → 原样回响应；
#         这是整个服务唯一的业务入口，方法分派全在 mcp_server.py 里
# 【为什么】传输层只负责"搬帧"，协议语义全部复用 stdio 版实现，两处永不走岔
@app.post("/mcp")                                                 # 注册 POST /mcp 路由
def mcp_endpoint(payload: Dict[str, Any],                         # 请求体已被 FastAPI 解析成 dict
                 authorization: Optional[str] = Header(default=None, alias="Authorization")) -> Any:  # 注入 Authorization 头
    # 【什么是 Header(default=None)】Authorization 头不是必填：客户端没带就是 None，
    #   交给 _check_token 决定放不放行（服务端没设 MCP_SERVER_TOKEN 时不鉴权）
    _check_token(authorization)                                   # 先做令牌校验，不过直接抛 401

    try:
        # 【为什么要先 dumps 再 parse】协议层要求按"文本"校验（parse_message 收字符串），
        #   而 FastAPI 已经帮我们把 body 解析成 dict 了，所以这里转回字符串再交给协议层，
        #   好处是校验逻辑与 stdio 版共用同一份 parse_message，不写第二遍
        msg = parse_message(json.dumps(payload, ensure_ascii=False))  # dict 转回 JSON 文本后按协议解析校验
    except ValueError as e:                                  # 不是合法 JSON-RPC → -32600
        # 【为什么是 200 而不是 400】协议错误也要让客户端能解析出 error 对象，
        #   回 400 的话客户端只能看到 HTTP 状态码，拿不到 JSON-RPC 的 code/message
        return JSONResponse(status_code=200,
                            content=json.loads(create_error(0, INVALID_REQUEST, str(e))))  # 构造 -32600 错误帧并返回

    response = _server.handle_request(msg)                        # 交给共享的协议处理：返回响应 JSON 字符串或 None
    if response is None:                                     # notifications/*：按规范不回包
        return JSONResponse(status_code=202, content={})          # 回 202 空体表示"已受理，无响应"

    _logger.debug(f"MCP(HTTP) 处理: {msg.get('method')}")         # 记一条 debug 日志，含方法名
    # 【为什么 json.loads 之后再返回】handle_request 返回的是一条 JSON 字符串，
    #   这里转回 dict 交给 FastAPI 序列化，避免"字符串里套 JSON"被二次转义
    return JSONResponse(content=json.loads(response))             # 解析回 dict 后作为 JSON 响应返回


# 【什么】可选令牌校验
# 【作用】没设 MCP_SERVER_TOKEN 就不鉴权（本地调试省事）；设了必须带 Authorization: Bearer
# 【为什么】鉴权做成可选：本地模拟零配置即可跑，生产部署设个环境变量就有保护
def _check_token(authorization: Optional[str]) -> None:           # 入参为 Authorization 头值，校验不过抛 401
    expected = str(os.getenv("MCP_SERVER_TOKEN", "") or "").strip()  # 读期望令牌（环境变量），去空白
    if not expected:                                              # 服务端没设令牌
        return                                                    # 不鉴权，直接放行
    if not authorization or not authorization.startswith("Bearer "):  # 没带令牌或格式不对
        raise HTTPException(status_code=401, detail="缺少 Bearer 令牌")  # 抛 401
    if authorization[len("Bearer "):].strip() != expected:        # 令牌值比对（去掉 Bearer 前缀后精确比较）
        raise HTTPException(status_code=401, detail="MCP 令牌无效")    # 抛 401


# 【什么】脚本入口
# 【作用】直接 `python -m modules.mcp.mcp_http_server` 时把服务跑起来；
#   uvicorn 延迟导入（放在 if 里），不跑服务就不引入这个依赖
# 【为什么】模块被 import 时不该拉起服务器；入口收敛在 __main__ 块里
if __name__ == "__main__":                                        # 仅当作为脚本直接运行时
    import uvicorn                                                # 延迟导入 ASGI 服务器

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("MCP_SERVER_PORT", "8800") or 8800))  # 监听全部网卡，端口可由环境变量覆盖，默认 8800
