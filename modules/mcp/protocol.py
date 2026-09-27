"""
MCP 协议解析模块
================
基于 JSON-RPC 2.0 实现 Model Context Protocol 标准化消息封装与解析。
MCP 核心方法：
- initialize: 连接握手
- tools/list: 获取工具列表
- tools/call: 调用工具
"""
import json
from dataclasses import dataclass, field
from typing import Any, Optional

# 【什么】JSON-RPC 2.0 标准错误码
# 【作用】Server/Client 统一按码表达错误类别，调用方看码即可判断该怎么处理
PARSE_ERROR = -32700        # 解析失败：收到的文本不是合法 JSON
INVALID_REQUEST = -32600    # 请求无效：消息结构不符合 JSON-RPC 请求格式
METHOD_NOT_FOUND = -32601   # 方法不存在：Server 没有实现这个 method
INVALID_PARAMS = -32602     # 参数无效：method 存在，但 params 缺失或类型不对
INTERNAL_ERROR = -32603     # 内部错误：Server 处理过程中抛了未捕获异常


# 【什么】一条 JSON-RPC 2.0 请求的数据类
# 【作用】把 method/params/id 打包成对象，序列化成一行 JSON 后写进 Server 的 stdin
@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 请求。"""
    method: str                                  # 方法名，如 initialize / tools/list / tools/call
    params: dict = field(default_factory=dict)   # 方法参数；default_factory 保证每条请求互不共享
    id: int = 0                                  # 请求号，响应原样带回，用于配对请求与响应

    # 【什么】把请求对象转成符合协议的 dict
    # 【作用】固定写入 jsonrpc="2.0"，Server 端靠这个字段判版本
    def to_dict(self) -> dict:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }

    # 【什么】把 dict 序列化成一行 JSON 字符串
    # 【作用】stdio 传输是一行一帧，ensure_ascii=False 保留中文原样，不做 \uXXXX 转义
    def to_json(self) -> str:
        return json.dumps(obj=self.to_dict(), ensure_ascii=False)


# 【什么】JSON-RPC 2.0 错误对象的数据类
# 【作用】只装在响应里，code+message 必填，data 是可选的补充信息
@dataclass
class JSONRPCError:
    """JSON-RPC 2.0 错误对象。"""
    code: int           # 错误码，取上面那五个常量之一
    message: str        # 错误描述，给人看的
    data: Any = None    # 可选的补充信息（如 Server 侧原始异常文本）

    # 【什么】把错误对象转成 dict
    # 【作用】data 为 None 时不写这个键，避免响应里出现 "data": null 的噪声
    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


# 【什么】一条 JSON-RPC 2.0 响应的数据类
# 【作用】成功带 result、失败带 error，二者互斥，Client 只看 is_error 分支
@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 响应。"""
    id: int                                # 对应请求的 id，用于配对
    result: Any = None                     # 成功结果；无返回内容时是 None
    error: Optional[JSONRPCError] = None   # 错误对象；非 None 就表示这次调用失败了

    # 【什么】把响应对象转成 dict
    # 【作用】有 error 只写 error 键、有 result 只写 result 键，两者不会同时出现
    def to_dict(self) -> dict:
        d = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return d

    # 【什么】把响应序列化成一行 JSON 字符串
    # 【作用】Server 端 print 这一行到 stdout，Client 的 readline 正好读一帧
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    # 【什么】判定这次调用是否失败
    # 【作用】调用方统一用 is_error 分支，不用自己判 result 是不是 None
    @property
    def is_error(self) -> bool:
        return self.error is not None


# 【什么】构造一条请求的 JSON 字符串
# 【作用】Client 侧统一入口；req_id 默认 1，多请求并发时应显式传不同 id
def create_request(method: str, params: Optional[dict] = None, req_id: int = 1) -> str:
    """创建 JSON-RPC 请求 JSON 字符串。"""
    req = JSONRPCRequest(method=method, params=params or {}, id=req_id)
    return req.to_json()


# 【什么】构造一条成功响应的 JSON 字符串
# 【作用】Server 处理完 tools/list、tools/call 等方法后回给 Client
def create_response(req_id: int, result: Any) -> str:
    """创建成功响应 JSON 字符串。"""
    resp = JSONRPCResponse(id=req_id, result=result)
    return resp.to_json()


# 【什么】构造一条错误响应的 JSON 字符串
# 【作用】失败路径统一走这里，code 用协议常量，data 可带补充信息
def create_error(req_id: int, code: int, message: str, data: Any = None) -> str:
    """创建错误响应 JSON 字符串。"""
    resp = JSONRPCResponse(
        id=req_id,
        error=JSONRPCError(code=code, message=message, data=data),
    )
    return resp.to_json()


# 【什么】把一行 JSON 文本解析成 dict
# 【作用】协议入口的三道校验：不是 JSON、不是对象、版本不是 2.0，任一不过都抛 ValueError
def parse_message(raw: str) -> dict:
    """解析 JSON-RPC 消息，返回 dict。"""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}")
    if not isinstance(msg, dict):
        raise ValueError("消息必须是JSON对象")
    # 版本必须是 "2.0"：少了这道判断，1.0 的消息会被当成 2.0 处理，字段含义对不上
    if msg.get("jsonrpc") != "2.0":
        raise ValueError("仅支持 JSON-RPC 2.0")
    return msg
