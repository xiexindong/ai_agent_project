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

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 请求。"""
    method: str
    params: dict = field(default_factory=dict)
    id: int = 0

    def to_dict(self) -> dict:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }

    def to_json(self) -> str:
        return json.dumps(obj=self.to_dict(), ensure_ascii=False)


@dataclass
class JSONRPCError:
    """JSON-RPC 2.0 错误对象。"""
    code: int
    message: str
    data: Any = None

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 响应。"""
    id: int
    result: Any = None
    error: Optional[JSONRPCError] = None

    def to_dict(self) -> dict:
        d = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @property
    def is_error(self) -> bool:
        return self.error is not None


def create_request(method: str, params: dict = None, req_id: int = 1) -> str:
    """创建 JSON-RPC 请求 JSON 字符串。"""
    req = JSONRPCRequest(method=method, params=params or {}, id=req_id)
    return req.to_json()


def create_response(req_id: int, result: Any) -> str:
    """创建成功响应 JSON 字符串。"""
    resp = JSONRPCResponse(id=req_id, result=result)
    return resp.to_json()


def create_error(req_id: int, code: int, message: str, data: Any = None) -> str:
    """创建错误响应 JSON 字符串。"""
    resp = JSONRPCResponse(
        id=req_id,
        error=JSONRPCError(code=code, message=message, data=data),
    )
    return resp.to_json()


def parse_message(raw: str) -> dict:
    """解析 JSON-RPC 消息，返回 dict。"""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}")
    if not isinstance(msg, dict):
        raise ValueError("消息必须是JSON对象")
    if msg.get("jsonrpc") != "2.0":
        raise ValueError("仅支持 JSON-RPC 2.0")
    return msg
