"""
MCP 协议解析模块
================
基于 JSON-RPC 2.0 实现 Model Context Protocol 标准化消息封装与解析。
MCP 核心方法：
- initialize: 连接握手
- tools/list: 获取工具列表
- tools/call: 调用工具
"""
import json                                                         # 标准库：JSON 编解码，协议帧的序列化与解析都靠它
from dataclasses import dataclass, field                           # dataclass 装饰器与字段工厂，声明协议消息结构
from typing import Any, Optional                                   # 类型标注工具

# 【什么】JSON-RPC 2.0 标准错误码
# 【作用】Server/Client 统一按码表达错误类别，调用方看码即可判断该怎么处理
# 【为什么】错误码是协议层的通用语言：跨语言两端只看数字就能对齐错误语义
PARSE_ERROR = -32700        # 解析失败：收到的文本不是合法 JSON
INVALID_REQUEST = -32600    # 请求无效：消息结构不符合 JSON-RPC 请求格式
METHOD_NOT_FOUND = -32601   # 方法不存在：Server 没有实现这个 method
INVALID_PARAMS = -32602     # 参数无效：method 存在，但 params 缺失或类型不对
INTERNAL_ERROR = -32603     # 内部错误：Server 处理过程中抛了未捕获异常


# 【什么】一条 JSON-RPC 2.0 请求的数据类
# 【作用】把 method/params/id 打包成对象，序列化成一行 JSON 后写进 Server 的 stdin
# 【为什么】用 dataclass 而非手写 dict：字段名拼错在构造期就暴露，不到线上才发现
@dataclass                                                           # 声明为数据类，自动生成构造器
class JSONRPCRequest:                                                # JSON-RPC 2.0 请求结构
    """JSON-RPC 2.0 请求。"""
    method: str                                  # 方法名，如 initialize / tools/list / tools/call
    params: dict = field(default_factory=dict)   # 方法参数；default_factory 保证每条请求互不共享
    id: int = 0                                  # 请求号，响应原样带回，用于配对请求与响应

    # 【什么】把请求对象转成符合协议的 dict
    # 【作用】固定写入 jsonrpc="2.0"，Server 端靠这个字段判版本
    # 【为什么】版本字段由这里统一写死，业务代码永远不会忘了带或写错版本号
    def to_dict(self) -> dict:                                     # 返回协议 dict
        return {                                                   # 组装标准请求结构
            "jsonrpc": "2.0",                                      # 协议版本标识（固定 2.0）
            "method": self.method,                                 # 方法名
            "params": self.params,                                 # 方法参数
            "id": self.id,                                         # 请求号
        }

    # 【什么】把 dict 序列化成一行 JSON 字符串
    # 【作用】stdio 传输是一行一帧，ensure_ascii=False 保留中文原样，不做 \uXXXX 转义
    # 【为什么】中文转义后体积膨胀 3 倍且不可读；一行一帧正好配合对端 readline
    def to_json(self) -> str:                                      # 返回一行 JSON 文本
        return json.dumps(obj=self.to_dict(), ensure_ascii=False)  # 序列化：不转义非 ASCII 字符


# 【什么】JSON-RPC 2.0 错误对象的数据类
# 【作用】只装在响应里，code+message 必填，data 是可选的补充信息
# 【为什么】错误独立成类：与正常结果类型分开，响应里两者互斥不会混淆
@dataclass                                                           # 声明为数据类
class JSONRPCError:                                                  # JSON-RPC 2.0 错误对象
    """JSON-RPC 2.0 错误对象。"""
    code: int           # 错误码，取上面那五个常量之一
    message: str        # 错误描述，给人看的
    data: Any = None    # 可选的补充信息（如 Server 侧原始异常文本）

    # 【什么】把错误对象转成 dict
    # 【作用】data 为 None 时不写这个键，避免响应里出现 "data": null 的噪声
    # 【为什么】协议允许省略 data；带上 null 会让严格的对端校验器报错
    def to_dict(self) -> dict:                                     # 返回错误 dict
        d = {"code": self.code, "message": self.message}           # 先装必填两字段
        if self.data is not None:                                  # 有补充信息才写
            d["data"] = self.data                                  # 附加 data 键
        return d                                                   # 返回组装结果


# 【什么】一条 JSON-RPC 2.0 响应的数据类
# 【作用】成功带 result、失败带 error，二者互斥，Client 只看 is_error 分支
# 【为什么】互斥封装在一个类里：调用方不用自己保证"两个字段只填一个"
@dataclass                                                           # 声明为数据类
class JSONRPCResponse:                                               # JSON-RPC 2.0 响应结构
    """JSON-RPC 2.0 响应。"""
    id: int                                # 对应请求的 id，用于配对
    result: Any = None                     # 成功结果；无返回内容时是 None
    error: Optional[JSONRPCError] = None   # 错误对象；非 None 就表示这次调用失败了

    # 【什么】把响应对象转成 dict
    # 【作用】有 error 只写 error 键、有 result 只写 result 键，两者不会同时出现
    # 【为什么】协议规定 result 与 error 互斥；序列化层强制保证，出错在源头
    def to_dict(self) -> dict:                                     # 返回协议 dict
        d = {"jsonrpc": "2.0", "id": self.id}                      # 先装版本与请求号
        if self.error:                                             # 失败分支
            d["error"] = self.error.to_dict()                      # 写 error 键
        else:                                                      # 成功分支
            d["result"] = self.result                              # 写 result 键
        return d                                                   # 返回组装结果

    # 【什么】把响应序列化成一行 JSON 字符串
    # 【作用】Server 端 print 这一行到 stdout，Client 的 readline 正好读一帧
    # 【为什么】与请求同一套序列化约定：不转义中文、单行成帧
    def to_json(self) -> str:                                      # 返回一行 JSON 文本
        return json.dumps(self.to_dict(), ensure_ascii=False)      # 序列化：不转义非 ASCII 字符

    # 【什么】判定这次调用是否失败
    # 【作用】调用方统一用 is_error 分支，不用自己判 result 是不是 None
    # 【为什么】"result 为 None"可能是合法的空结果，只有 error 非 None 才是失败
    @property                                                      # 声明为只读属性
    def is_error(self) -> bool:                                    # 返回布尔：是否为错误响应
        return self.error is not None                              # 看 error 字段是否被填


# 【什么】构造一条请求的 JSON 字符串
# 【作用】Client 侧统一入口；req_id 默认 1，多请求并发时应显式传不同 id
# 【为什么】收敛成一个函数：请求结构只有一个出处，改格式不用满处找
def create_request(method: str, params: Optional[dict] = None, req_id: int = 1) -> str:  # 入参：方法名+可选参数+请求号
    """创建 JSON-RPC 请求 JSON 字符串。"""
    req = JSONRPCRequest(method=method, params=params or {}, id=req_id)  # 构造请求对象（None 参数兜成空 dict）
    return req.to_json()                                           # 序列化成一行 JSON 返回


# 【什么】构造一条成功响应的 JSON 字符串
# 【作用】Server 处理完 tools/list、tools/call 等方法后回给 Client
# 【为什么】成功路径只有一个构造入口，响应结构不会各处拼得不一致
def create_response(req_id: int, result: Any) -> str:              # 入参：请求号+结果内容
    """创建成功响应 JSON 字符串。"""
    resp = JSONRPCResponse(id=req_id, result=result)               # 构造成功响应对象
    return resp.to_json()                                          # 序列化成一行 JSON 返回


# 【什么】构造一条错误响应的 JSON 字符串
# 【作用】失败路径统一走这里，code 用协议常量，data 可带补充信息
# 【为什么】错误响应也走同一个数据类：互斥逻辑由 to_dict 保证，调用方只管填字段
def create_error(req_id: int, code: int, message: str, data: Any = None) -> str:  # 入参：请求号+错误码+描述+可选补充
    """创建错误响应 JSON 字符串。"""
    resp = JSONRPCResponse(                                        # 构造错误响应对象
        id=req_id,                                                 # 带回请求号
        error=JSONRPCError(code=code, message=message, data=data), # 装错误对象
    )
    return resp.to_json()                                          # 序列化成一行 JSON 返回


# 【什么】把一行 JSON 文本解析成 dict
# 【作用】协议入口的三道校验：不是 JSON、不是对象、版本不是 2.0，任一不过都抛 ValueError
# 【为什么】校验前置在解析层：坏消息在进业务逻辑之前就被拦下，错误语义统一是 ValueError
def parse_message(raw: str) -> dict:                               # 入参为一行 JSON 文本，返回解析后的 dict
    """解析 JSON-RPC 消息，返回 dict。"""
    try:
        msg = json.loads(raw)                                      # 先按 JSON 解析
    except json.JSONDecodeError as e:                              # 不是合法 JSON
        raise ValueError(f"JSON解析失败: {e}")                     # 转成 ValueError 统一错误类型
    if not isinstance(msg, dict):                                  # 是 JSON 但不是对象（如数组/数字）
        raise ValueError("消息必须是JSON对象")                     # 抛结构错误
    # 版本必须是 "2.0"：少了这道判断，1.0 的消息会被当成 2.0 处理，字段含义对不上
    if msg.get("jsonrpc") != "2.0":                                # 版本字段缺失或不是 2.0
        raise ValueError("仅支持 JSON-RPC 2.0")                    # 抛版本错误
    return msg                                                     # 校验通过，返回消息 dict
