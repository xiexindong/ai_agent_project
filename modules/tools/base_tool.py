"""
工具基类模块
=============
定义本地工具标准接口：名称、描述、参数 schema、执行方法。
所有工具（本地 + MCP 远程适配）统一继承此基类，让 ToolRegistry 和 ReActAgent
完全不感知工具是跑在进程内还是子进程里。

【统一接口的设计】BaseTool 规定了 4 个字段 + 1 个抽象方法：
    name        : 工具名（Function Calling 用它匹配 tool_calls[].name）
    description : 给 LLM 看的"这工具干啥用"（影响 LLM 选不选它）
    parameters  : JSON Schema 形参说明书（影响 LLM 怎么传参）
    execute()   : 真正干活的抽象方法，子类必须实现
    run()       : 模板方法（参数校验→execute→封装 ToolResult），子类不用管

【本地工具 vs MCP 工具共享接口例子】
    本地 CalculatorTool（继承 BaseTool，tool_type="local"）：
        run({"expression":"3*7"}) → ToolResult(success=True, result=21)

    远程 MCPToolAdapter（适配外部 web_search，tool_type="mcp"）：
        run({"query":"今天天气"}) → ToolResult(success=True, result=[...])

    对 ReActAgent 来说两者完全等价：都拿 ToolResult，不关心工具在哪跑。
"""
from abc import ABC, abstractmethod  # 抽象基类工具：ABC=抽象基类标记，abstractmethod=强制子类实现
from dataclasses import dataclass, field  # dataclass 自动生成样板代码；field 用于给可变默认值（如 dict）
from typing import Any, Optional  # 类型注解：任意类型/可选
from modules.common.logger import get_logger  # 统一日志工厂
from modules.common.exceptions import ToolException  # 工具异常基类：参数缺失等用它抛

_logger = get_logger("tool")  # 模块级 logger：本文件日志统一带 "tool" 前缀


# 【什么】工具执行结果对象（本地与 MCP 共用）
# 【作用】把成功值、错误原因、耗时打包成统一返回，调用方不看异常只看它
@dataclass  # 数据类装饰器：按字段声明自动生成 __init__/__repr__/__eq__
class ToolResult:  # 工具执行结果对象：success/result/error/duration_ms/meta
    """
    工具执行结果（本地和 MCP 工具共用，ReActAgent 只看这个对象）。

    【字段含义】
        success     : True=成功 / False=失败（含被安全闸门拦截）
        result      : 成功时的返回值（任意类型，如 int / dict / list）
        error       : 失败时的错误描述（会回喂给 LLM，模型据此调整策略）
        duration_ms : 执行耗时毫秒数（用于性能日志，由 run() 自动填充）

    【两种结果对比】
        成功: ToolResult(success=True, result=21, duration_ms=12)
        失败: ToolResult(success=False, error="工具不存在: calc", duration_ms=0)
    """
    success: bool            # True=成功 / False=失败（含被安全闸门拦截）
    result: Any = None       # 成功返回值，任意类型（int / dict / list）
    error: str = ""          # 失败原因，会回喂给 LLM 让它改策略
    duration_ms: int = 0     # 执行耗时毫秒，由 run() 自动填充
    # 【什么】附加信息（Harness 写入：风险等级、是否被裁剪/脱敏、拦截码）
    # 【作用】不改主字段就能带上治理信息；默认空字典，旧代码无感
    meta: dict = field(default_factory=dict)  # 附加信息：Harness 写入的风险等级/拦截码等，默认空 dict

    # 【什么】把结果对象转成普通字典
    # 【作用】便于序列化进日志、链路或接口返回体
    def to_dict(self) -> dict:  # 转 dict：方便 JSON 序列化与日志输出
        return {  # 按固定 key 拼 dict
            "success": self.success,  # 是否成功
            "result": self.result,  # 成功时的返回值
            "error": self.error,  # 失败原因
            "duration_ms": self.duration_ms,  # 耗时毫秒
            "meta": dict(self.meta or {}),  # meta 拷贝一份（防外部改到内部状态）
        }


# 【什么】所有工具的抽象基类
# 【作用】把「名称/描述/参数schema/执行」统一成一个接口，屏蔽本地与 MCP 差异
class BaseTool(ABC):  # 继承 ABC 抽象基类：不能直接实例化，子类必须实现 execute()
    """
    本地工具基类。
    子类需实现 name / description / parameters / execute。

    【继承关系类比】类似前端 React 的 BaseComponent / 后端 NestJS 的 BaseController——
    把所有工具的"通用骨架"放这里，子类只填自己的业务逻辑。
    """

    # ------------------------------------------------------------------
    # 【能力声明】Harness 据此裁决：风险多高、要不要人批准、在哪执行、需要什么权限
    #
    #   子类可用三种写法之一覆盖：
    #     capability = {"risk": "high", "side_effect": True, "sandbox": "subprocess"}
    #     capability = ToolCapability(name="xxx", risk=RiskLevel.HIGH, ...)
    #     def capability(self): return ToolCapability(...)
    #
    #   不声明 = None，Harness 按 TOOL_DEFAULT_RISK（默认 low）处理，
    #   即"只读、可直接执行"——与接入 Harness 之前的行为一致。
    # ------------------------------------------------------------------
    capability: Any = None  # 能力声明：Harness 治理用；子类可覆盖为 dict 或 ToolCapability，默认 None 表示只读低风险

    # 【什么】构造函数：固化工具的三要素
    # 【作用】把调用方传进来的原始 parameters 归一成标准 schema
    def __init__(self, name: str, description: str, parameters: dict):  # 构造：工具名/描述/参数 schema
        self.name = name                        # 工具名，Function Calling 用它匹配 tool_calls[].name
        self.description = description          # 给 LLM 看的用途说明，影响它选不选这个工具
        self.parameters_schema = {              # 对外暴露的标准 schema，只留下面三个字段
            "type": "object",                   # 固定为 object：JSON Schema 顶层类型
            "properties": parameters.get("properties", {}),   # 形参说明书；调用方没传则取空 dict
            "required": parameters.get("required", []),       # 必填参数名列表；没传则取空 list
        }

    # 【什么】工具类型标识
    # 【作用】区分 local / mcp，注册日志与 mcp 工具的权限码都看这个值
    @property  # property 装饰器：把方法变成只读属性
    def tool_type(self) -> str:  # 返回工具类型字符串
        """工具类型：local / mcp。"""
        return "local"  # 基类默认 local，MCP 适配器子类覆盖为 "mcp"

    # 【什么】转成 OpenAI Function Calling 的工具描述
    # 【作用】这份 schema 发给模型，决定它能调哪些工具、传什么参数
    def to_openai_format(self) -> dict:  # 转成 OpenAI 标准 tools 数组元素格式
        """转换为 OpenAI Function Calling 标准格式。"""
        return {  # 固定两层结构：外层 type + 内层 function
            "type": "function",  # 固定 type
            "function": {  # function 对象
                "name": self.name,  # 工具名
                "description": self.description,  # 用途描述
                "parameters": self.parameters_schema,  # 参数 schema（已归一化）
            },
        }

    # 【什么】参数校验：只查必填项是否在且不为 None
    # 【作用】把非法参数挡在 execute 之前，缺参抛 ToolException
    def validate_params(self, params: dict) -> dict:  # 校验参数合法性，通过后返回原/防空 dict
        """参数合法性校验：检查必填项（边界2）。"""
        if params is None:  # 调用方没传参数或传了 None
            params = {}  # 防空：转空 dict，避免后续报错
        required = self.parameters_schema.get("required", [])  # 取出 schema 里的必填参数列表
        missing = [r for r in required if r not in params or params[r] is None]  # 列表推导：缺名字或值为 None 都算缺失
        if missing:  # 有缺失参数
            raise ToolException(  # 抛异常让 run() 捕获并转成失败 ToolResult
                f"工具参数缺失: {self.name} 缺少必填参数 {missing}",  # 错误主信息（给 LLM 看）
                f"provided={list(params.keys())}",  # 附带给出实际传了哪些 key（排查用）
            )
        return params  # 校验通过：原样返回参数 dict

    # 【什么】工具真正的业务逻辑（子类必须实现）
    # 【作用】只负责干活；参数校验与结果封装由 run() 统一代劳
    # 【签名为什么是 *args, **kwargs】各工具入参不同（calculator 收 expression、
    #   data_query 收 sql 等），基类只能放开参数形态，具体参数由子类声明。
    @abstractmethod  # 抽象方法：子类必须覆盖，否则实例化时报 TypeError
    def execute(self, *args, **kwargs) -> ToolResult:  # 子类实现真正业务逻辑
        """执行工具逻辑，子类实现。"""
        pass  # 基类空实现，占位等子类覆盖

    # 【什么】统一执行入口（模板方法）
    # 【作用】校验→执行→补耗时→包成 ToolResult，异常也转成失败结果不向上抛
    def run(self, params: dict) -> ToolResult:  # 外部调用入口：校验→执行→封装结果
        """统一执行入口：参数校验 → 执行 → 结果封装。"""
        import time  # 内置 time 模块：计时用
        start = time.time()  # 记开始时间戳（秒）
        try:  # 主执行路径
            validated = self.validate_params(params)  # 先校验参数（缺参会抛 ToolException）
            result = self.execute(**validated)  # 把校验后的参数打散成关键字参数传给 execute
            if not isinstance(result, ToolResult):  # execute 返回的不是 ToolResult（比如直接返回 int/str）
                result = ToolResult(success=True, result=result)  # 自动包成成功 ToolResult
            result.duration_ms = int((time.time() - start) * 1000)  # 补耗时字段：秒→毫秒取整
            _logger.info(  # 记执行结果日志
                f"工具执行: {self.name} | status={'成功' if result.success else '失败'} "  # 工具名 + 成功/失败状态
                f"| 耗时={result.duration_ms}ms"  # 执行耗时毫秒
            )
            return result  # 返回封装好的 ToolResult
        except ToolException as e:  # 参数校验或业务层抛出的 ToolException
            duration = int((time.time() - start) * 1000)  # 异常也要算耗时
            _logger.error(f"工具参数异常: {self.name} | {e.message}")  # 记 error 日志
            return ToolResult(success=False, error=e.message, duration_ms=duration)  # 包成失败 ToolResult（不向上抛）
        except Exception as e:  # 一切其他异常（业务逻辑未捕获的）
            duration = int((time.time() - start) * 1000)  # 异常耗时也要算
            _logger.error(f"工具执行异常: {self.name} | {type(e).__name__}: {e}")  # 记 error 日志：异常类名 + 消息
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}", duration_ms=duration)  # 包成失败 ToolResult（不向上抛）
