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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from modules.common.logger import get_logger
from modules.common.exceptions import ToolException

_logger = get_logger("tool")


@dataclass
class ToolResult:
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
    success: bool
    result: Any = None
    error: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


class BaseTool(ABC):
    """
    本地工具基类。
    子类需实现 name / description / parameters / execute。

    【继承关系类比】类似前端 React 的 BaseComponent / 后端 NestJS 的 BaseController——
    把所有工具的"通用骨架"放这里，子类只填自己的业务逻辑。
    """

    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters_schema = {
            "type": "object",
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", []),
        }

    @property
    def tool_type(self) -> str:
        """工具类型：local / mcp。"""
        return "local"

    def to_openai_format(self) -> dict:
        """转换为 OpenAI Function Calling 标准格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def validate_params(self, params: dict) -> dict:
        """参数合法性校验：检查必填项（边界2）。"""
        if params is None:
            params = {}
        required = self.parameters_schema.get("required", [])
        missing = [r for r in required if r not in params or params[r] is None]
        if missing:
            raise ToolException(
                f"工具参数缺失: {self.name} 缺少必填参数 {missing}",
                f"provided={list(params.keys())}",
            )
        return params

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具逻辑，子类实现。"""
        pass

    def run(self, params: dict) -> ToolResult:
        """统一执行入口：参数校验 → 执行 → 结果封装。"""
        import time
        start = time.time()
        try:
            validated = self.validate_params(params)
            result = self.execute(**validated)
            if not isinstance(result, ToolResult):
                result = ToolResult(success=True, result=result)
            result.duration_ms = int((time.time() - start) * 1000)
            _logger.info(
                f"工具执行: {self.name} | status={'成功' if result.success else '失败'} "
                f"| 耗时={result.duration_ms}ms"
            )
            return result
        except ToolException as e:
            duration = int((time.time() - start) * 1000)
            _logger.error(f"工具参数异常: {self.name} | {e.message}")
            return ToolResult(success=False, error=e.message, duration_ms=duration)
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            _logger.error(f"工具执行异常: {self.name} | {type(e).__name__}: {e}")
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}", duration_ms=duration)
