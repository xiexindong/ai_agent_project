"""
计算器工具
==========
基础数学计算工具，支持四则运算、幂运算、三角函数。

【为什么不用 eval()？】eval 能直接算 "3*7"，但也能跑 "os.system('rm -rf /')"。
    用 Python 自带的 ast 模块把表达式解析成"语法树"，再逐节点白名单求值——
    遇到 _SAFE_OPS / _SAFE_FUNCS 之外的节点（如调用 open/import os）直接抛错，
    类似前端用 DOMPurify 过滤 XSS，先把能干的事圈死。

【合法/非法表达式对比】
    ✓ "3*7"           → 21
    ✓ "sqrt(16)"      → 4
    ✓ "sin(pi/2)"     → 1.0
    ✓ "2**10"         → 1024
    ✗ "__import__('os')" → ValueError: 不支持的函数: __import__
    ✗ "open('/etc/passwd')" → ValueError: 不支持的函数: open
    ✗ "1/0"           → ZeroDivisionError（被 try/except 包成 ToolResult.error）
"""
import math
import ast
import operator
from typing import Dict
from modules.tools.base_tool import BaseTool, ToolResult

_SAFE_OPS = {
    ast.Add: operator.add,              # 加法 a + b
    ast.Sub: operator.sub,              # 减法 a - b
    ast.Mult: operator.mul,             # 乘法 a * b
    ast.Div: operator.truediv,          # 除法 a / b，除数为 0 会抛 ZeroDivisionError
    ast.Pow: operator.pow,              # 幂 a ** b，如 2**10 = 1024
    ast.USub: operator.neg,             # 取负 -a
    ast.UAdd: operator.pos,             # 正号 +a（值不变，保持表达式可通过）
    ast.Mod: operator.mod,              # 取模 a % b
    ast.FloorDiv: operator.floordiv,    # 整除 a // b
}

_SAFE_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,   # 三角函数（入参为弧度）
    "asin": math.asin, "acos": math.acos, "atan": math.atan,   # 反三角函数
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,   # 开方与自然对数、常用对数
    "exp": math.exp, "abs": abs, "ceil": math.ceil, "floor": math.floor,   # 指数、绝对值、上下取整
    "pi": math.pi, "e": math.e, "degrees": math.degrees, "radians": math.radians,   # 常量与角度弧度换算
}


# 【什么】安全数学计算器（AST 白名单求值）
# 【作用】替代 eval()，只放行白名单运算符与函数，挡住任意代码执行
class CalculatorTool(BaseTool):
    """安全数学计算器：仅允许数学表达式，禁止任意代码执行。"""

    def __init__(self):
        super().__init__(
            name="calculator",   # 工具名；Permission 里对应 tool:calculator
            description="数学计算器，支持四则运算、幂运算、三角函数等。输入数学表达式字符串，返回计算结果。",
            parameters={
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2+3*4', 'sqrt(16)', 'sin(pi/2)'",
                    }   # 唯一的入参；传非字符串会在 ast.parse 处报错
                },
                "required": ["expression"],   # 必填；空串或纯空白会在 execute 里先被拦掉
            },
        )

    # 【什么】递归遍历 AST 求值
    # 【作用】逐节点对照白名单，白名单外的节点一律抛 ValueError
    def _eval_node(self, node):
        """递归安全求值 AST 节点。"""
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量: {node.value}")
        if isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if not op:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if not op:
                raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
            return op(self._eval_node(node.operand))
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            func = _SAFE_FUNCS.get(func_name)
            if not func:
                raise ValueError(f"不支持的函数: {func_name}")
            args = [self._eval_node(a) for a in node.args]
            return func(*args)
        if isinstance(node, ast.Name):
            val = _SAFE_FUNCS.get(node.id)
            if val is None:
                raise ValueError(f"未知变量: {node.id}")
            return val
        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    # 【什么】解析并执行数学表达式
    # 【作用】ast.parse(mode="eval") 后交给 _eval_node，异常一律转成失败 ToolResult
    def execute(self, expression: str, **kwargs) -> ToolResult:
        """安全执行数学表达式。"""
        expr = expression.strip()
        if not expr:
            return ToolResult(success=False, error="表达式为空")
        try:
            tree = ast.parse(expr, mode="eval")
            result = self._eval_node(tree)
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
            return ToolResult(success=True, result=result)
        except Exception as e:
            return ToolResult(success=False, error=f"计算失败: {e}")
