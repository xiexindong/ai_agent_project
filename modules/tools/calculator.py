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
import math  # 数学库：sin/cos/sqrt/log 等函数与 pi/e 常量来自这里
import ast  # 抽象语法树模块：把表达式字符串解析成 AST，替代危险 eval
import operator  # 运算符函数库：add/sub/mul 等，与 AST 节点类型配对用
from typing import Dict  # 类型注解：字典
from modules.tools.base_tool import BaseTool, ToolResult  # 工具基类与统一结果对象

_SAFE_OPS = {  # 允许的运算符白名单：AST 节点类型 → 对应运算函数
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

_SAFE_FUNCS = {  # 允许的函数与常量白名单：名字 → 可调用对象/常量值
    "sin": math.sin, "cos": math.cos, "tan": math.tan,   # 三角函数（入参为弧度）
    "asin": math.asin, "acos": math.acos, "atan": math.atan,   # 反三角函数
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,   # 开方与自然对数、常用对数
    "exp": math.exp, "abs": abs, "ceil": math.ceil, "floor": math.floor,   # 指数、绝对值、上下取整
    "pi": math.pi, "e": math.e, "degrees": math.degrees, "radians": math.radians,   # 常量与角度弧度换算
}


# 【什么】安全数学计算器（AST 白名单求值）
# 【作用】替代 eval()，只放行白名单运算符与函数，挡住任意代码执行
class CalculatorTool(BaseTool):  # 安全数学计算器：AST 白名单求值，禁止 eval()
    """安全数学计算器：仅允许数学表达式，禁止任意代码执行。

    【输入输出对照】
        run({"expression": "3*7"})            → ToolResult(success=True,  result=21)
        run({"expression": "sqrt(16)+2**10"}) → ToolResult(success=True,  result=1028)
        run({"expression": ""})               → ToolResult(success=False, error="表达式为空")
        run({"expression": "__import__('os').system('echo hi')"})
                                              → ToolResult(success=False, error="计算失败: 不支持的函数: None")
    """

    # 【能力声明】纯计算、无副作用、只读，Harness 直接放行
    capability = {  # 类属性：Harness 治理时读这张表
        "risk": "low",  # 风险等级：低（纯计算）
        "side_effect": False,  # 无副作用（不发通知、不写库）
        "sandbox": "none",          # 本身已是 AST 白名单求值，无需额外隔离
        "description": "只读计算，无外部副作用",  # 给人看的描述
    }

    def __init__(self):  # 构造：注册工具名/描述/参数 schema
        super().__init__(  # 调父类 BaseTool 构造，固化三要素
            name="calculator",   # 工具名；Permission 里对应 tool:calculator
            description="数学计算器，支持四则运算、幂运算、三角函数等。输入数学表达式字符串，返回计算结果。",  # 给 LLM 看的用途说明
            parameters={  # JSON Schema 形参说明书
                "properties": {  # 形参字段集
                    "expression": {  # 唯一入参：表达式字符串
                        "type": "string",  # 类型字符串
                        "description": "数学表达式，如 '2+3*4', 'sqrt(16)', 'sin(pi/2)'",  # 用法说明
                    }   # 唯一的入参；传非字符串会在 ast.parse 处报错
                },
                "required": ["expression"],   # 必填；空串或纯空白会在 execute 里先被拦掉
            },
        )

    # 【什么】递归遍历 AST 求值
    # 【作用】逐节点对照白名单，白名单外的节点一律抛 ValueError
    def _eval_node(self, node):  # 递归求值 AST 节点：每种节点类型一个分支
        """递归安全求值 AST 节点。"""
        if isinstance(node, ast.Expression):  # 顶层 Expression 节点（mode="eval" 时的根）
            return self._eval_node(node.body)  # 剥掉外壳，递归求 body
        if isinstance(node, ast.Constant):  # 常量节点（数字/字符串等字面量）
            if isinstance(node.value, (int, float)):  # 只放行 int/float 数字
                return node.value  # 直接返回数字值
            raise ValueError(f"不支持的常量: {node.value}")  # 其他常量（如字符串）拒绝
        if isinstance(node, ast.BinOp):  # 二元运算节点（a + b 这种）
            op = _SAFE_OPS.get(type(node.op))  # 查白名单拿对应运算函数
            if not op:  # 白名单外（如位运算、比较运算）
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")  # 报错拒绝
            return op(self._eval_node(node.left), self._eval_node(node.right))  # 先递归左右子树，再套运算函数
        if isinstance(node, ast.UnaryOp):  # 一元运算节点（-a / +a）
            op = _SAFE_OPS.get(type(node.op))  # 查白名单
            if not op:  # 不在白名单
                raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")  # 报错拒绝
            return op(self._eval_node(node.operand))  # 先递归操作数，再套一元运算
        if isinstance(node, ast.Call):  # 函数调用节点（如 sin(x)）
            func_name = node.func.id if isinstance(node.func, ast.Name) else None  # 只放行简单名调用；属性调用（如 os.system）的 func 是 Attribute，func_name 为 None
            func = _SAFE_FUNCS.get(func_name) if func_name else None   # 属性调用时 func_name 是 None
            if not func:  # 函数不在白名单
                raise ValueError(f"不支持的函数: {func_name}")  # 报错拒绝（os.system 走到这里）
            args = [self._eval_node(a) for a in node.args]  # 先递归求实参列表
            return func(*args)  # 把实参打散传给白名单函数
        if isinstance(node, ast.Name):  # 裸名字节点（如 pi / e）
            val = _SAFE_FUNCS.get(node.id)  # 去白名单查（pi/e 在白名单里）
            if val is None:  # 未知名（如 x、os）
                raise ValueError(f"未知变量: {node.id}")  # 报错拒绝
            return val  # 返回白名单常量值
        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")  # 其他节点类型一律拒绝（如 Attribute、Subscript）

    # 【什么】解析并执行数学表达式
    # 【作用】ast.parse(mode="eval") 后交给 _eval_node，异常一律转成失败 ToolResult
    def execute(self, expression: str, **kwargs) -> ToolResult:  # 业务入口：expression=表达式字符串，**kwargs 吞掉额外参数
        """安全执行数学表达式。"""
        expr = expression.strip()  # 去首尾空白
        if not expr:  # 空串或纯空白
            return ToolResult(success=False, error="表达式为空")  # 提前返回失败，不进 AST 解析
        try:  # 解析与求值主路径
            tree = ast.parse(expr, mode="eval")  # 把字符串解析成 AST（只允许单个表达式，不允许语句）
            result = self._eval_node(tree)  # 递归白名单求值
            if isinstance(result, float):  # 结果是浮点数
                if result.is_integer():  # 浮点值恰为整数（如 4.0）
                    result = int(result)  # 转成 int（让 4.0 显示成 4，更整洁）
            return ToolResult(success=True, result=result)  # 包成成功 ToolResult
        except Exception as e:  # 一切解析/求值异常（含 ValueError/ZeroDivisionError/SyntaxError）
            return ToolResult(success=False, error=f"计算失败: {e}")  # 包成失败 ToolResult，把错误原因带回
