"""
工具注册器模块
==============
统一管理本地工具 + MCP 远程工具的注册、查找、调度。
覆盖边界3（工具死循环拦截、重复调用去重）。

核心定位：
    它是 Agent 调用一切工具的唯一入口（类似前端的统一 API 请求层）。
    无论是进程内的本地工具（calculator/data_query），还是 MCP 子进程里的
    远程工具（web_search/get_current_time），注册到这里之后都走同一套
    execute_tool() 流程，上层 ReActAgent 完全不感知工具的物理位置。

三道安全闸门（防止 LLM 陷入"反复调同一个工具"的死循环）：
    1. 轮次上限     —— graph 层 recursion_limit=30 兜底
    2. 单工具次数   —— 本文件 _max_calls_per_tool=5
    3. 相同参数去重 —— 本文件 _call_records 记录"工具名+参数指纹"
"""
from typing import Dict, List, Optional
from collections import defaultdict
from modules.common.logger import get_logger
from modules.common.exceptions import ToolException
from modules.tools.base_tool import BaseTool, ToolResult

_logger = get_logger("registry")


class ToolRegistry:
    """
    双模式工具注册器（本地 + MCP 远程）。

    内部维护 3 张"账本"：
        _tools         工具名 → 工具实例 的注册表（核心字典）
        _call_records  会话ID → 已执行的"工具名+参数指纹"集合（去重用）
        _call_counts   工具名 → 累计调用次数（限频用）
    """

    def __init__(self):
        # 工具注册表：name 作为唯一键，本地工具和 MCPToolAdapter 都放这里
        self._tools: Dict[str, BaseTool] = {}
        # 调用去重：记录每个会话已执行过的「工具+参数」（边界3：防死循环）
        # 用 defaultdict(set) 类比 JS 的 { sessionId: new Set() }，首次访问自动建空集合
        self._call_records: Dict[str, set] = defaultdict(set)
        # 每个工具的累计调用次数（当前轮次内），超限即拦截
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._max_calls_per_tool = 5  # 单工具最大调用次数

    def register(self, tool: BaseTool):
        """
        注册单个工具（本地工具或 MCP 适配器均可）。

        Args:
            tool: 任何实现了 BaseTool 接口的工具实例（需有 name/tool_type/run()）。

        Note:
            同名工具重复注册时后者覆盖前者，仅记 warning 不报错，
            保证启动装配阶段重复注册不会中断流程。
        """
        if tool.name in self._tools:
            _logger.warning(f"工具重复注册，覆盖: {tool.name}")
        self._tools[tool.name] = tool
        _logger.info(f"工具注册: {tool.name} ({tool.tool_type})")

    def register_batch(self, tools: List[BaseTool]):
        """
        批量注册工具。

        Args:
            tools: 工具实例列表，逐个调用 register()，重名策略与单个注册一致。
        """
        for tool in tools:
            self.register(tool)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        按名称获取工具实例。

        Args:
            name: 工具名（即 Function Calling 返回的 tool_calls[].function.name）。

        Returns:
            工具实例；不存在时返回 None，由调用方决定如何提示 LLM。
        """
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """
        列出所有已注册工具。

        Returns:
            工具实例列表（基于 dict.values()，注册顺序即遍历顺序）。
        """
        return list(self._tools.values())

    def get_openai_tools_schema(self, user_level: str = "normal") -> list:
        """
        生成 OpenAI Function Calling 格式的工具说明书（JSON Schema 列表）。

        这份 schema 会作为 tools 参数发给 DeepSeek，LLM 据此决定"能调哪些工具、
        每个工具要什么参数"。权限在此处做源头过滤：普通用户拿到的说明书里
        根本没有 MCP 工具，模型自然无法发起越权调用。

        Args:
            user_level: 用户权限等级，"normal"（普通）/ "advanced"（高级）/ "admin"。

        Returns:
            可直接传给 OpenAI 兼容接口的 tools 数组（list[dict]）。
        """
        schemas = []
        for tool in self._tools.values():
            # 权限检查：普通用户跳过 MCP 远程工具（仅保留本地基础工具）
            if user_level == "normal" and tool.tool_type == "mcp":
                continue
            schemas.append(tool.to_openai_format())
        return schemas

    def execute_tool(
        self,
        name: str,
        params: dict,
        session_id: str = "",
    ) -> ToolResult:
        """
        执行工具调用（工具调度的核心入口）。

        【处理流水线】每一步不通过都直接返回失败的 ToolResult，绝不向上抛异常
            工具存在性检查 → 次数上限检查 → 相同参数去重 → 真正执行 → 记账

        【四种场景的 ToolResult 对比】
            ✓ 正常执行（首次调 calculator(3*7)）：
                ToolResult(success=True, result="21", duration_ms=12)

            ✗ 工具不存在（LLM 幻觉出工具名"calc"）：
                ToolResult(success=False, error="工具不存在: calc")

            ✗ 单工具次数超限（calculator 已调 5 次还在调）：
                ToolResult(success=False, error="工具 calculator 调用次数超限(5)，疑似死循环，已拦截")

            ✗ 相同参数去重（同会话再次 calculator(3*7)）：
                ToolResult(success=False, error="工具 calculator 已用相同参数调用过，拒绝重复执行（防死循环）")

        被闸门拦截时 success=False 的 error 会回喂给 LLM，
        模型读到"已超限/重复调用"后通常会转为直接作答。

        Args:
            name:       LLM 通过 Function Calling 指定的工具名。
            params:     LLM 生成的调用参数（JSON 已解析为 dict）。
            session_id: 当前会话ID，用于会话级去重隔离；为空时跳过去重记录。

        Returns:
            ToolResult: 统一结果对象（success/result/error/duration_ms）。

        Note:
            去重账本只在"成功发起执行后"才记录；失败的调用不计数，
            避免一次参数错误把工具的调用配额浪费掉。
        """
        # ① 存在性检查：LLM 可能幻觉出一个未注册的工具名
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(success=False, error=f"工具不存在: {name}")

        # ② 闸门一：单工具调用次数限制（超过 5 次判定为疑似死循环）
        if self._call_counts[name] >= self._max_calls_per_tool:
            return ToolResult(
                success=False,
                error=f"工具 {name} 调用次数超限({self._max_calls_per_tool})，疑似死循环，已拦截",
            )

        # ③ 闸门二：相同参数去重
        # 生成"工具名+参数指纹"，同一会话内完全相同的调用只允许执行一次
        param_key = self._hash_params(params)
        dedup_key = f"{name}:{param_key}"
        if dedup_key in self._call_records.get(session_id, set()):
            _logger.warning(f"工具重复调用拦截: {dedup_key} session={session_id}")
            return ToolResult(
                success=False,
                error=f"工具 {name} 已用相同参数调用过，拒绝重复执行（防死循环）",
            )

        # ④ 真正执行：BaseTool.run() 内部还会做必填参数校验并统一捕获异常
        result = tool.run(params)

        # ⑤ 记账：累计次数 + 记录参数指纹（仅在执行后记录，失败不占配额）
        self._call_counts[name] += 1
        if session_id:
            self._call_records[session_id].add(dedup_key)

        return result

    @staticmethod
    def _hash_params(params: dict) -> str:
        """
        生成参数指纹（用于去重时判断两次调用的参数是否完全一致）。

        Args:
            params: 工具参数字典，如 {"expression": "1+2", "precision": 2}。

        Returns:
            参数字符串指纹。先对 key 排序再序列化，保证
            {"a":1,"b":2} 与 {"b":2,"a":1} 得到相同指纹（与传参顺序无关）。
            params 为空时返回空字符串。
        """
        if not params:
            return ""
        # sorted() 按 key 排序，使指纹不受字典书写顺序影响
        return str(sorted(params.items()))

    def reset_session(self, session_id: str):
        """
        重置调用记录（每轮新提问或开启新会话前调用）。

        清除两本账：该会话的参数去重集合（允许新一轮再次调用相同工具）、
        以及全局单工具次数计数。次数计数不按会话隔离，因此这里统一 clear。

        Args:
            session_id: 要清除去重记录的会话ID。
        """
        if session_id in self._call_records:
            del self._call_records[session_id]
        self._call_counts.clear()

    def get_default_tools(self) -> List[BaseTool]:
        """
        获取内置默认工具集（进程内本地工具）。

        采用函数内延迟 import，避免本模块与具体工具类之间产生模块加载循环依赖。

        Returns:
            [CalculatorTool, DataQueryTool] 两个新实例：
            CalculatorTool 安全计算器（AST 白名单求值）、
            DataQueryTool 模拟业务数据查询（用户/订单/库存）。
        """
        from modules.tools.calculator import CalculatorTool
        from modules.tools.data_query import DataQueryTool
        return [CalculatorTool(), DataQueryTool()]
