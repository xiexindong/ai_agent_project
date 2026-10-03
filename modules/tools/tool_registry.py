"""
工具注册器模块
==============
【一句话定位】Agent 调用一切工具的唯一入口。
覆盖边界3：工具死循环拦截、重复调用去重。

【通俗理解：Agent 世界的"统一请求层"】
    前端项目不会让每个组件各自裸发 axios，而是统一走封装好的 request()：
    统一加 token、统一拦截重复提交、统一处理响应。
    ToolRegistry 就是这个 request() ——
    本地工具（calculator/data_query，进程内直接跑）也好，
    MCP 远程工具（web_search/get_current_time，子进程里跑）也罢，
    注册进来之后都走同一套 execute_tool() 流水线，
    上层 ReActAgent 只管"下指令"，完全不感知工具的物理位置。

【为什么要设防死循环闸门】
    LLM 靠逐轮"生成文本"决定下一步，可能反复生成同一个调用 ——
    就像用户疯狂连点提交按钮。所以执行前设三道闸门：
        闸门1  轮次上限     graph 层 recursion_limit=30 兜底（在别的文件）
        闸门2  单工具次数   本文件 _max_calls_per_tool=5
        闸门3  相同参数去重 本文件 _call_records 记录"工具名+参数指纹"

【装配 Harness 后：执行前后多出一条"治理流水线"（类 NestJS 请求生命周期）】
    execute_tool 在真正 run() 前后各加了几层，类比 NestJS 一目了然：
        能力声明 capability_of  ≈ 路由元数据（这个工具有多危险、在哪跑）
        → 策略裁决 before_call  ≈ Guard（禁止/需审批）+ Pipe（参数处理）
          + 中间件（参数/权限/业务范围/限流/预算）+ 人工审批（高风险）
        → 沙箱执行 execute       ≈ Controller handler（按能力声明选工位）
        → 结果处理 after_call    ≈ Interceptor 后置（裁剪脱敏 + token 记账 + 审计）
    未装配 harness 时整条管线全部跳过，行为与改造前完全一致。
"""
from typing import Any, Dict, List, Optional
from collections import defaultdict
from modules.common.logger import get_logger
from modules.common.exceptions import ToolException
from modules.harness.models import ToolCallRequest
from modules.tools.base_tool import BaseTool, ToolResult

_logger = get_logger("registry")


# 【什么】工具注册与调度中心（本地 + MCP 双模式）
# 【作用】工具调用的唯一入口，本文件自带两道防死循环闸门（限次 + 同参去重）
class ToolRegistry:
    """双模式工具注册器（本地 + MCP 远程）。

    内部维护 4 本"账本"（类比前端的一个 store 对象，key 不同用途不同）：
        _tools         注册表  工具名 → 工具实例（登记了才能被调度）
        _call_records  去重账  会话ID → 已执行过的"工具名+参数指纹"集合
        _call_counts   限次账  工具名 → 累计调用次数（超上限判死循环）
        _result_cache  缓存账  会话ID → {去重键: 上次成功的 ToolResult}
    """

    def __init__(self, harness=None):
        # ------------------------------------------------------------------
        # 【harness 是什么】工具运行时框架（modules.harness.ToolHarness）
        #   负责能力登记、调用前校验、人工审批、沙箱执行、结果过滤与审计。
        #   为 None 时本类退化为"注册 + 去重 + 限次"的基础版，行为不变。
        #   类比 NestJS：不装 harness = 只有 Controller；装了 = Guard/Pipe/
        #   Interceptor 整套管线生效。
        # ------------------------------------------------------------------
        self.harness = harness                                                 # 工具 Harness；None = 不启用治理
        # 【_tools 长什么样】注册 CalculatorTool + DataQueryTool 之后：
        #   {'calculator': <CalculatorTool object>, 'data_query': <DataQueryTool object>}
        #   取值：_tools['calculator'] → <CalculatorTool object>（.name='calculator', .tool_type='local'）
        #         _tools['calc']       → KeyError，所以对外统一走 get_tool()（未命中返回 None）
        self._tools: Dict[str, BaseTool] = {}                                  # 工具名 → 工具实例
        # 【_call_records 长什么样】去重键 = f"{工具名}:{参数指纹}"（指纹=参数项排序后的列表）
        #   s1 调 calculator('3*7')   → {'s1': {"calculator:[('expression', '3*7')]"}}
        #   s1 再调 calculator('1+1') → {'s1': {"calculator:[('expression', '3*7')]",
        #                                        "calculator:[('expression', '1+1')]"}}
        #   s2 调 calculator('3*7')   → 换会话不算重复，s2 自己另起一本账
        self._call_records: Dict[str, set] = defaultdict(set)                  # 会话ID → {"工具名:参数指纹"}，去重用
        self._call_counts: Dict[str, int] = defaultdict[str, int](int)         # 工具名 → 累计调用次数，限频用
        # ------------------------------------------------------------------
        # 【结果缓存】会话ID → {去重键: 上次成功的 ToolResult}
        #   只缓存成功结果，用途是「同参数再次调用」时把上次答案直接回传。
        #
        #   【为什么要这个？】去重如果只回一句"拒绝重复执行"，模型拿到的是失败
        #   信号，会换个说法继续调同一个工具 —— 实测 ReAct 跑满 10 轮、同一个
        #   计算器重复调 9 次，单次对话被拖到 13 秒（约 10 倍 token）。
        #
        #   【对比例子】
        #     ✗ 只有去重：第二次 calculator('3*7') → error="拒绝重复执行"
        #                 模型：工具坏了？换个说法再试……（死循环继续烧 token）
        #     ✓ 有缓存：  第二次 calculator('3*7') → success=True, result='21'
        #                 模型：答案拿到了，可以收尾了
        # ------------------------------------------------------------------
        self._result_cache: Dict[str, dict] = defaultdict(dict)   # {会话ID: {去重键: 上次成功的 ToolResult}}
        self._max_calls_per_tool = 5                              # 单工具累计调用上限 5 次，超出判为死循环

    # 【什么】把工具实例放进注册表
    # 【作用】登记后工具才能被调度；同名工具后者覆盖前者，只记 warning 不报错
    #        （启动装配阶段重复注册不该中断流程，类似配置重复 key 以最后一次为准）
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
        # ① 能力登记：风险等级/副作用/沙箱/权限在这里被 Harness 记成台账，
        #   调用时直接查表，不用每次重新解析工具声明
        cap = self.harness.register(tool) if self.harness else None
        _logger.info(f"工具注册: {tool.name} ({tool.tool_type})"
                     + (f" | 风险={cap.risk.value}" if cap else ""))

    # 【什么】批量注册工具
    # 【作用】启动阶段一次性装配，逐个走 register()，重名策略与单个注册一致
    def register_batch(self, tools: List[BaseTool]):
        """
        批量注册工具。

        Args:
            tools: 工具实例列表，逐个调用 register()，重名策略与单个注册一致。
        """
        for tool in tools:
            self.register(tool)

    # 【什么】按名称取工具实例
    # 【作用】执行前的存在性检查入口；取不到返回 None，由上层转成错误 ToolResult
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        按名称获取工具实例。

        Args:
            name: 工具名（即 Function Calling 返回的 tool_calls[].function.name）。

        Returns:
            工具实例；不存在时返回 None，由调用方决定如何提示 LLM。
        """
        return self._tools.get(name)

    # 【什么】列出全部已注册工具
    # 【作用】给全量调用或管理接口遍历用，顺序即注册顺序
    def list_tools(self) -> List[BaseTool]:
        """
        列出所有已注册工具。

        Returns:
            工具实例列表（基于 dict.values()，注册顺序即遍历顺序）。
        """
        return list[BaseTool](self._tools.values())

    # 【什么】生成发给模型的工具 schema 列表
    # 【作用】权限的"源头过滤"：模型看不到的工具就调不到（allowed 优先于 user_level）
    #        类比路由守卫：不是"进了接口再拦"，而是压根不把入口暴露出去
    def get_openai_tools_schema(self, user_level: str = "normal", allowed: Optional[list] = None) -> list:
        """
        生成 OpenAI Function Calling 格式的工具说明书（JSON Schema 列表）。

        这份 schema 会作为 tools 参数发给 DeepSeek，LLM 据此决定"能调哪些工具、
        每个工具要什么参数"。权限在此处做源头过滤：模型看不到的工具，就不可能调用。

        Args:
            user_level: 用户权限等级，"normal" / "advanced" / "admin"。
            allowed:    本轮工具白名单（工具实例 list 或工具名 list，两种都收）；
                        为 None 时回退到按 user_level 过滤。

        Returns:
            可直接传给 OpenAI 兼容接口的 tools 数组（list[dict]）。

        【输入输出对照】注册表已注册 calculator、data_query：
            allowed=[<CalculatorTool>, <DataQueryTool>]  → ['calculator', 'data_query']
            allowed=[<CalculatorTool>]                   → ['calculator']
            allowed=['data_query']                       → ['data_query']
            allowed=[<CalculatorTool>, 'data_query']     → ['calculator', 'data_query']
            allowed=['不存在']                            → []
            allowed=[]                                   → []   # 空白名单=全屏蔽，非不过滤
            allowed=None                                 → ['calculator', 'data_query']
        """
        # 归一化成名字集合：[<CalculatorTool>, 'data_query'] → {'calculator', 'data_query'}
        # （统一成名字比较，实例和字符串两种白名单写法都能用）
        allowed_names = None
        if allowed is not None:
            allowed_names = {t if isinstance(t, str) else getattr(t, "name", "") for t in allowed}
        schemas = []
        for tool in self._tools.values():
            if allowed_names is not None:
                if tool.name not in allowed_names:
                    _logger.debug(f"工具未授权，不暴露给模型: {tool.name}")
                    continue
            elif user_level == "normal" and tool.tool_type == "mcp":
                continue                     # 兜底规则：普通用户默认见不到 MCP 远程工具
            schemas.append(tool.to_openai_format())
        return schemas

    # 【什么】单工具调用次数上限
    # 【作用】装配 Harness 时用它的预算配置，否则沿用本文件的默认值 5
    def _tool_call_limit(self) -> int:
        if self.harness is not None:
            spec = getattr(self.harness.budgets, "spec", None)
            return int(getattr(spec, "max_calls_per_tool", 0) or self._max_calls_per_tool)
        return self._max_calls_per_tool

    # ==================================================================
    # 【什么】工具调度的核心流水线（本文件的"main 函数"）
    # 【作用】校验 → 治理裁决 → 限次 → 去重 → 执行 → 过滤 → 记账
    #        每一步都返回 ToolResult，全程不抛异常（上层不用写 try/catch）
    #
    # 【执行顺序（编号与下方代码一一对应，也呼应 harness.py 文件头的 12 步）】
    #   ① 查注册表    工具不存在 → 直接回失败
    #   ② 能力声明    capability_of(tool) → cap（工具"体检表"，见下方字段说明）
    #   ③ 治理裁决    before_call(req)：策略 + 中间件 + 人工审批
    #                 拦下 → record_intercepted() 留痕，中文原因回给模型
    #   ④ registry 自己的两本账：
    #                 ④-1 单工具限次（闸门2）
    #                 ④-2 同参去重（闸门3，命中成功缓存直接回传，省一轮 LLM）
    #   ⑤ 组装实参    模型参数 + 服务端私有参数（context 后写入，模型盖不住）
    #   ⑥ 执行        harness.execute(...) 按沙箱声明选工位；未装配就是 tool.run()
    #   ⑦ 记账        计数 +1、去重键入账（真正执行后才记：被拦/去重不占配额）
    #   ⑧ 结果处理    after_call(...) 裁剪脱敏 + 审计；成功结果进缓存
    #
    # 【两本账并行：防打转 vs 管预算】
    #   registry 的 _call_counts / _call_records 管"同一个会话别原地打转"
    #   harness  的 RunBudget 管"这一轮总共能花多少"（次数 / token / 墙钟）
    #   都过才放行，互不替代。
    # ==================================================================
    def execute_tool(
        self,
        name: str,
        params: dict,
        session_id: str = "",
        context: Optional[dict] = None,
        principal: Any = None,
        user_id: str = "",
        iteration: int = 0,
    ) -> ToolResult:
        """
        执行工具调用（工具调度的核心入口）。不通过时返回失败的 ToolResult，不向上抛异常。

        Args:
            name:       LLM 通过 Function Calling 指定的工具名。
            params:     LLM 生成的调用参数（JSON 已解析为 dict）。
            session_id: 会话ID，用于会话级去重隔离；为空时跳过去重。
            context:    服务端注入的私有参数（_data_scope/_dept_code/_user_code/_mask_level），
                        不出现在 schema 里，模型无法伪造。
            principal:  权限载体，Harness 用它做权限/业务范围/脱敏/审计。
            user_id:    用户标识，仅用于审计与限流。
            iteration:  第几轮 ReAct，仅用于审计。

        Returns:
            ToolResult: 统一结果对象（success/result/error/duration_ms/meta）。

        【输入输出对照：8 个"剧本"看懂返回值】
        每条剧本 = 前提 → 调用 → 返回，并标注拦在第几步（对应上方流程编号①~⑧）。

        ── 第一组：正常路径 ──
        前提：本会话还没调过 calculator
            calculator({'expression': '3*7'}) 首次调用
            → success=True,  result="21"                （⑥ 真跑了一遍计算）

        前提：上一条刚跑完
            calculator({'expression': '3*7'}) 同会话再调一次
            → success=True,  result="21"                （④-2 不真跑，回放缓存，duration_ms=0）

        ── 第二组：registry 自带闸门拦下（开箱即有，不装 harness 也生效）──
        前提：模型幻觉出注册表里没有的名字              （拦在①）
            调用 "calc"
            → success=False, error="工具不存在: calc"

        前提：calculator 本会话已真跑 5 次，这是第 6 次 （拦在④-1）
            → success=False, error="工具 calculator 调用次数超限(5)，疑似死循环，已拦截"

        ── 第三组：Harness 治理拦下（都拦在③；未装配 harness 时这几条不会发生）──
        前提：drop_table 被策略配成 critical 级 = 禁止执行
            → success=False, error="工具 drop_table 风险等级为 critical，已配置为禁止执行"

        前提：当前登录身份没有 data_query 要求的权限码
            → success=False, error="当前身份无权调用工具 data_query"

        前提：财务身份(D_FIN) 却想查 HR 部门(D_HR) 的数据，超出业务范围
            → success=False, error="dept_code=D_HR 超出本部门范围(D_FIN)"

        前提：send_notice 是高风险工具须人工点"批准"，这次没批就取消
            → success=False, error="工具 send_notice 未获人工批准，已取消执行"
        """
        # ① 查注册表。name = LLM 回的 tool_calls[].function.name（纯字符串）：
        #   "calculator" / "data_query" → 命中，拿到工具实例
        #   "calc"                      → 模型幻觉出的名字，下一行直接拦掉
        #   get_tool("calculator") → <CalculatorTool object>（有 .name / .run() / .capability）
        #   get_tool("calc")       → None
        tool: BaseTool | None = self.get_tool(name)
        if not tool:
            return ToolResult(success=False, error=f"工具不存在: {name}")

        # ② 能力声明：cap = 该工具的"体检表"（取值三级：工具自声明 → 配置覆盖 →
        #   策略兜底，这里拿到的已是最终值）。字段速查：
        #     risk         危险等级 low < medium < high < critical
        #     side_effect  是否对外动真格（发通知/删数据）；只记录，不决定是否人工批准
        #     sandbox      在哪跑：none 进程内 / subprocess 子进程 / docker 容器
        #     data_scopes  会碰哪些业务数据，如 ('users','orders','inventory')
        #     permissions  需要的权限码；max_result_chars 结果最长字数；timeout_seconds 超时秒数
        #
        #   三个内置工具的表：
        #     calculator  → risk=low,  sandbox=none         纯计算直接跑
        #     data_query  → risk=low,  data_scopes=(users,orders,inventory)  只读但要查数据范围
        #     send_notice → risk=high, side_effect=True, sandbox=subprocess  发出去收不回，须人批准
        #
        #   框架拿这张表只干三件事：
        #     看 risk        → 直接跑 / 等人批准 / 禁止执行
        #     看 data_scopes → 查参数有没有越界（财务的人去查 HR 的部门）
        #     看 sandbox     → 决定在进程内 / 子进程 / 容器里跑
        #
        #   cap 实际长相（calculator 实测值）：
        #   ToolCapability(name='calculator', risk=<RiskLevel.LOW: 'low'>, side_effect=False,
        #                  sandbox='none', requires_approval=False, permissions=(), data_scopes=(),
        #                  max_result_chars=0, timeout_seconds=0.0, description='只读计算，无外部副作用')
        #   harness 未装配 → cap=None，③ 治理段整段跳过（改造前的原行为）
        cap = self.harness.capability_of(tool) if self.harness is not None else None
        # req = 本次调用的"档案袋"：工具名、参数、会话/用户、能力表打包给 Harness
        req = ToolCallRequest(
            tool_name=name, arguments=dict[str, Any](params or {}), session_id=session_id, user_id=user_id,
            principal=principal, capability=cap, iteration=iteration, context=dict(context or {}),
        )
        # ③ 治理裁决（Harness 未装配时整段跳过）：
        #   before_call = Guard（策略）+ Pipe（参数）+ 中间件 + 人工审批 的总入口。
        #   【为什么拦截也要审计】越权尝试正是安全审计最该看到的东西
        if self.harness is not None:
            verdict = self.harness.before_call(req)
            if not verdict.allowed:
                self.harness.record_intercepted(req, cap, verdict.reason, verdict.code)
                return ToolResult(
                    success=False, error=verdict.reason,
                    meta={"code": verdict.code, "risk": cap.risk.value if cap else ""},
                )
            params = verdict.arguments          # 人工批准时可能改过参数（modify）

        # ④-1 单工具限次（闸门2）：累计超上限后不再执行，
        #   回失败结果让模型看到"此路不通"，主动换思路收尾
        limit = self._tool_call_limit()
        if self._call_counts[name] >= limit:
            return ToolResult(
                success=False,
                error=f"工具 {name} 调用次数超限({limit})，疑似死循环，已拦截",
            )

        # ④-2 同参去重（闸门3）：去重键 = "工具名:参数指纹"
        #   命中且上次成功 → 原样回传缓存，不再执行第二次。
        #   【为什么回传而不是报错】报错会让模型以为"工具坏了"继续重试；
        #   把答案原样返回，模型才有依据收尾 —— 既省一轮 LLM，也省 10 倍 token。
        #   命中但上次失败 → 拒绝重复执行，让模型换参数再试。
        param_key = self._hash_params(params)
        dedup_key = f"{name}:{param_key}"
        if dedup_key in self._call_records.get(session_id, set()):
            cached = self._result_cache.get(session_id, {}).get(dedup_key)
            if cached is not None and cached.success:
                _logger.info(f"工具重复调用命中缓存，直接回传上次结果: {dedup_key}")
                return ToolResult(success=True, result=cached.result, duration_ms=0)
            _logger.warning(f"工具重复调用拦截: {dedup_key} session={session_id}")
            return ToolResult(
                success=False,
                error=f"工具 {name} 已用相同参数调用过，拒绝重复执行（防死循环）",
            )

        # ⑤ 组装实参：context 是服务端注入的私有参数（_data_scope/_dept_code 等），
        #   后写入所以能盖住模型给的值 —— 模型伪造不了数据范围
        call_params = dict(params or {})
        if context:
            call_params.update(context)         # 私有参数后写入，模型给的参数盖不住它

        # ⑥ 执行：Harness 按 cap.sandbox 选工位（进程内/子进程/容器）；
        #   未装配 harness 时就是原来的 tool.run()
        if self.harness is not None:
            result = self.harness.execute(tool, call_params, cap)
        else:
            result = tool.run(call_params)

        # ⑦ 记账：只在真正发起执行后才计数 —— 被拦截/去重的调用不占配额
        self._call_counts[name] += 1   # 只在真正发起执行后计数：失败调用不占配额
        if session_id:
            self._call_records[session_id].add(dedup_key)

        # ⑧ 结果过滤（裁剪 + 脱敏）与审计；过滤后再缓存，
        #   保证重复调用回传的是同一份文本（模型前后看到的一致）
        if self.harness is not None:
            result = self.harness.after_call(result, req, cap)
        if session_id and result.success:
            # 【为什么只缓存成功的】失败结果被回传会让模型误以为"答案就是这样"，
            #   失败必须让它看到错误，才有机会改参数重试。
            self._result_cache[session_id][dedup_key] = result

        return result

    # 【什么】生成参数指纹
    # 【作用】排序后序列化，保证传参顺序不同也能判定为同一次调用
    #   类比 JS 手写 memoize：缓存键常用 JSON.stringify(args)，但 stringify
    #   依赖 key 顺序（{"a":1,"b":2} 与 {"b":2,"a":1} 串不同）；
    #   这里先 sorted() 再 str()，与顺序无关，指纹稳定。
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
        return str(sorted(params.items()))

    # 【什么】清空某会话的去重账本与结果缓存，并重置全局调用计数
    # 【作用】新一轮提问前调用。不清的后果：
    #   去重账不清 → 这一轮调相同参数会被上一轮的记录误判成"重复"
    #   缓存不清   → 这一轮拿到的是上一轮的旧答案
    #   计数不清   → 上一轮的配额被带到这一轮，明明没调几次就被拦
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
        # 结果缓存必须与去重账本一起清，否则新一轮拿到的是上一轮的旧答案
        if session_id in self._result_cache:
            del self._result_cache[session_id]
        self._call_counts.clear()
        # 新一轮提问 = 新预算：把上一轮的预算账本一并销掉，避免配额被上轮吃掉
        if self.harness is not None:
            self.harness.end_run(session_id)

    # 【什么】返回两个内置本地工具的新实例
    # 【作用】函数内延迟 import 兜住循环依赖，装配默认工具集时直接调用
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
