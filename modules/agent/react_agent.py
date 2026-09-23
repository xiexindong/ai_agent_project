"""
ReAct Agent 模块
================
标准 ReAct 推理循环：Reason思考 → Action执行 → Observation观察。
整合 RAG 知识库、向量记忆、本地工具、MCP远程工具全链路。
覆盖边界2（输出校验）、边界3（死循环拦截）、边界6（Prompt防护）。
"""
# 【什么】标准库 + 校验器 + LLM 客户端 + 提示词注册表
# 【作用】清洗输入、检测注入、修 JSON、取提示词模板，都是 ReAct 每轮要用的东西
import time
from typing import Optional
from modules.common.logger import get_logger
from modules.common.exceptions import LLMException, ToolException, handle_exception
from modules.common.validator import sanitize_input, check_prompt_injection, repair_json
from modules.agent.llm import LLMClient
from modules.agent.state import AgentState
from modules.prompt import PromptBuilder, get_registry

_logger = get_logger("react")


# 【什么】ReAct 推理引擎（Reason-Action-Observation 循环）
# 【作用】一轮一轮地「思考 → 调工具 → 看结果」，直到模型决定直接作答
class ReActAgent:
    """ReAct 推理引擎（Reason-Action-Observation 循环）。

    【提示词从哪来】不再写死在类里，全部取自 modules/prompt 的注册表：
        react.role(v1)    角色 + 核心规则 + 回答要求（变量 max_iter）
        react.reason(v1)  8 个上下文槽位（query/技能/权限/RAG/记忆/工具历史/观察/工具清单）
        react 策略块      Reason→Action→Observation 流程，由 strategies.py 生成
    三者在 PromptBuilder 里拼成 system + user 两条消息。

    【为什么要迁走而不是留在类里】
        留在类里，改一句话要翻本文件，且"换个问法"只能改正文；
        迁走之后：模板带版本可回滚，问法由 strategy 参数切换，
        上下文超长由 builder 按预算裁剪（权限 > 工具 > 观察 > 技能 > RAG > 记忆）。

    【旧文案还在吗】在。react.system(v1) 的正文与迁移前的 SYSTEM_PROMPT 逐字
        一致，react.reason(v1) 与旧 REASON_PROMPT 逐字一致，需要原文时按名取用即可。
    """

    # 【什么】装配五个可选依赖与轮次上限
    # 【作用】依赖都可以不给：缺哪个就退化到「不做这件事」的旧行为
    def __init__(
        self,
        llm_client: LLMClient | None = None,       # 不给就自建单例客户端
        tool_registry=None,                        # 工具注册表，为 None 则无工具可用
        rag_chain=None,                            # RAG 链，为 None 则不召回知识库
        memory_manager=None,                       # 记忆管理器，为 None 则不召回记忆
        skill_registry=None,                       # 技能注册表，为 None 则不走 SOP
        router=None,                               # L0 路由层，为 None 则每轮面对全部工具
    ):
        self.llm = llm_client or LLMClient()        # LLM 客户端
        self.tool_registry = tool_registry          # 工具注册表
        self.rag = rag_chain                        # RAG 链
        self.memory = memory_manager                # 记忆管理器
        self.skill_registry = skill_registry        # 技能注册表
        # L0 路由层（可选）：没有它时退化为「全部工具」，行为与改造前一致
        self.router = router
        from config import get_settings
        s = get_settings()
        self._max_iter = s.max_react_iterations     # ReAct 轮次上限，写进提示词与 state.max_iterations

        # ------------------------------------------------------------------
        # 【提示词装配在这里定型，之后每轮只重拼 user】
        #   role  : react.role(v1) 渲染结果，含「末尾附【摘要】标签」
        #           —— generate_answer 之后靠这个标签抽取摘要，不能改掉
        #   system: role + react 策略块（流程 + 工具清单），每轮都一样，
        #           所以这里算一次存成 self.system_prompt，供 generate_answer 复用
        #
        # 【max_context_tokens 管什么】
        #   上下文（权限/工具/观察/技能/RAG/记忆）合计的 token 上限，
        #   来自配置 MAX_PROMPT_CONTEXT_TOKENS（默认 3000）。
        #   超了由 builder 裁剪：先截 RAG，再丢记忆，权限边界永远最后才动。
        #   不设的话，RAG 召回一大段时 prompt 会一路膨胀到把上下文窗口吃满。
        # ------------------------------------------------------------------
        role = get_registry().render("react.role", max_iter=self._max_iter)
        self._builder = PromptBuilder(
            template_name="react.reason",                             # 8 个上下文槽位的模板
            strategy="react",                                          # Reason→Action→Observation 流程块
            role=role,                                                 # 上面渲染好的角色文案
            max_iter=self._max_iter,                                   # 写进模板变量 max_iter
            max_context_tokens=s.max_prompt_context_tokens,            # 上下文 token 预算，超了由 builder 裁剪
        )
        self.system_prompt = self._builder.system_prompt()             # 每轮都一样，算一次存着复用

    # 【什么】开跑前的准备工作：清洗、注入检测、技能、记忆、RAG
    # 【作用】把这些一次性召回塞进 state，ReAct 每轮只读不再重算
    def prepare_state(
        self,
        query: str,                         # 用户原始提问
        session_id: str,                    # 会话 ID
        user_id: str,                       # 用户标识
        user_level: str = "normal",         # 用户等级，默认 normal
        principal=None,                     # 权限载体，None = 未启用权限
    ) -> AgentState:
        """
        初始化 Agent 状态：输入清洗 → Prompt注入检测 → 记忆召回 → RAG召回。
        覆盖边界6（Prompt注入防护）。

        Args:
            principal: 权限载体（Principal）。由入口解析一次后透传，
                       决定本次请求能看哪些知识库、能调哪些工具、能看多少数据。
                       None = 未启用权限（CLI 调试场景），按不放宽也不加严处理。
        """
        if check_prompt_injection(query):
            return AgentState(
                session_id=session_id,
                user_id=user_id,
                user_query=query,
                principal=principal,
                finished=True,
                final_answer="检测到潜在的指令注入，已拦截。请使用正常方式提问。",
                error="prompt_injection_blocked",
            )

        clean_query = sanitize_input(query)
        state = AgentState(
            session_id=session_id,
            user_id=user_id,
            user_query=clean_query,
            max_iterations=self._max_iter,
            principal=principal,
        )

        if self.skill_registry:
            try:
                matched_skill, skill_ctx = self.skill_registry.build_skill_context(clean_query)
                if matched_skill:
                    state.active_skill = matched_skill.name
                    state.skill_context = skill_ctx
            except Exception as e:
                _logger.warning(f"技能匹配降级（按通用助手流程处理）: {e}")
                state.active_skill = ""
                state.skill_context = ""

        if self.memory:
            try:
                query_emb = None
                from modules.memory.embedding_service import get_embedding_service
                query_emb = get_embedding_service().embed(clean_query)
                mem_ctx = self.memory.build_context(
                    session_id, user_id, clean_query, query_emb
                )
                state.memory_context = mem_ctx["context_text"]
                state.long_term_memory = mem_ctx["long_term_memory"]
                state.short_term_memory = mem_ctx["short_term_memory"]
                state.memory_compressed = mem_ctx["compressed"]
                state.memory_low_confidence = mem_ctx.get("low_confidence", False)
                state.summary_embedding = query_emb
            except Exception as e:
                _logger.warning(f"记忆召回降级（关闭长期记忆）: {e}")
                state.memory_context = ""

        if self.rag:
            try:
                expr = self._kb_expr(principal)
                rag_ctx = self.rag.build_context(clean_query, expr=expr)
                state.rag_context = rag_ctx
                state.rag_sources = self.rag.retrieve(clean_query, expr=expr) if rag_ctx else []
                if expr:
                    _logger.info(f"知识库权限过滤已生效: {expr}")
            except Exception as e:
                _logger.warning(f"RAG召回异常降级: {e}")
                state.rag_context = ""

        if self.memory and self.memory.pg:
            try:
                stds = self.memory.pg.match_standard_answers(clean_query, limit=2)   # 最多取 2 条标准答案
                if stds:
                    std_text = "\n".join(
                        f"【标准答案{i + 1}】问：{s['question']}\n答：{s['answer']}"
                        for i, s in enumerate(stds)
                    )
                    state.rag_context = (
                        f"【人工审核确认的标准答案（优先采信）】\n{std_text}\n\n{state.rag_context}"
                    )
                    _logger.info(f"标准答案库命中 {len(stds)} 条，已注入 Few-shot")
            except Exception as e:
                _logger.debug(f"标准答案匹配失败: {e}")

        return state


    # 【什么】生成知识库过滤表达式
    # 【作用】让 RAG 只在可见库里捞；生成失败返回 None，按不限制处理
    @staticmethod
    def _kb_expr(principal) -> Optional[str]:
        """生成知识库过滤表达式；生成失败返回 None（宁可放开检索也不能让对话失败）。"""
        if principal is None:
            return None
        try:
            from modules.enterprise.rbac_service import get_rbac_service
            return get_rbac_service().kb_expr(principal)
        except Exception as e:
            _logger.warning(f"知识库权限表达式生成失败（按不限制处理）: {e}")
            return None

    # 【什么】按权限过滤本轮工具
    # 【作用】越权的工具根本不进 schema；principal 为空时原样返回，方便 CLI 调试
    @staticmethod
    def _allowed_tools(principal, tools: list) -> list:
        """按权限过滤工具；principal 为空时返回原样（CLI 调试场景）。"""
        if principal is None or not tools:
            return tools
        try:
            from modules.enterprise.rbac_service import get_rbac_service
            return get_rbac_service().allowed_tools(principal, tools)
        except Exception as e:
            _logger.warning(f"工具权限过滤失败（按最小权限处理）: {e}")
            return []

    # 【什么】把权限边界写成一段 prompt 文本
    # 【作用】模型得先知道自己的边界，才谈得上不越界
    @staticmethod
    def _permission_context(principal) -> str:
        """把权限边界写进 prompt：模型必须先知道自己的边界，才谈得上不越界。"""
        if principal is None:
            return "（未启用权限约束）"
        scope_desc = {"self": "仅本人数据", "dept": "本部门数据", "all": "全公司数据"}
        return "\n".join([
            f"- 当前用户：{principal.real_name}（工号 {principal.user_code}，{principal.dept_name}）",
            f"- 可见知识库：{', '.join(principal.kb_codes) or '全部'}（禁止引用范围外的内容）",
            f"- 业务数据范围：{scope_desc.get(principal.data_scope, principal.data_scope)}",
            f"- 敏感字段：{'已按策略脱敏，禁止追问完整值' if principal.mask_level else '未脱敏'}",
            "- 越权内容一律视为不存在：不得提及、不得暗示其存在",
        ])

    # 【什么】构造注入工具的私有参数
    # 【作用】这些下划线参数不出现在 Function Calling schema 里，只在执行时注入
    @staticmethod
    def _tool_context(principal) -> Optional[dict]:
        """
        构造注入工具的私有参数（不在 Function Calling schema 中出现）。

        _scopes     按数据类型的可见范围，由角色的 data_scope 与显式 data:xxx:yyy 权限叠加得出
        _dept_code  用于部门域过滤
        _user_code  用于本人域过滤
        _mask_level 字段脱敏强度
        """
        if principal is None:
            return None
        return {
            "_scopes": {dt: principal.scope_for(dt) for dt in ("users", "orders", "inventory")},   # 三类数据各自的可见范围
            "_dept_code": principal.dept_code,      # 部门域过滤用
            "_user_code": principal.user_code,      # 本人域过滤用
            "_mask_level": principal.mask_level,    # 字段脱敏强度：0 不脱敏 / 1 部分 / 2 强
        }

    # 【什么】工具调用审计：一次调用一条记录
    # 【作用】成功记 allow、失败记 error，带入参与耗时，供事后追责
    @staticmethod
    def _audit_tool_call(principal, tool_name: str, arguments: dict, result, session_id: str = ""):
        """工具调用审计：成功记 allow，失败记 error，并带上入参与耗时。

        session_id 随审计一起落 MySQL，用于把 PG 侧的工具流水与审计记录对齐。
        """
        if principal is None:
            return
        try:
            from modules.enterprise import audit
            audit.log_audit(
                principal.user_code,
                "tool_call",
                f"tool:{tool_name}",
                "allow" if result.success else "error",
                dept_code=principal.dept_code,                          # 按部门归集
                session_id=session_id,                                  # 与 PG 侧工具流水对齐
                detail={
                    "arguments": arguments,                             # 原始入参
                    "duration_ms": getattr(result, "duration_ms", 0),   # 耗时，缺失记 0
                    "error": (result.error if not result.success else ""),   # 失败才带错误信息
                },
            )
        except Exception as e:
            _logger.warning(f"工具审计写入失败: {e}")

    # 【什么】ReAct 的 Reason 步骤：让模型决定调工具还是直接答
    # 【作用】一次调用改写 need_tool_call / action / final_answer，决定图的下一跳
    def reason_step(self, state: AgentState) -> AgentState:
        """
        ReAct Reason 步骤：让 LLM 看 query + 历史 + 工具清单，决定"调工具还是直接答"。

        【一次调用可能改写 state 的 4 个字段】
            reasoning      : LLM 的思考过程文本（debug 用）
            need_tool_call : True=要调工具, False=直接答
            action         : 当要调工具时，存 {name, arguments, id}
            final_answer   : 当不调工具时，直接填最终答案

        【两种决策结果对比】
            ✓ 调工具（用户问"3*7"，iteration=1，<max=10）:
                state.need_tool_call = True
                state.action = {"name": "calculator", "arguments": {"expression":"3*7"}, "id":"call_abc"}
                state.final_answer = ""
                → 图路由拿到 need_tool_call=True，走 act 节点

            ✗ 直接答（用户问"你好"，或 iteration 已达 max=10）:
                state.need_tool_call = False
                state.action = None
                state.final_answer = "你好！有什么可以帮你的？"
                → 图路由走 answer 节点收尾

        【覆盖的安全边界】
            边界1 - Prompt 注入防护：在 prepare_state 阶段已做（本函数上游）
            边界2 - 输出校验/参数修复：LLM 返回的 tool_calls 参数 JSON 可能非法
                    （尾逗号/缺括号），由 repair_json 在 llm.invoke_with_tools 内修复
            边界3 - 死循环拦截：若 iteration ≥ max_iterations 且仍要调工具，
                    强制 need_tool_call=False 走 answer，防 ReAct 无限循环

        【为什么 max_iterations=10？】项目硬约束：ReAct 循环不超过 10 次防死循环，
            超出视为模型陷入"调工具-看结果-再调工具"的死循环，必须强制终止。
        """
        state.iteration += 1
        _logger.info(f"ReAct循环 iteration={state.iteration}/{state.max_iterations}")

        tools_schema = []
        tools_desc = "（无可用工具）"
        if self.tool_registry:
            selected = self.tool_registry.list_tools()
            if self.router:
                # ------------------------------------------------------------------
                # 【L0 路由：只在本轮第 1 次算名单，之后沿用】
                #   每轮都重算要再调一次 embedding，且名单中途变化会让模型困惑
                #   （上一轮还在的工具这一轮突然没了）。
                #   沿用同一份名单：工具集稳定，ReAct 才能稳定收敛。
                #
                # 【这一步之后远程工具才真正就位】
                #   网关目录里的工具此刻才连 server、注册进 registry；
                #   没被召回的远程工具不会起进程，也不会进 prompt。
                # ------------------------------------------------------------------
                if not state.routed_tools:
                    decision = self.router.route(
                        state.user_query,
                        registry=self.tool_registry,
                        skill_name=state.active_skill,
                    )
                    state.routed_tools = list(decision.tool_names)
                selected = [
                    t for t in self.tool_registry.list_tools()
                    if t.name in state.routed_tools
                ]
                _logger.info(
                    f"本轮工具名单 {len(selected)}/{len(state.routed_tools or [])} 个"
                    f"（候选池 {len(self.tool_registry.list_tools())}）: "
                    + "、".join(t.name for t in selected)
                )
            allowed = self._allowed_tools(state.principal, selected)
            user_level = state.principal.user_level if state.principal else "normal"
            tools_schema = self.tool_registry.get_openai_tools_schema(
                user_level=user_level, allowed=allowed
            )
            tools_desc = "\n".join(
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in tools_schema
            ) or "（无可用工具）"

        tool_history = "\n".join(
            f"  - {log.get('name', '')}({log.get('arguments', {})}) → {log.get('result', '')[:100]}"
            for log in state.tool_calls_history
        ) or "（无）"

        # ------------------------------------------------------------------
        # 【这一行替代了原来的 REASON_PROMPT.format(...)】
        #   同样的 8 个槽位，多得到三件事：
        #     1. 变量表由模板自己算出来，少传一个就抛 PromptError，
        #        不会出现"槽位空着但照样发请求"的静默降质
        #     2. 上下文按 token 预算裁剪，超长 RAG 不再整段塞进去
        #     3. 返回值里带 prompt_tokens 与 trimmed，裁剪动作可查（见 debug 日志）
        #   messages 已经是 [system, user] 两条，system 用 __init__ 里算好的那份。
        # ------------------------------------------------------------------
        rec = self._builder.build_record(
            state.user_query,
            permission_context=self._permission_context(state.principal),
            skill_context=state.skill_context or "（未匹配到专门技能，按通用企业助手流程处理）",
            rag_context=state.rag_context or "（知识库无相关内容）",
            memory_context=state.memory_context or "（无历史记忆）",
            tool_history=tool_history,
            observation=state.observation or "（初始状态，暂无工具观察）",
            tools_desc=tools_desc,
        )
        if rec["trimmed"]:
            _logger.warning(f"提示词上下文超预算已裁剪: {rec['trimmed']}")

        result = self.llm.invoke_with_tools(
            messages=rec["messages"],
            tools=tools_schema,
        )

        state.reasoning = result["content"] or ""
        tool_calls = result.get("tool_calls")

        if tool_calls and state.iteration < state.max_iterations:
            # ------------------------------------------------------------------
            # 【重复动作强制收敛】模型拿到工具结果后仍反复调同一个工具时的刹车
            #   判定：本次动作与「最近一次成功调用」的工具名与参数完全一致
            #   处理：不再执行，把上次结果写回 observation，直接进入回答环节
            #
            # 【为什么不只靠 max_iterations=10】
            #   那个上限只是最后一道刹车，在此之前会白跑 9 轮 LLM。
            #   实测：同一句"3*7"，模型连续调了 9 次计算器，
            #   单次对话被拖到 14 秒、token 约为正常 10 倍。
            #   命中重复立刻收尾，同样的问题从 14 秒降到 1 轮内解决。
            #
            # 【输入输出对照】
            #   输入：tool_calls[0] = {'name':'calculator','arguments':{'expression':'3*7'}}
            #         history 里已有同名同参且成功的一条（result='21'）
            #   输出：need_tool_call=False，observation='...结果如下...21'
            # ------------------------------------------------------------------
            repeated = self._last_same_success_call(state, tool_calls[0])
            if repeated is not None:
                _logger.warning(
                    f"检测到重复工具调用，强制收敛: {tool_calls[0].get('name')}"
                )
                state.need_tool_call = False
                state.observation = (
                    f"工具 {repeated.get('name', '')} 已用相同参数执行过，"
                    f"结果如下（不要重复调用，直接据此作答）：\n{repeated.get('result', '')}"
                )
                state.final_answer = ""
                return state
            state.need_tool_call = True
            state.action = tool_calls[0]
        else:
            state.need_tool_call = False
            state.final_answer = result["content"] or "抱歉，我暂时无法回答这个问题。"

        if state.iteration >= state.max_iterations and state.need_tool_call:
            _logger.warning(f"达到最大循环次数{state.max_iterations}，强制终止ReAct循环")
            state.need_tool_call = False
            state.final_answer = result["content"] or "已达到最大推理次数，基于当前信息给出最佳答案。"

        return state

    # 【什么】在历史里找「同名同参且成功过」的那次调用
    # 【作用】给重复调用刹车提供依据；找不到就返回 None，表示本动作是新的
    @staticmethod
    def _last_same_success_call(state: AgentState, call: dict) -> Optional[dict]:
        """在历史里找与本次动作完全相同（工具名+参数）且执行成功过的那次记录。

        Args:
            state: 当前 Agent 状态，读它的 tool_calls_history
            call:  本次模型产出的动作，形如 {'name': 'calculator', 'arguments': {...}}

        Returns:
            命中的历史记录 dict（含 result）；没有完全相同的成功记录则返回 None。
        """
        name = call.get("name", "")
        args = call.get("arguments", {})
        for log in reversed(getattr(state, "tool_calls_history", []) or []):
            if log.get("success") and log.get("name") == name and log.get("arguments") == args:
                return log
        return None

    # 【什么】ReAct 的 Action 步骤：真正执行工具
    # 【作用】把执行结果写回 observation 与两条日志，供下一轮 Reason 参考
    def act_step(self, state: AgentState) -> AgentState:
        """ReAct Action 步骤：执行工具调用（本地 or MCP）。"""
        action = state.action                        # 本轮待执行动作
        tool_name = action.get("name", "")           # 工具名
        arguments = action.get("arguments", {})      # 参数字典

        _logger.info(f"工具调用: {tool_name} | args={arguments}")

        if not isinstance(arguments, dict) or not arguments:
            """边界2兜底：JSON 修复后仍无有效参数时不再硬调工具，改为向用户追问缺失参数。"""
            _logger.warning(f"工具 {tool_name} 参数缺失，触发需求澄清（边界2）")
            state.tool_call_logs.append({
                "name": tool_name,
                "arguments": arguments if isinstance(arguments, dict) else {},
                "result": "参数缺失，已中断调用并向用户追问",
                "success": False,
                "duration_ms": 0,
            })
            state.observation = (
                f"工具 {tool_name} 缺少必要参数，本轮不执行。"
                "请直接向用户追问缺失的参数，不要用猜测的默认值调用。"
            )
            state.action = {}
            return state

        if not self.tool_registry:
            state.observation = "工具注册器未初始化，无法调用工具"
            return state

        result = self.tool_registry.execute_tool(
            name=tool_name,
            params=arguments,
            session_id=state.session_id,
            context=self._tool_context(state.principal),
        )

        self._audit_tool_call(state.principal, tool_name, arguments, result, state.session_id)

        log_entry = {
            "name": tool_name,                  # 工具名
            "arguments": arguments,             # 入参，重复调用检测靠它比对
            "result": str(result.result)[:500] if result.success else result.error,   # 成功截前 500 字，失败存错误
            "success": result.success,          # 是否成功
            "duration_ms": result.duration_ms,  # 耗时（毫秒）
        }
        state.tool_calls_history.append(log_entry)   # 供下一轮 prompt 与重复检测
        state.tool_call_logs.append(log_entry)       # 供 save 节点落 tool_call_log 表

        if result.success:
            state.observation = str(result.result)
        else:
            state.observation = f"工具执行失败: {result.error}"
            _logger.warning(f"工具执行失败: {tool_name} | {result.error}")

        state.action = {}
        return state

    # 【什么】生成最终完整答案
    # 【作用】模型已给答案就直接沿用，否则汇总 RAG/观察/推理再问一次模型
    def generate_answer(self, state: AgentState) -> AgentState:
        """生成最终完整答案。"""
        if state.final_answer and not state.need_tool_call:
            pass                                # 已有答案且不再调工具：原样保留，不多花一次调用
        else:
            prompt = (
                f"## 用户提问\n{state.user_query}\n\n"
                f"## 知识库参考\n{state.rag_context or '无'}\n\n"
                f"## 工具观察结果\n{state.observation or '无'}\n\n"
                f"## 推理过程\n{state.reasoning}\n\n"
                "请基于以上信息，给出完整详细的最终答案。"
            )
            state.final_answer = self.llm.invoke(
                prompt,
                system_prompt=self.system_prompt,
            )

        state.finished = True
        return state

    # 【什么】生成一句话摘要（用于向量入库）
    # 【作用】优先抠模型自带的【摘要】标签，其次调摘要生成器，都不行就截提问前 50 字
    def generate_summary(self, state: AgentState) -> str:
        """生成精简摘要（用于向量入库）。"""
        if not state.final_answer:
            return state.user_query[:50]        # 兜底：没有答案时用提问前 50 字
        if "【摘要】" in state.final_answer:
            import re
            m = re.search(r'【摘要】(.+?)(?:$|\n)', state.final_answer, re.S)
            if m:
                return m.group(1).strip()[:100]     # 标签里的摘要最多留 100 字
        if self.memory:
            try:
                summary = self.memory.summary_generator.generate_summary(
                    state.user_query, state.final_answer
                )
                return summary
            except Exception as e:
                _logger.warning(f"摘要生成降级: {e}")
        return state.user_query[:50]
