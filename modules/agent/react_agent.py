"""
ReAct Agent 模块
================
标准 ReAct 推理循环：Reason思考 → Action执行 → Observation观察。
整合 RAG 知识库、向量记忆、本地工具、MCP远程工具全链路。
覆盖边界2（输出校验）、边界3（死循环拦截）、边界6（Prompt防护）。
"""
import time
from typing import Optional
from modules.common.logger import get_logger
from modules.common.exceptions import LLMException, ToolException, handle_exception
from modules.common.validator import sanitize_input, check_prompt_injection, repair_json
from modules.agent.llm import LLMClient
from modules.agent.state import AgentState

_logger = get_logger("react")


class ReActAgent:
    """ReAct 推理引擎（Reason-Action-Observation 循环）。"""

    SYSTEM_PROMPT = (
        "你是一个企业级智能AI助手，具备知识库问答、历史记忆、工具调用能力。\n\n"
        "## 工作流程（ReAct）\n"
        "1. Reason：分析用户问题，判断是否需要调用工具\n"
        "2. Action：如需工具，调用对应工具获取信息\n"
        "3. Observation：观察工具返回结果，决定是否继续调用或给出答案\n\n"
        "## 核心规则\n"
        "- 仅基于知识库和工具返回的事实回答，禁止编造\n"
        "- 知识库无相关内容时，如实告知并尝试工具补充\n"
        "- 工具调用需确认参数正确性\n"
        " - 最多循环{max_iter}次，超过后直接给出当前最佳答案\n\n"
        "## 回答要求\n"
        "- 提供完整详细的答案（包含推理依据、数据来源）\n"
        "- 末尾附【摘要】标签，用一句话总结本轮对话核心"
    )

    REASON_PROMPT = (
        "## 用户提问\n{query}\n\n"
        "## 激活技能（SOP作业流程）\n{skill_context}\n\n"
        "## 权限边界（硬性约束，越界即违规）\n{permission_context}\n\n"
        "## 知识库参考\n{rag_context}\n\n"
        "## 历史记忆\n{memory_context}\n\n"
        "## 历史工具调用\n{tool_history}\n\n"
        "## 工具观察结果\n{observation}\n\n"
        "## 可用工具（Function Calling）\n{tools_desc}\n\n"
        "请分析当前情况，决定下一步行动：\n"
        "- 如果已激活技能，优先遵循该技能的作业流程(SOP)，并优先使用其推荐工具\n"
        "- 如果已有足够信息回答，直接给出最终答案\n"
        "- 如果需要更多信息，选择合适的工具调用\n"
        "- 严格使用 Function Calling 格式输出工具调用"
    )

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tool_registry=None,
        rag_chain=None,
        memory_manager=None,
        skill_registry=None,
    ):
        self.llm = llm_client or LLMClient()
        self.tool_registry = tool_registry
        self.rag = rag_chain
        self.memory = memory_manager
        self.skill_registry = skill_registry
        from config import get_settings
        s = get_settings()
        self._max_iter = s.max_react_iterations

    def prepare_state(
        self,
        query: str,
        session_id: str,
        user_id: str,
        user_level: str = "normal",
        principal=None,
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
                stds = self.memory.pg.match_standard_answers(clean_query, limit=2)
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
            "_scopes": {dt: principal.scope_for(dt) for dt in ("users", "orders", "inventory")},
            "_dept_code": principal.dept_code,
            "_user_code": principal.user_code,
            "_mask_level": principal.mask_level,
        }

    @staticmethod
    def _audit_tool_call(principal, tool_name: str, arguments: dict, result):
        """工具调用审计：成功记 allow，失败记 error，并带上入参与耗时。"""
        if principal is None:
            return
        try:
            from modules.enterprise import audit
            audit.log_audit(
                principal.user_code,
                "tool_call",
                f"tool:{tool_name}",
                "allow" if result.success else "error",
                dept_code=principal.dept_code,
                detail={
                    "arguments": arguments,
                    "duration_ms": getattr(result, "duration_ms", 0),
                    "error": (result.error if not result.success else ""),
                },
            )
        except Exception as e:
            _logger.warning(f"工具审计写入失败: {e}")

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
            all_tools = self.tool_registry.list_tools()
            allowed = self._allowed_tools(state.principal, all_tools)
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

        prompt = self.REASON_PROMPT.format(
            query=state.user_query,
            permission_context=self._permission_context(state.principal),
            skill_context=state.skill_context or "（未匹配到专门技能，按通用企业助手流程处理）",
            rag_context=state.rag_context or "（知识库无相关内容）",
            memory_context=state.memory_context or "（无历史记忆）",
            tool_history=tool_history,
            observation=state.observation or "（初始状态，暂无工具观察）",
            tools_desc=tools_desc,
        )

        result = self.llm.invoke_with_tools(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT.format(max_iter=self._max_iter)},
                {"role": "user", "content": prompt},
            ],
            tools=tools_schema,
        )

        state.reasoning = result["content"] or ""
        tool_calls = result.get("tool_calls")

        if tool_calls and state.iteration < state.max_iterations:
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

    def act_step(self, state: AgentState) -> AgentState:
        """ReAct Action 步骤：执行工具调用（本地 or MCP）。"""
        action = state.action
        tool_name = action.get("name", "")
        arguments = action.get("arguments", {})

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

        self._audit_tool_call(state.principal, tool_name, arguments, result)

        log_entry = {
            "name": tool_name,
            "arguments": arguments,
            "result": str(result.result)[:500] if result.success else result.error,
            "success": result.success,
            "duration_ms": result.duration_ms,
        }
        state.tool_calls_history.append(log_entry)
        state.tool_call_logs.append(log_entry)

        if result.success:
            state.observation = str(result.result)
        else:
            state.observation = f"工具执行失败: {result.error}"
            _logger.warning(f"工具执行失败: {tool_name} | {result.error}")

        state.action = {}
        return state

    def generate_answer(self, state: AgentState) -> AgentState:
        """生成最终完整答案。"""
        if state.final_answer and not state.need_tool_call:
            pass
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
                system_prompt=self.SYSTEM_PROMPT.format(max_iter=self._max_iter),
            )

        state.finished = True
        return state

    def generate_summary(self, state: AgentState) -> str:
        """生成精简摘要（用于向量入库）。"""
        if not state.final_answer:
            return state.user_query[:50]
        if "【摘要】" in state.final_answer:
            import re
            m = re.search(r'【摘要】(.+?)(?:$|\n)', state.final_answer, re.S)
            if m:
                return m.group(1).strip()[:100]
        if self.memory:
            try:
                summary = self.memory.summary_generator.generate_summary(
                    state.user_query, state.final_answer
                )
                return summary
            except Exception as e:
                _logger.warning(f"摘要生成降级: {e}")
        return state.user_query[:50]
