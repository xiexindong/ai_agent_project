"""
LangGraph 图构建模块
=====================

【这文件干啥的？】把 ReActAgent 的"思考→调工具→再思考"循环，
装进一张 LangGraph 状态图里，让它能：分支判断（要不要调工具）、
循环（调完工具回到思考）、断点续跑（状态可序列化存 PG）。

【为什么用图，不直接 while 循环？】
    while 循环：所有跳转逻辑糊在一坨 if/else 里，看不清流程全貌
    LangGraph 图：节点=工序，边=跳转，整张流程一目了然，还自带状态管理、
    可视化导出、断点续跑能力。类比 Node 生态里 NestJS 的"装饰器路由 + 依赖注入"
    相对 Express 手写 if 路由的升级。

【图结构】
    prepare → reason → should_continue? → act → reason (循环)
                                    → answer → save → END

【执行轨迹例子】用户问"3*7 等于多少"
    输入: query="3*7 等于多少", session_id="abc123", user_id="cli-user"
    轨迹:
        prepare  → 召回历史记忆、匹配技能"数学计算"、注入 SOP
        reason   → LLM 决策"要调 calculator 工具"  (need_tool_call=True)
        should_continue → 返回 "act"
        act      → 执行 calculator(3*7) → observation="21"
        reason   → LLM 看到结果 21，决策"不再调工具，直接答"  (finished=True)
        should_continue → 返回 "answer"
        answer   → 生成"3*7 等于 21" + 摘要"数学计算:3*7=21"
        save     → 落 PG（原文+摘要+向量+工具日志）→ END
    输出: {final_answer:"3*7 等于 21", iteration:2, tool_call_logs:[...]}
"""
from typing import Annotated, Any
from langgraph.graph import StateGraph, END
from modules.common.logger import get_logger
from modules.agent.state import AgentStateDict
from modules.agent.react_agent import ReActAgent

_logger = get_logger("graph")


class AgentGraph:
    """LangGraph 状态化 Agent 编排器。"""

    def __init__(self, react_agent: ReActAgent):
        self.agent = react_agent
        self._graph = self._build_graph()

    def _build_graph(self):
        """
        构建 LangGraph 状态图：5 个节点 + 6 条边（含 1 条条件边）。

        【节点 = 工序】每个节点接收 state、返回 state 更新片段
            prepare : 安全检查 + 召回记忆/技能/RAG，初始化 state
            reason  : LLM 决策——直接答 or 调工具（ReAct 的"思考"）
            act     : 执行工具（本地 Tool / 远程 MCP），结果回写 observation
            answer  : 生成完整答案 + 一句话摘要
            save    : 落库（原文+摘要+向量+工具日志），状态序列化供断点续跑

        【边 = 跳转】固定边直连，条件边由 _should_continue 决定去向
            prepare → reason                              （固定，准备工作做完必进推理）
            reason → should_continue? → act | answer      （条件边，看模型决策）
            act → reason                                  （固定，工具结果回到思考，形成 ReAct 循环）
            answer → save → END                           （固定，答完落库结束）

        【条件分支结果对比】_should_continue 返回值决定走哪条边
            ✓ 返回 "act"    : 模型要调工具 → 走 act → 回 reason 继续思考
            ✗ 返回 "answer" : 模型已得出答案 / 出错 / 达最大轮次 → 走 answer 收尾

            例子（用户问"3*7"）：
                第1次 reason: need_tool_call=True  → 返回 "act"  → 调 calculator
                第2次 reason: finished=True        → 返回 "answer" → 生成答案收尾
        """
        workflow = StateGraph(AgentStateDict)

        # ---- 定义节点 ----
        workflow.add_node("prepare", self._prepare_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("answer", self._answer_node)
        workflow.add_node("save", self._save_node)

        # ---- 定义边 ----
        workflow.set_entry_point("prepare")
        workflow.add_edge("prepare", "reason")
        # 条件分支：reason 后判断是否需要工具
        # path_map 显式写出 {"act":"act","answer":"answer"} —— 即使 key==node 名，
        # 也保留键值对写法，便于 LangGraph 校验/可视化导出（项目工程约定）
        workflow.add_conditional_edges(
            "reason",
            self._should_continue,
            {
                "act": "act",
                "answer": "answer",
            },
        )
        # act 执行后回到 reason（ReAct 循环）
        workflow.add_edge("act", "reason")
        workflow.add_edge("answer", "save")
        workflow.add_edge("save", END)

        compiled = workflow.compile()
        _logger.info("LangGraph 图构建完成: prepare → reason → [act → reason 循环] → answer → save")
        return compiled

    # ==================== 节点实现 ====================

    def _prepare_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """准备节点：初始化状态、召回记忆与RAG上下文。"""
        _logger.info(f"节点[prepare]: session={state.get('session_id', '')}")
        breakpoint()  # 【调试断点】图节点[prepare]入口：可看 state 初始状态

        # 如果是首轮调用，需要初始化状态
        if not state.get("prepared"):
            full_state = self.agent.prepare_state(
                query=state.get("user_query", ""),
                session_id=state.get("session_id", ""),
                user_id=state.get("user_id", ""),
            )
            updates = full_state.to_dict()
            updates["prepared"] = True
            # 合并到现有 state
            return updates
        return {}

    def _reason_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """推理节点：LLM ReAct 分析，判断是否调用工具。"""
        from modules.agent.state import AgentState
        agent_state = AgentState.from_dict(state)
        breakpoint()  # 【调试断点】图节点[reason]入口：可看 agent_state（含 observation 上轮结果）
        _logger.info(
            f"节点[reason]: iteration={agent_state.iteration + 1} "
            f"query='{agent_state.user_query[:30]}...'"
        )
        updated = self.agent.reason_step(agent_state)
        updates = {
            "iteration": updated.iteration,
            "reasoning": updated.reasoning,
            "action": updated.action,
            "need_tool_call": updated.need_tool_call,
            "final_answer": updated.final_answer,
            "finished": updated.finished,
        }
        return updates

    def _act_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行节点：调用工具（本地 or MCP）。"""
        from modules.agent.state import AgentState
        agent_state = AgentState.from_dict(state)
        breakpoint()  # 【调试断点】图节点[act]入口：可看 agent_state.action（要调的工具）
        _logger.info(f"节点[act]: tool={agent_state.action.get('name', '')}")
        updated = self.agent.act_step(agent_state)
        return {
            "observation": updated.observation,
            "action": updated.action,
            "tool_calls_history": updated.tool_calls_history,
            "tool_call_logs": updated.tool_call_logs,
        }

    def _answer_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """答案节点：生成完整答案 + 精简摘要。"""
        from modules.agent.state import AgentState
        agent_state = AgentState.from_dict(state)
        breakpoint()  # 【调试断点】图节点[answer]入口：可看 agent_state（最终状态）
        _logger.info("节点[answer]: 生成最终答案")
        updated = self.agent.generate_answer(agent_state)
        # 生成摘要
        summary = self.agent.generate_summary(updated)
        return {
            "final_answer": updated.final_answer,
            "llm_summary": summary,
            "finished": True,
        }

    def _save_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """保存节点：记忆入库（原文+摘要+向量），Agent状态持久化。"""
        session_id = state.get("session_id", "")
        user_id = state.get("user_id", "")
        breakpoint()  # 【调试断点】图节点[save]入口：可看 state.final_answer / state.llm_summary / tool_call_logs
        _logger.info(f"节点[save]: session={session_id}")

        if self.agent.memory and session_id and user_id:
            try:
                # 生成摘要向量
                summary = state.get("llm_summary", "")
                from modules.memory.embedding_service import get_embedding_service
                svc = get_embedding_service()
                summary_emb = svc.embed(summary) if summary else None

                # 生成【问句向量】：长期记忆的主检索列（问句 vs 历史问句，对称检索更准）
                # 优先复用 perceive 阶段已算好的那个（react_agent.py 里 embed(clean_query)，
                # 暂存在 state.summary_embedding 字段里——注意它存的其实是问句向量，字段名有历史包袱），
                # 复用可以省一次 embedding 调用；拿不到才现场算
                user_query = state.get("user_query", "")
                query_emb = state.get("summary_embedding") or (
                    svc.embed(user_query) if user_query else None
                )

                # 四数据落库（原文+摘要+摘要向量+问句向量，缺一不可）
                self.agent.memory.save_turn(
                    session_id=session_id,
                    user_id=user_id,
                    user_query=user_query,
                    llm_full_answer=state.get("final_answer", ""),
                    llm_summary=summary,
                    summary_embedding=summary_emb or [],
                    query_embedding=query_emb or [],
                    tool_call_log=state.get("tool_call_logs", []),
                    rag_sources=[
                        {"source": s.get("file_name", ""), "score": s.get("rerank_score", s.get("score", 0))}
                        for s in state.get("rag_sources", [])
                    ],
                )

                # Agent 状态持久化（断点续跑）
                from modules.agent.state import AgentState
                agent_state = AgentState.from_dict(state)
                self.agent.memory.pg.update_session_state(
                    session_id, agent_state.to_dict()
                )
                _logger.info("对话记忆 + Agent状态 已持久化")
            except Exception as e:
                _logger.error(f"记忆保存异常: {e}")
        return {}

    def _should_continue(self, state: dict) -> str:
        """
        条件路由：reason 之后该往哪走？返回 "act" 或 "answer"。

        【这函数是整张图的"红绿灯"】LangGraph 用它的返回值当 key，
        去 path_map {"act":"act","answer":"answer"} 里查下一步走哪个节点。

        【三种分支结果对比】
            情况 A：finished=True
                state 例：{finished:True, final_answer:"3*7 等于 21", ...}
                → 返回 "answer" → 走 answer 节点收尾

            情况 B：need_tool_call=True 且 error 为空
                state 例：{need_tool_call:True, action:{name:"calculator",...}, error:None, ...}
                → 返回 "act" → 走 act 节点执行工具，执行完回 reason 继续思考

            情况 C：need_tool_call=True 但 error 有值（工具/LLM 异常等）
                state 例：{need_tool_call:True, error:"连接超时", ...}
                → 走最后的兜底 return "answer" → 不再调工具，直接收尾防死循环

        【一句话记忆】"答完了走 answer，要调工具且没出错走 act，出错也走 answer"
        """
        breakpoint()  # 【调试断点】条件路由：可看 state.need_tool_call / state.finished（决定走 act 还是 answer）
        if state.get("finished"):
            return "answer"
        if state.get("need_tool_call") and not state.get("error"):
            return "act"
        return "answer"

    # ==================== 执行入口 ====================

    def run(
        self,
        query: str,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """执行完整 Agent 流程。"""
        _logger.info(f"Agent执行: session={session_id} | query='{query[:50]}'")
        breakpoint()  # 【调试断点】Agent图执行入口：可看 query / session_id / user_id
        # 标注 AgentStateDict：LangGraph invoke 期待状态 TypedDict，纯 dict 会被类型检查器拒收
        initial_state: AgentStateDict = {
            "session_id": session_id,
            "user_id": user_id,
            "user_query": query,
            "prepared": False,
        }
        result = self._graph.invoke(initial_state, {"recursion_limit": 30})
        _logger.info(
            f"Agent完成: iterations={result.get('iteration', 0)} | "
            f"tool_calls={len(result.get('tool_call_logs', []))} | "
            f"finished={result.get('finished', False)}"
        )
        return result
