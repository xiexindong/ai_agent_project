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
    prepare → (复杂流程? plan) → reason → should_continue? → act → reason (循环)
                                                       → answer → save → END

    plan 是可选节点：AGENT_PLANNING_ENABLED 关闭时永不进入，
    开了也只有"判定为复杂流程"才进（见 ReActAgent.should_plan）。

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
# 【什么】LangGraph 图框架 + 敏感词校验 + 状态与 ReAct 引擎
# 【作用】建图、跑图、落库都要它们；check_sensitive 只在自动初审里用
from typing import Annotated, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig   # 给 invoke/get_state 的 config 标注类型
from modules.common.logger import get_logger
from modules.common.validator import check_sensitive
from modules.agent.state import AgentStateDict
from modules.agent.react_agent import ReActAgent
from config import get_settings

_logger = get_logger("graph")


# 【什么】LangGraph 状态图的编排器
# 【作用】把 ReAct 的循环装进图里，换取分支判断、断点续跑与状态持久化
class AgentGraph:
    """LangGraph 状态化 Agent 编排器。"""

    # 【什么】拿一个 ReActAgent 建图
    # 【作用】图只负责编排，真正的推理与工具执行都在 ReActAgent 里
    def __init__(self, react_agent: ReActAgent):
        self.agent = react_agent                 # 业务执行体：prepare/reason/act/answer 都在它身上
        self._checkpointer = None                # 断点续跑用；未开启或不可用时保持 None
        self._graph = self._build_graph()        # 编译后的图，run() 直接 invoke 它

    # 【什么】按「6 节点 + 8 条边」搭出整张图并编译
    # 【作用】流程全貌集中在这里，改跳转只改这一段
    def _build_graph(self):
        """
        构建 LangGraph 状态图：6 个节点 + 8 条边（含 2 条条件边）。

        【节点 = 工序】每个节点接收 state、返回 state 更新片段
            prepare : 安全检查 + 召回记忆/技能/RAG，初始化 state
            plan    : 复杂流程拆成有序步骤（Plan-and-Execute，可选节点）
            reason  : LLM 决策——直接答 or 调工具（ReAct 的"思考"）
            act     : 执行工具（本地 Tool / 远程 MCP），结果回写 observation
            answer  : 生成完整答案 + 一句话摘要
            save    : 落库（原文+摘要+向量+工具日志），状态序列化供断点续跑

        【边 = 跳转】固定边直连，条件边由 _after_prepare / _should_continue 决定去向
            prepare → plan | reason | answer               （条件边，看是否复杂 + 是否被拦截）
            plan → reason                                  （固定，规划完必进推理）
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

        # 【为什么下面 5 行要 type: ignore】langgraph 1.2 把 add_node 的 action 形参
        #   收窄成 StateNode（Runnable 体系），传绑定方法（MethodType）会被判不兼容；
        #   运行期 langgraph 会把它包成 RunnableCallable，行为完全正常。
        workflow.add_node("prepare", self._prepare_node)   # type: ignore
        workflow.add_node("plan", self._plan_node)         # type: ignore   # 复杂流程先拆步骤（可选节点）
        workflow.add_node("reason", self._reason_node)     # type: ignore
        workflow.add_node("act", self._act_node)           # type: ignore
        workflow.add_node("answer", self._answer_node)     # type: ignore
        workflow.add_node("save", self._save_node)         # type: ignore

        workflow.set_entry_point("prepare")
        workflow.add_conditional_edges(
            "prepare",
            self._after_prepare,
            {
                "plan": "plan",             # 判定为复杂流程 → 先规划再推理
                "reason": "reason",
                "answer": "answer",
            },
        )
        workflow.add_edge("plan", "reason")     # 规划完必进推理；之后 reason↔act 循环不再回 plan
        workflow.add_conditional_edges(
            "reason",
            self._should_continue,
            {
                "act": "act",
                "answer": "answer",
            },
        )
        workflow.add_edge("act", "reason")
        workflow.add_edge("answer", "save")
        workflow.add_edge("save", END)

        if get_settings().checkpoint_enabled:
            try:
                from langgraph.checkpoint.memory import MemorySaver
                self._checkpointer = MemorySaver()
            except Exception as e:
                _logger.warning(f"checkpoint 不可用（断点续跑关闭）: {e}")
        try:
            compiled = workflow.compile(checkpointer=self._checkpointer)
        except TypeError:
            compiled = workflow.compile()
            self._checkpointer = None
        _logger.info("LangGraph 图构建完成: prepare → reason → [act → reason 循环] → answer → save")
        return compiled


    # 【什么】prepare 节点：召回记忆/技能/RAG，初始化 state
    # 【作用】只跑一次（靠 prepared 标记），避免循环回来重复召回
    def _prepare_node(self, state: dict[str, Any]) -> "AgentStateDict":
        """准备节点：初始化状态、召回记忆与RAG上下文。"""
        breakpoint()                # TODO: 调试用，提交前删除
        _logger.info(f"节点[prepare]: session={state.get('session_id', '')}")

        if not state.get("prepared"):
            full_state = self.agent.prepare_state(
                query=state.get("user_query", ""),
                session_id=state.get("session_id", ""),
                user_id=state.get("user_id", ""),
                principal=state.get("principal"),
            )
            updates = full_state.to_dict()
            updates["prepared"] = True      # 打上已准备标记，循环回到本节点时不再重复召回
            return updates
        return {}                           # 已准备过：返回空更新，原样往下走

    # 【什么】plan 节点：给复杂流程拆一份执行计划
    # 【作用】只跑一次（图结构上不会回到本节点）；规划不可用时不进这个节点
    def _plan_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """规划节点：把 query 拆成有序步骤，写进 sub_tasks 供后续每轮参考。"""
        from modules.agent.state import AgentState
        agent_state = AgentState.from_dict(state)
        _logger.info("节点[plan]: 生成执行计划")
        updated = self.agent.plan_step(agent_state)
        return {
            "sub_tasks": updated.sub_tasks,
            "current_task_index": updated.current_task_index,
            "task_results": updated.task_results,
            "plan_rationale": updated.plan_rationale,
            "plan_replan_count": updated.plan_replan_count,
        }

    # 【什么】reason 节点：调一次 ReAct 的 Reason 步骤
    # 【作用】把 LLM 的决策结果（要不要调工具、答案是什么）写回图状态
    def _reason_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """推理节点：LLM ReAct 分析，判断是否调用工具。"""
        from modules.agent.state import AgentState
        agent_state = AgentState.from_dict(state)
        _logger.info(
            f"节点[reason]: iteration={agent_state.iteration + 1} "
            f"query='{agent_state.user_query[:30]}...'"
        )
        # 调用 ReAct 步骤
        # ------------------------------------------------------------------
        # 【reason_step 内含防打转的三道刹车，命中任何一道都会跳过 act 直接收尾】
        #   ① 同名同参：本次动作与最近一次成功调用完全一致 → 回灌旧结果收尾
        #   ② 交替环：把「历史 + 本次待执行」各压成指纹（工具名|排序后参数），
        #      检查序列尾部是否为周期 2~3 的片段原样重复两遍（A→B→A→B）
        #   ③ 无进展：动作不成环也不重复，但最近 3 次成功调用的结果一字不差
        #   命中即 need_tool_call=False，把原因写进 observation 让模型直接作答
        #   （实现见 react_agent.py:684 起的 loop_guard 段）
        # ------------------------------------------------------------------
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

    # 【什么】act 节点：执行本轮工具调用
    # 【作用】把观察结果与两条工具日志回写状态，下一轮 reason 才看得到
    def _act_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行节点：调用工具（本地 or MCP）。"""
        from modules.agent.state import AgentState
        agent_state = AgentState.from_dict(state)
        _logger.info(f"节点[act]: tool={agent_state.action.get('name', '')}")
        updated = self.agent.act_step(agent_state)
        return {
            "observation": updated.observation,
            "action": updated.action,
            "tool_calls_history": updated.tool_calls_history,
            "tool_call_logs": updated.tool_call_logs,
            # 【计划推进结果要写回图状态】本步标记完成/失败、指针前移、
            #   以及可能的重规划都发生在 act_step 里，不回写的话下一轮看到的还是旧计划
            "sub_tasks": updated.sub_tasks,
            "current_task_index": updated.current_task_index,
            "task_results": updated.task_results,
            "plan_replan_count": updated.plan_replan_count,
        }

    # 【什么】answer 节点：出最终答案 + 一句话摘要
    # 【作用】收尾节点，跑完把 finished 置 True，交给 save
    def _answer_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """答案节点：生成完整答案 + 精简摘要。"""
        from modules.agent.state import AgentState
        agent_state = AgentState.from_dict(state)
        _logger.info("节点[answer]: 生成最终答案")
        updated = self.agent.generate_answer(agent_state)
        summary = self.agent.generate_summary(updated)
        return {
            "final_answer": updated.final_answer,
            "llm_summary": summary,
            "finished": True,
        }

    # 【什么】save 节点：把这一轮对话与状态落库
    # 【作用】原文 + 摘要 + 向量 + 工具日志入库，顺带跑初审入队
    def _save_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """保存节点：记忆入库（原文+摘要+向量），Agent状态持久化。"""
        session_id = state.get("session_id", "")
        user_id = state.get("user_id", "")
        _logger.info(f"节点[save]: session={session_id}")

        if self.agent.memory and session_id and user_id:
            try:
                summary = state.get("llm_summary", "")
                from modules.memory.embedding_service import get_embedding_service
                svc = get_embedding_service()
                summary_emb = svc.embed(summary) if summary else None

                user_query = state.get("user_query", "")
                query_emb = state.get("summary_embedding") or (
                    svc.embed(user_query) if user_query else None
                )

                history_id = self.agent.memory.save_turn(
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

                from modules.agent.state import AgentState
                agent_state = AgentState.from_dict(state)
                snapshot = agent_state.to_dict()
                # principal 是 Principal 活体对象，json.dumps 无法序列化，会让整段落库抛异常
                # 且快照里也不需要它（续跑时由请求方重新解析传入），落库前剔除
                snapshot.pop("principal", None)
                self.agent.memory.pg.update_session_state(session_id, snapshot)
                self._save_tool_logs(state)
                self._auto_review(state, history_id)
                _logger.info("对话记忆 + Agent状态 已持久化")
            except Exception as e:
                _logger.error(f"记忆保存异常: {e}")
        return {}

    # 【什么】把本轮工具调用逐条写进 tool_call_log 表
    # 【作用】与 chat_history 里的 JSONB 互补，是审计的原始依据
    def _save_tool_logs(self, state: dict) -> None:
        """工具调用全量落 tool_call_log 表（审计依据，与 chat_history 的 JSONB 互补）。"""
        for tc in state.get("tool_call_logs", []):
            self.agent.memory.pg.save_tool_log(
                state.get("session_id", ""),
                state.get("user_id", ""),
                tc.get("name", ""),
                tc.get("type", "local"),                        # 工具类型：local / mcp
                tc.get("arguments", {}) or {},                  # 入参，None 时记空字典
                str(tc.get("result", ""))[:2000],               # 结果最多存 2000 字
                "success" if tc.get("success", True) else "error",   # 缺省按成功记
                tc.get("duration_ms", 0),                       # 耗时，缺失记 0
            )

    # 【什么】自动初审：命中风险就挂人工复核队列
    # 【作用】四类风险只入队优先级最高的那条，同一轮不会刷出多条待办
    def _auto_review(self, state: dict, history_id) -> None:
        """自动初审：命中四类风险之一即挂人工复核（合规留痕 + 数据反哺）。

        四类触发：sensitive（注入/敏感词）> tool_fail（工具失败）> low_sim（记忆低置信）> no_rag（无知识库依据）
        只入队优先级最高的一类，避免同一轮对话刷出多条待办。
        """
        if not get_settings().review_auto_enabled or not history_id:
            return
        query = state.get("user_query", "")
        reasons = []
        if state.get("error") == "prompt_injection_blocked" or check_sensitive(query):
            reasons.append(("sensitive", query))                            # 优先级 1：注入/敏感词
        if any(not tc.get("success", True) for tc in state.get("tool_call_logs", [])):
            reasons.append(("tool_fail", "本轮存在工具调用失败"))             # 优先级 2：工具失败
        if state.get("memory_low_confidence"):
            reasons.append(("low_sim", "长期记忆召回低置信"))                 # 优先级 3：记忆低置信
        if self.agent.rag and not state.get("rag_sources"):
            reasons.append(("no_rag", query))                               # 优先级 4：无知识库依据

        if not reasons:
            return
        review_type, content = reasons[0]                                   # 只取排序最靠前的那一类
        self.agent.memory.pg.add_to_review_queue(
            state.get("session_id", ""), state.get("user_id", ""),
            history_id, content[:500], review_type,                         # 内容最多存 500 字
        )
        _logger.info(f"自动初审入队: type={review_type} history_id={history_id}")

    # 【什么】prepare 之后的路由：进循环还是直接收尾
    # 【作用】注入拦截/异常的答案直接发走，不进 reason 免得被 LLM 覆盖
    def _after_prepare(self, state: dict) -> str:
        """prepare 之后的路由：注入拦截/异常直接收尾，否则进 ReAct 循环。

        没有这条边时，prepare 里置的 finished 会被 reason 节点的 LLM 输出覆盖，
        注入拦截话术实际发不出去（守卫形同虚设）。
        """
        if state.get("error") or (state.get("finished") and state.get("final_answer")):
            return "answer"
        # 【规划判定】复杂流程先拆步骤；简单问题直接进 ReAct，行为与改造前一致
        from modules.agent.state import AgentState
        if self.agent.should_plan(AgentState.from_dict(state)):
            return "plan"
        return "reason"

    # 【什么】整张图的「红绿灯」：返回 act 或 answer
    # 【作用】LangGraph 拿返回值当 key 查 path_map，决定下一步走哪个节点
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
        if state.get("finished"):
            return "answer"
        if state.get("need_tool_call") and not state.get("error"):
            return "act"
        return "answer"


    # 【什么】跑完整条 Agent 流程（外部唯一入口）
    # 【作用】组装初始状态、按需断点续跑，最后把终态 dict 返回给调用方
    def run(
        self,
        query: str,                 # 本轮用户提问
        session_id: str,            # 会话 ID，兼作 thread_id
        user_id: str,               # 用户标识
        principal=None,             # 权限载体，None = 未启用权限
        resume: bool = True,        # True = 有未完成快照时接着跑
    ) -> dict[str, Any]:
        """
        执行完整 Agent 流程。

        Args:
            principal: 权限载体（Principal），由 API 入口解析后透传。
                       它决定本次会话能看哪些知识库、能调哪些工具、能看多少数据。
                       None = 未启用权限（CLI 调试）。
        """
        _logger.info(
            f"Agent执行: session={session_id} | user={user_id} | "
            f"权限={'已加载' if principal else '未启用'} | query='{query[:50]}'"
        )
        initial_state: AgentStateDict = {
            "session_id": session_id,       # 会话 ID，兼作 thread_id
            "user_id": user_id,             # 用户标识
            "user_query": query,            # 原始提问（prepare 节点会清洗）
            "principal": principal,         # 权限载体
            "prepared": False,              # False = 本轮还没跑 prepare
        }
        # RunnableConfig 是 TypedDict，标成它之后传给 invoke/get_state 才不会被判成普通 dict
        config: RunnableConfig = {"configurable": {"thread_id": session_id}, "recursion_limit": 30}   # 30 = 图内最大跳转步数

        if resume and self._checkpointer is not None:
            try:
                snapshot = self._graph.get_state(config)
                if snapshot and getattr(snapshot, "next", None):
                    # ← 续跑：从 next 指向的节点继续
                    _logger.warning(f"检测到未完成任务，断点续跑: session={session_id} next={snapshot.next}")
                    return self._graph.invoke(None, config)
            except Exception as e:
                _logger.warning(f"断点恢复失败，按新任务执行: {e}")

        # 【workflow 怎么跑起来】LangGraph 的 invoke 收到 initial_state 后：
        #   1. 把初始状态 dict 装进 state；
        #   2. 从入口节点 prepare 开始，按边定义的拓扑顺序逐个跑节点；
        #   3. 每个节点 return 的更新片段合并回 state（不是替换，是增量更新）；
        #   4. 遇到条件边（_should_continue）返回值当 key 查 path_map，决定下步去哪；
        #   5. 大模型在 reason_node 和 answer_node 里被调用——不是 invoke 直接调 LLM，
        #      而是 invoke 驱动节点、节点内部再调 self.agent.reason_step() / generate_answer()，
        #      后者通过 LLMClient 发起 HTTP。一直跑到 save → END，终态 dict 返回。
        result: dict[str, Any] | Any = self._graph.invoke(initial_state, config) #新任务：从入口节点开始
        _logger.info(
            f"Agent完成: iterations={result.get('iteration', 0)} | "
            f"tool_calls={len(result.get('tool_call_logs', []))} | "
            f"finished={result.get('finished', False)}"
        )
        return result
