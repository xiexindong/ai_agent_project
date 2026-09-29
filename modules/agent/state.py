"""
Agent 状态定义模块
==================
LangGraph 状态化 Agent 的核心状态结构。
支持任务拆解、断点续跑、多轮 ReAct 循环状态追踪。
"""
from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass, field
from langgraph.graph import MessagesState


# 【什么】交给 LangGraph 的全局状态字典
# 【作用】框架节点函数只认 dict，这份 TypedDict 约定了所有可用字段名
class AgentStateDict(TypedDict, total=False):
    """
    LangGraph Agent 全局状态（TypedDict，支持部分字段）。
    贯穿 ReAct 循环、工具调度、记忆管理全链路。
    """
    session_id: str                 # 会话 ID，同时是断点续跑的 thread_id
    user_id: str                    # 用户标识
    user_query: str                 # 本轮用户提问（prepare 阶段已清洗）

    principal: Any                  # 权限载体 Principal；None = 未启用权限

    messages: list                  # 对话消息列表，喂给 LLM 的原始 messages

    rag_context: str                # RAG 召回拼成的上下文文本
    rag_sources: list               # RAG 命中的来源列表，落库时记在 rag_sources

    active_skill: str               # 命中的技能名，未命中为空串
    skill_context: str              # 技能 SOP 上下文，注入 prompt
    # L0 路由选出的本轮工具名单（只算一次，后续 ReAct 轮次沿用）
    routed_tools: list

    memory_context: str             # 长短期记忆拼成的上下文文本
    long_term_memory: list          # 长期记忆条目
    short_term_memory: list         # 短期记忆条目
    memory_compressed: bool         # True = 本轮记忆做过压缩
    memory_low_confidence: bool     # True = 长期记忆召回相似度偏低

    iteration: int                  # 当前 ReAct 轮次，从 1 起算
    max_iterations: int             # 轮次上限，默认 10，达到即强制收尾
    reasoning: str                  # 本轮 LLM 的思考文本（debug 用）
    action: dict                    # 待执行动作 {name, arguments, id}
    observation: str                # 上一轮工具执行结果
    tool_call_logs: list            # 工具流水，save 节点写 tool_call_log 表
    tool_calls_history: list        # 工具历史，供重复调用检测与 prompt 展示

    # 【计划三件套】Plan-and-Execute：复杂流程先拆步骤，再逐步执行
    #   sub_tasks          计划步骤列表（每步是 dict：id/instruction/tool_hint/depends_on/status/result）
    #   current_task_index 当前执行到第几步（下标，与 sub_tasks 对齐）
    #   task_results       已完成步骤的结果流水（审计与 prompt 展示用）
    sub_tasks: list                 # 任务拆解后的子任务列表
    current_task_index: int         # 当前执行到第几个子任务
    task_results: list              # 各子任务的结果
    plan_rationale: str             # 计划总体思路（为什么要这么拆）；空 = 本轮没规划
    plan_replan_count: int          # 本轮已经重规划过几次（上限由配置控制）

    final_answer: str               # 最终答案，answer 节点填写
    llm_summary: str                # 一句话摘要，用于向量入库
    summary_embedding: list         # 摘要向量

    need_tool_call: bool            # True = 本轮要去 act 节点调工具
    finished: bool                  # True = 流程结束，走 answer/save
    error: str                      # 错误码，如 prompt_injection_blocked
    prepared: bool                  # True = prepare 节点已跑过，不重复召回


# 【什么】同一份状态的数据类版本（非 LangGraph 用）
# 【作用】节点内点属性操作更顺手，入口 from_dict 转进来、出口 to_dict 转回去
@dataclass
class AgentState:
    """Agent 状态数据类（便于直接操作，非 LangGraph 用）。"""
    session_id: str = ""                                       # 会话 ID，也是断点续跑的 thread_id
    user_id: str = ""                                          # 用户标识
    user_query: str = ""                                       # 本轮提问（prepare 清洗后的文本）

    principal: Any = None                                      # 权限载体 Principal；None = 未启用权限

    messages: list = field(default_factory=list)               # 对话消息列表
    rag_context: str = ""                                      # RAG 上下文文本
    rag_sources: list = field(default_factory=list)            # RAG 命中来源列表
    active_skill: str = ""                                     # 命中的技能名，未命中为空串
    skill_context: str = ""                                    # 技能 SOP 上下文
    # 本轮工具名单：由 AgentRouter 在第 1 轮算出，之后沿用（避免每轮重算 embedding）
    routed_tools: list = field(default_factory=list)
    memory_context: str = ""                                   # 记忆上下文文本
    long_term_memory: list = field(default_factory=list)       # 长期记忆条目
    short_term_memory: list = field(default_factory=list)      # 短期记忆条目
    memory_compressed: bool = False                            # True = 本轮记忆已压缩
    memory_low_confidence: bool = False                        # True = 长期记忆召回低置信

    iteration: int = 0                                         # 当前 ReAct 轮次，从 1 起算
    max_iterations: int = 10                                   # 轮次上限 10，达到即强制收尾
    reasoning: str = ""                                        # 本轮 LLM 思考文本
    action: dict = field(default_factory=dict)                 # 待执行动作 {name, arguments, id}
    observation: str = ""                                      # 上一轮工具执行结果
    tool_call_logs: list = field(default_factory=list)         # 工具流水，save 节点落表
    tool_calls_history: list = field(default_factory=list)     # 工具历史，用于重复调用检测

    sub_tasks: list = field(default_factory=list)              # 拆解出的子任务（计划步骤）
    current_task_index: int = 0                                # 当前子任务下标
    task_results: list = field(default_factory=list)           # 各子任务的结果
    plan_rationale: str = ""                                   # 计划总体思路；空 = 本轮没规划
    plan_replan_count: int = 0                                 # 本轮已重规划次数

    final_answer: str = ""                                     # 最终答案
    llm_summary: str = ""                                      # 一句话摘要，用于向量入库
    summary_embedding: list = field(default_factory=list)      # 摘要向量

    need_tool_call: bool = False                               # True = 走 act 节点调工具
    finished: bool = False                                     # True = 流程结束
    error: str = ""                                            # 错误码，如 prompt_injection_blocked
    prepared: bool = False                                     # True = prepare 已跑过，不重复召回

    # 【什么】dataclass → dict 的序列化出口
    # 【作用】LangGraph 只认 dict，节点返回值必须先转成 dict 才能合并回全局状态
    def to_dict(self) -> AgentStateDict:
        """
        把 dataclass 对象转回普通 dict，交给 LangGraph。

        为什么需要这一步：
        LangGraph 的节点函数签名是 (state: dict) -> dict，框架只认 dict，
        不认我们自定义的 dataclass 对象。所以每个节点在调用完业务逻辑
        （用 dataclass 方便点属性操作）后，必须用 to_dict() 把改动转回
        dict 形式返回，LangGraph 才会把这些字段合并回全局状态。

        调用位置：graph.py 的 _prepare_node（第72行）—— prepare_state 返回
        的是 AgentState 对象，必须 to_dict() 后才能作为节点返回值。
        类比 JS：class 实例 → plain object 的序列化，方便框架统一处理。
        """
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_query": self.user_query,
            "principal": self.principal,
            "messages": self.messages,
            "rag_context": self.rag_context,
            "rag_sources": self.rag_sources,
            "active_skill": self.active_skill,
            "skill_context": self.skill_context,
            "routed_tools": self.routed_tools,
            "memory_context": self.memory_context,
            "long_term_memory": self.long_term_memory,
            "short_term_memory": self.short_term_memory,
            "memory_compressed": self.memory_compressed,
            "memory_low_confidence": self.memory_low_confidence,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "reasoning": self.reasoning,
            "action": self.action,
            "observation": self.observation,
            "tool_call_logs": self.tool_call_logs,
            "tool_calls_history": self.tool_calls_history,
            "sub_tasks": self.sub_tasks,
            "current_task_index": self.current_task_index,
            "task_results": self.task_results,
            "plan_rationale": self.plan_rationale,
            "plan_replan_count": self.plan_replan_count,
            "final_answer": self.final_answer,
            "llm_summary": self.llm_summary,
            "summary_embedding": self.summary_embedding,
            "need_tool_call": self.need_tool_call,
            "finished": self.finished,
            "error": self.error,
            "prepared": self.prepared,
        }

    # 【什么】dict → dataclass 的反序列化入口（to_dict 的逆操作）
    # 【作用】框架给的是 dict，业务里用 dataclass，缺字段按默认值补齐不报错
    @classmethod
    def  from_dict(cls, d: dict) -> "AgentState":
        """
        从 dict 恢复成 dataclass 对象（to_dict 的逆操作）。

        为什么需要：LangGraph 传给节点的 state 永远是 dict，
        但业务逻辑里用 dataclass（点属性访问、有默认值、不易写错 key）更方便。
        所以每个节点入口先 from_dict(state) 把 dict 转成对象，
        业务操作完再 to_dict() 返回给框架。

        注意：每个字段都用 d.get(key, 默认值)，即使 dict 里缺某个字段
        也不会报错（LangGraph 状态是增量合并的，某轮可能只传了部分字段）。
        类比 JS：plain object → class 实例的反序列化。
        """
        return cls(
            session_id=d.get("session_id", ""),
            user_id=d.get("user_id", ""),
            user_query=d.get("user_query", ""),
            principal=d.get("principal"),
            messages=d.get("messages", []),
            rag_context=d.get("rag_context", ""),
            rag_sources=d.get("rag_sources", []),
            active_skill=d.get("active_skill", ""),
            skill_context=d.get("skill_context", ""),
            routed_tools=d.get("routed_tools", []),
            memory_context=d.get("memory_context", ""),
            long_term_memory=d.get("long_term_memory", []),
            short_term_memory=d.get("short_term_memory", []),
            memory_compressed=d.get("memory_compressed", False),
            memory_low_confidence=d.get("memory_low_confidence", False),
            iteration=d.get("iteration", 0),
            max_iterations=d.get("max_iterations", 10),      # 缺省 10：与 dataclass 默认值保持一致
            reasoning=d.get("reasoning", ""),
            action=d.get("action", {}),
            observation=d.get("observation", ""),
            tool_call_logs=d.get("tool_call_logs", []),
            tool_calls_history=d.get("tool_calls_history", []),
            sub_tasks=d.get("sub_tasks", []),
            current_task_index=d.get("current_task_index", 0),
            task_results=d.get("task_results", []),
            plan_rationale=d.get("plan_rationale", ""),
            plan_replan_count=int(d.get("plan_replan_count", 0) or 0),
            final_answer=d.get("final_answer", ""),
            llm_summary=d.get("llm_summary", ""),
            summary_embedding=d.get("summary_embedding", []),
            need_tool_call=d.get("need_tool_call", False),
            finished=d.get("finished", False),
            error=d.get("error", ""),
            prepared=d.get("prepared", False),
        )
