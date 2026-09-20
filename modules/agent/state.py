"""
Agent 状态定义模块
==================
LangGraph 状态化 Agent 的核心状态结构。
支持任务拆解、断点续跑、多轮 ReAct 循环状态追踪。
"""
from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass, field
from langgraph.graph import MessagesState


class AgentStateDict(TypedDict, total=False):
    """
    LangGraph Agent 全局状态（TypedDict，支持部分字段）。
    贯穿 ReAct 循环、工具调度、记忆管理全链路。
    """
    session_id: str
    user_id: str
    user_query: str

    principal: Any

    messages: list

    rag_context: str
    rag_sources: list

    active_skill: str
    skill_context: str

    memory_context: str
    long_term_memory: list
    short_term_memory: list
    memory_compressed: bool

    iteration: int
    max_iterations: int
    reasoning: str
    action: dict
    observation: str
    tool_call_logs: list
    tool_calls_history: list

    sub_tasks: list
    current_task_index: int
    task_results: list

    final_answer: str
    llm_summary: str
    summary_embedding: list

    need_tool_call: bool
    finished: bool
    error: str
    prepared: bool


@dataclass
class AgentState:
    """Agent 状态数据类（便于直接操作，非 LangGraph 用）。"""
    session_id: str = ""
    user_id: str = ""
    user_query: str = ""

    principal: Any = None

    messages: list = field(default_factory=list)
    rag_context: str = ""
    rag_sources: list = field(default_factory=list)
    active_skill: str = ""
    skill_context: str = ""
    memory_context: str = ""
    long_term_memory: list = field(default_factory=list)
    short_term_memory: list = field(default_factory=list)
    memory_compressed: bool = False

    iteration: int = 0
    max_iterations: int = 10
    reasoning: str = ""
    action: dict = field(default_factory=dict)
    observation: str = ""
    tool_call_logs: list = field(default_factory=list)
    tool_calls_history: list = field(default_factory=list)

    sub_tasks: list = field(default_factory=list)
    current_task_index: int = 0
    task_results: list = field(default_factory=list)

    final_answer: str = ""
    llm_summary: str = ""
    summary_embedding: list = field(default_factory=list)

    need_tool_call: bool = False
    finished: bool = False
    error: str = ""
    prepared: bool = False

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
            "memory_context": self.memory_context,
            "long_term_memory": self.long_term_memory,
            "short_term_memory": self.short_term_memory,
            "memory_compressed": self.memory_compressed,
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
            "final_answer": self.final_answer,
            "llm_summary": self.llm_summary,
            "summary_embedding": self.summary_embedding,
            "need_tool_call": self.need_tool_call,
            "finished": self.finished,
            "error": self.error,
            "prepared": self.prepared,
        }

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
            memory_context=d.get("memory_context", ""),
            long_term_memory=d.get("long_term_memory", []),
            short_term_memory=d.get("short_term_memory", []),
            memory_compressed=d.get("memory_compressed", False),
            iteration=d.get("iteration", 0),
            max_iterations=d.get("max_iterations", 10),
            reasoning=d.get("reasoning", ""),
            action=d.get("action", {}),
            observation=d.get("observation", ""),
            tool_call_logs=d.get("tool_call_logs", []),
            tool_calls_history=d.get("tool_calls_history", []),
            sub_tasks=d.get("sub_tasks", []),
            current_task_index=d.get("current_task_index", 0),
            task_results=d.get("task_results", []),
            final_answer=d.get("final_answer", ""),
            llm_summary=d.get("llm_summary", ""),
            summary_embedding=d.get("summary_embedding", []),
            need_tool_call=d.get("need_tool_call", False),
            finished=d.get("finished", False),
            error=d.get("error", ""),
            prepared=d.get("prepared", False),
        )
