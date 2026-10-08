"""
LangGraph 多 Agent 实现（完整版）
================================
用 LangGraph 的 StateGraph + 子图（Subgraph）实现多 Agent 协作。

与手写版本的对比：
- 手写版本：纯 Python 类 + ThreadPoolExecutor，严格上下文隔离
- LangGraph 版本：StateGraph + 节点 + 边，共享 State，框架管理状态流转

本文件特点：
1. 使用 LangGraph 的 StateGraph 定义主图
2. 使用子图（Subgraph）定义每个 Worker
3. 使用 add_conditional_edges 实现依赖路由
4. 使用 checkpoint 支持断点续跑
5. 支持异步调用子 Agent（async/await）
6. 支持同步调用子 Agent
7. 消息总线通信机制
8. 上下文边界控制
9. 完整的错误处理和降级策略
"""

import asyncio
from typing import Annotated, TypedDict, Literal, Optional
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# =============================================================================
# 1. 定义 State（共享状态）
# =============================================================================

class TeamState(TypedDict):
    """多 Agent 团队的共享状态"""
    # 用户输入
    query: str
    
    # 主管拆的任务
    subtasks: list[dict]  # [{id, assignee, instruction, depends_on, status, result, error}]
    
    # 当前执行到哪个任务
    current_task_id: str | None
    
    # 各 Worker 的产出
    worker_outputs: dict[str, str]  # {task_id: content}
    
    # 最终答案
    final_answer: str | None
    
    # 是否需要审核
    needs_review: bool
    
    # 审核意见
    review_feedback: str | None
    
    # 执行状态
    status: Literal["planning", "dispatching", "synthesizing", "reviewing", "done", "error"]
    
    # 错误信息
    error: str | None
    
    # 消息队列（Agent 间通信）
    messages: list[dict]  # [{sender, receiver, kind, content, timestamp}]
    
    # 执行统计
    stats: dict  # {total_tasks, completed_tasks, failed_tasks, total_tokens}


# =============================================================================
# 2. 消息总线（Message Bus）
# =============================================================================

class MessageBus:
    """Agent 间通信的消息总线"""
    
    def __init__(self):
        self._messages: list[dict] = []
        self._lock = asyncio.Lock()
    
    async def send(self, sender: str, receiver: str, kind: str, content: str):
        """发送消息"""
        async with self._lock:
            self._messages.append({
                "sender": sender,
                "receiver": receiver,
                "kind": kind,  # finding / clarify / objection / delegate
                "content": content,
                "timestamp": asyncio.get_event_loop().time()
            })
    
    async def inbox(self, agent_name: str) -> list[dict]:
        """接收消息（只拿发给自己的）"""
        async with self._lock:
            return [
                m for m in self._messages
                if m["receiver"] in (agent_name, "*") and m["sender"] != agent_name
            ]
    
    async def broadcast(self, sender: str, kind: str, content: str):
        """广播消息（发给所有人）"""
        await self.send(sender, "*", kind, content)


# =============================================================================
# 3. 定义子图（Subgraph）- Worker 的独立图
# =============================================================================

class WorkerState(TypedDict):
    """Worker 的独立状态（不与主图共享）"""
    task: dict  # 当前任务
    context: str  # 上下文（上游产出、消息等）
    result: str | None  # 执行结果
    error: str | None  # 错误信息
    steps: int  # 执行步数


def create_worker_subgraph(worker_type: str):
    """创建 Worker 的子图"""
    
    def worker_prepare(state: WorkerState) -> dict:
        """Worker 准备阶段：组装上下文"""
        task = state["task"]
        context = state.get("context", "")
        
        # 组装提示词
        prompt = f"""你是{worker_type}，负责执行任务。

任务：{task['instruction']}

上下文：
{context}

请完成任务，返回结果。"""
        
        return {
            "context": prompt,
            "steps": state.get("steps", 0) + 1
        }
    
    def worker_execute(state: WorkerState) -> dict:
        """Worker 执行阶段：调用 LLM"""
        context = state["context"]
        
        # 模拟调用 LLM（实际会调真实的 LLM）
        result = f"[{worker_type}] 完成：{context[:100]}..."
        
        return {
            "result": result,
            "steps": state["steps"] + 1
        }
    
    def worker_finalize(state: WorkerState) -> dict:
        """Worker 收尾阶段：整理结果"""
        return {
            "result": state["result"],
            "steps": state["steps"] + 1
        }
    
    # 创建子图
    subgraph = StateGraph(WorkerState)
    subgraph.add_node("prepare", worker_prepare)
    subgraph.add_node("execute", worker_execute)
    subgraph.add_node("finalize", worker_finalize)
    
    subgraph.set_entry_point("prepare")
    subgraph.add_edge("prepare", "execute")
    subgraph.add_edge("execute", "finalize")
    subgraph.add_edge("finalize", END)
    
    return subgraph.compile()


# =============================================================================
# 4. 定义节点函数（Node Functions）
# =============================================================================

def supervisor_node(state: TeamState) -> dict:
    """主管节点：拆任务、派活。只在第一次执行时拆任务。"""
    query = state["query"]
    
    # 如果已经有 subtasks，说明已经拆过了，直接返回当前状态
    if state["subtasks"]:
        return {}
    
    # 模拟主管拆任务（实际会调 LLM）
    # 这里简化为：根据关键词拆成查数据 + 算数据 + 写报告
    subtasks = []
    
    if "营收" in query or "数据" in query:
        subtasks.append({
            "id": "t1",
            "assignee": "researcher",
            "instruction": f"查询相关数据：{query}",
            "depends_on": [],
            "status": "pending",
            "result": None,
            "error": None
        })
    
    if "算" in query or "环比" in query or "分析" in query:
        subtasks.append({
            "id": "t2",
            "assignee": "analyst",
            "instruction": f"分析计算：{query}",
            "depends_on": ["t1"] if subtasks else [],
            "status": "pending",
            "result": None,
            "error": None
        })
    
    if "报告" in query or "写" in query or "总结" in query:
        subtasks.append({
            "id": "t3",
            "assignee": "writer",
            "instruction": f"撰写报告：{query}",
            "depends_on": ["t2"] if len(subtasks) > 1 else ["t1"] if subtasks else [],
            "status": "pending",
            "result": None,
            "error": None
        })
    
    # 如果没有拆出任务，直接回答
    if not subtasks:
        return {
            "final_answer": f"直接回答：{query}",
            "status": "done",
            "subtasks": [],
            "stats": {"total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0, "total_tokens": 0}
        }
    
    return {
        "subtasks": subtasks,
        "status": "dispatching",
        "current_task_id": None,
        "stats": {"total_tasks": len(subtasks), "completed_tasks": 0, "failed_tasks": 0, "total_tokens": 0}
    }


def researcher_node(state: TeamState) -> dict:
    """研究员 Worker：查资料（同步版本）"""
    subtasks = state["subtasks"]
    worker_outputs = state["worker_outputs"]
    
    # 找到第一个 pending 且依赖已满足的任务
    task = None
    for t in subtasks:
        if t["status"] == "pending" and t["assignee"] == "researcher":
            deps_met = all(dep in worker_outputs for dep in t["depends_on"])
            if deps_met:
                task = t
                break
    
    if task is None:
        return {"worker_outputs": worker_outputs, "subtasks": subtasks}
    
    task_id = task["id"]
    
    # 创建 Worker 子图
    worker_graph = create_worker_subgraph("researcher")
    
    # 组装上下文
    context = "\n".join(
        f"上游产出 {dep}: {worker_outputs.get(dep, '')[:200]}"
        for dep in task["depends_on"]
    )
    
    # 执行 Worker 子图（同步）
    worker_state: WorkerState = {
        "task": task,
        "context": context,
        "result": None,
        "error": None,
        "steps": 0
    }
    
    try:
        result = worker_graph.invoke(worker_state)
        output = result["result"]
        task["status"] = "completed"
        task["result"] = output
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        output = f"[研究员] 执行失败：{e}"
    
    return {
        "worker_outputs": {**worker_outputs, task_id: output},
        "subtasks": subtasks,
        "stats": {
            **state.get("stats", {}),
            "completed_tasks": state.get("stats", {}).get("completed_tasks", 0) + 1
        }
    }


async def researcher_node_async(state: TeamState) -> dict:
    """研究员 Worker：查资料（异步版本）"""
    subtasks = state["subtasks"]
    worker_outputs = state["worker_outputs"]
    
    # 找到第一个 pending 且依赖已满足的任务
    task = None
    for t in subtasks:
        if t["status"] == "pending" and t["assignee"] == "researcher":
            deps_met = all(dep in worker_outputs for dep in t["depends_on"])
            if deps_met:
                task = t
                break
    
    if task is None:
        return {"worker_outputs": worker_outputs, "subtasks": subtasks}
    
    task_id = task["id"]
    
    # 创建 Worker 子图
    worker_graph = create_worker_subgraph("researcher")
    
    # 组装上下文
    context = "\n".join(
        f"上游产出 {dep}: {worker_outputs.get(dep, '')[:200]}"
        for dep in task["depends_on"]
    )
    
    # 执行 Worker 子图（异步）
    worker_state: WorkerState = {
        "task": task,
        "context": context,
        "result": None,
        "error": None,
        "steps": 0
    }
    
    try:
        result = await worker_graph.ainvoke(worker_state)
        output = result["result"]
        task["status"] = "completed"
        task["result"] = output
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        output = f"[研究员] 执行失败：{e}"
    
    return {
        "worker_outputs": {**worker_outputs, task_id: output},
        "subtasks": subtasks,
        "stats": {
            **state.get("stats", {}),
            "completed_tasks": state.get("stats", {}).get("completed_tasks", 0) + 1
        }
    }


def analyst_node(state: TeamState) -> dict:
    """分析师 Worker：算数据（同步版本）"""
    subtasks = state["subtasks"]
    worker_outputs = state["worker_outputs"]
    
    # 找到第一个 pending 且依赖已满足的 analyst 任务
    task = None
    for t in subtasks:
        if t["status"] == "pending" and t["assignee"] == "analyst":
            deps_met = all(dep in worker_outputs for dep in t["depends_on"])
            if deps_met:
                task = t
                break
    
    if task is None:
        return {"worker_outputs": worker_outputs, "subtasks": subtasks}
    
    task_id = task["id"]
    
    # 获取上游产出
    upstream = "\n".join(
        worker_outputs.get(dep, "") for dep in task["depends_on"]
    )
    
    # 模拟分析计算（同步）
    result = f"[分析师] 分析完成：基于上游数据 {upstream[:100]}... → 计算结果 YYY"
    
    task["status"] = "completed"
    task["result"] = result
    
    return {
        "worker_outputs": {**worker_outputs, task_id: result},
        "subtasks": subtasks,
    }


def writer_node(state: TeamState) -> dict:
    """撰写员 Worker：写报告（同步版本）"""
    subtasks = state["subtasks"]
    worker_outputs = state["worker_outputs"]
    
    # 找到第一个 pending 且依赖已满足的 writer 任务
    task = None
    for t in subtasks:
        if t["status"] == "pending" and t["assignee"] == "writer":
            deps_met = all(dep in worker_outputs for dep in t["depends_on"])
            if deps_met:
                task = t
                break
    
    if task is None:
        return {"worker_outputs": worker_outputs, "subtasks": subtasks}
    
    task_id = task["id"]
    
    upstream = "\n".join(
        worker_outputs.get(dep, "") for dep in task["depends_on"]
    )
    
    # 模拟写报告（同步）
    result = f"[撰写员] 报告完成：综合所有数据 → 最终报告 ZZZ"
    
    task["status"] = "completed"
    task["result"] = result
    
    return {
        "worker_outputs": {**worker_outputs, task_id: result},
        "subtasks": subtasks,
        "final_answer": result  # 写报告后产出最终答案
    }


def synthesizer_node(state: TeamState) -> dict:
    """汇总节点：主管汇总所有产出"""
    worker_outputs = state["worker_outputs"]
    subtasks = state["subtasks"]
    
    # 如果所有任务都完成了，汇总答案
    all_completed = all(t["status"] in ("completed", "failed") for t in subtasks)
    
    if all_completed and state["final_answer"] is None:
        # 没有 writer 节点时，由主管汇总
        answer = "【主管汇总】\n" + "\n".join(
            f"- {t['assignee']}: {t.get('result', t.get('error', '无结果'))[:200]}"
            for t in subtasks
        )
        return {
            "final_answer": answer,
            "status": "reviewing" if state["needs_review"] else "done"
        }
    
    return {
        "status": "reviewing" if state["needs_review"] else "done"
    }


def reviewer_node(state: TeamState) -> dict:
    """审核员节点：挑毛病"""
    final_answer = state["final_answer"]
    
    # 模拟审核
    feedback = f"[审核员] 审核完成：答案基本正确，建议补充数据来源"
    
    return {
        "review_feedback": feedback,
        "status": "done"
    }


# =============================================================================
# 5. 定义路由函数（Conditional Edges）
# =============================================================================

def route_after_supervisor(state: TeamState) -> str:
    """主管拆完任务后，决定下一个节点"""
    if state["status"] == "done":
        return END

    # 找到下一个 pending 且依赖已满足的任务
    subtasks = state["subtasks"]
    worker_outputs = state["worker_outputs"]

    for task in subtasks:
        if task["status"] == "pending":
            # 检查依赖是否都完成了
            deps_met = all(
                dep in worker_outputs for dep in task["depends_on"]
            )
            if deps_met:
                # 根据 assignee 路由到不同 Worker
                return task["assignee"]

    # 没有可执行的任务了，去汇总
    return "synthesizer"


def route_after_worker(state: TeamState) -> str:
    """Worker 完成后，决定下一个节点"""
    # 回到 supervisor，让主管决定下一个任务
    return "supervisor"


def route_after_synthesizer(state: TeamState) -> str:
    """汇总后，决定是否需要审核"""
    if state["needs_review"]:
        return "reviewer"
    return END


def route_after_reviewer(state: TeamState) -> str:
    """审核完成后结束"""
    return END


# =============================================================================
# 6. 构建 LangGraph
# =============================================================================

def build_multiagent_graph():
    """构建多 Agent 的 LangGraph"""
    
    # 创建主图
    workflow = StateGraph(TeamState)
    
    # 添加节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("reviewer", reviewer_node)
    
    # 设置入口
    workflow.set_entry_point("supervisor")
    
    # 添加条件边：主管拆完任务后的路由
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "synthesizer": "synthesizer",
            END: END
        }
    )
    
    # Worker 完成后回到主管
    workflow.add_conditional_edges(
        "researcher",
        route_after_worker,
        {"supervisor": "supervisor"}
    )
    workflow.add_conditional_edges(
        "analyst",
        route_after_worker,
        {"supervisor": "supervisor"}
    )
    workflow.add_conditional_edges(
        "writer",
        route_after_worker,
        {"supervisor": "supervisor"}
    )
    
    # 汇总后的路由
    workflow.add_conditional_edges(
        "synthesizer",
        route_after_synthesizer,
        {
            "reviewer": "reviewer",
            END: END
        }
    )
    
    # 审核后结束
    workflow.add_edge("reviewer", END)
    
    # 编译（带 checkpoint 支持）
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# =============================================================================
# 7. 运行示例
# =============================================================================

async def run_multiagent_async(query: str, needs_review: bool = False):
    """运行多 Agent 团队（异步版本）"""
    app = build_multiagent_graph()
    
    # 初始状态
    initial_state: TeamState = {
        "query": query,
        "subtasks": [],
        "current_task_id": None,
        "worker_outputs": {},
        "final_answer": None,
        "needs_review": needs_review,
        "review_feedback": None,
        "status": "planning",
        "error": None,
        "messages": [],
        "stats": {"total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0, "total_tokens": 0}
    }
    
    # 运行（带 thread_id 支持断点续跑）
    config = {"configurable": {"thread_id": "run_001"}}
    result = await app.ainvoke(initial_state, config)
    
    return result


def run_multiagent(query: str, needs_review: bool = False):
    """运行多 Agent 团队（同步版本）"""
    app = build_multiagent_graph()
    
    # 初始状态
    initial_state: TeamState = {
        "query": query,
        "subtasks": [],
        "current_task_id": None,
        "worker_outputs": {},
        "final_answer": None,
        "needs_review": needs_review,
        "review_feedback": None,
        "status": "planning",
        "error": None,
        "messages": [],
        "stats": {"total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0, "total_tokens": 0}
    }
    
    # 运行（带 thread_id 支持断点续跑）
    config = {"configurable": {"thread_id": "run_001"}}
    result = app.invoke(initial_state, config)
    
    return result


# =============================================================================
# 8. 测试
# =============================================================================

if __name__ == "__main__":
    # 测试 1：简单查询（不需要拆任务）
    print("=" * 60)
    print("测试 1：简单查询")
    print("=" * 60)
    result = run_multiagent("你好")
    print(f"答案：{result['final_answer']}")
    print(f"状态：{result['status']}")
    print(f"统计：{result['stats']}")
    print()
    
    # 测试 2：需要拆任务的查询
    print("=" * 60)
    print("测试 2：复杂查询（查数据 + 分析 + 写报告）")
    print("=" * 60)
    result = run_multiagent("查一下 Q3 营收，算一下环比，写个报告", needs_review=True)
    print(f"答案：{result['final_answer']}")
    print(f"审核意见：{result['review_feedback']}")
    print(f"状态：{result['status']}")
    print(f"统计：{result['stats']}")
    print()
    
    # 测试 3：异步版本
    print("=" * 60)
    print("测试 3：异步版本")
    print("=" * 60)
    result = asyncio.run(run_multiagent_async("查一下 Q4 数据，分析一下"))
    print(f"答案：{result['final_answer']}")
    print(f"状态：{result['status']}")
    print()
    
    # 测试 4：断点续跑
    print("=" * 60)
    print("测试 4：断点续跑（用相同 thread_id）")
    print("=" * 60)
    app = build_multiagent_graph()
    config = {"configurable": {"thread_id": "run_001"}}
    
    # 获取当前状态
    state = app.get_state(config)
    print(f"当前状态：{state.values.get('status', 'N/A')}")
    print(f"已完成任务：{len([t for t in state.values.get('subtasks', []) if t['status'] == 'completed'])}")
    print(f"失败任务：{len([t for t in state.values.get('subtasks', []) if t['status'] == 'failed'])}")
