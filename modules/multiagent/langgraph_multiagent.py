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

import asyncio  # 异步原语：异步节点、异步运行入口都用它
from typing import Annotated, TypedDict, Literal, Optional  # 类型注解工具：TypedDict 定义共享状态，Literal 限定状态枚举
from dataclasses import dataclass, field  # 数据类工具（本文件暂未用，预留）
from langgraph.graph import StateGraph, END  # LangGraph 核心：状态图与终点标记
from langgraph.checkpoint.memory import MemorySaver  # 内存检查点：支持断点续跑


# =============================================================================
# 1. 定义 State（共享状态）
# =============================================================================

class TeamState(TypedDict):
    """多 Agent 团队的共享状态"""
    # 用户输入
    query: str  # 用户原始问题：整个图的唯一输入源头

    # 主管拆的任务
    subtasks: list[dict]  # [{id, assignee, instruction, depends_on, status, result, error}]

    # 当前执行到哪个任务
    current_task_id: str | None  # 当前正在执行的任务 id（可能为 None：还没开始或已结束）

    # 各 Worker 的产出
    worker_outputs: dict[str, str]  # {task_id: content}：Worker 产出的统一存放处

    # 最终答案
    final_answer: str | None  # 汇总/撰写完成后的最终答案（None 表示还没产出）

    # 是否需要审核
    needs_review: bool  # 开关：True 走审核节点，False 直接结束

    # 审核意见
    review_feedback: str | None  # 审核员的反馈意见

    # 执行状态
    status: Literal["planning", "dispatching", "synthesizing", "reviewing", "done", "error"]  # 状态机：规划/派发/汇总/审核/完成/出错

    # 错误信息
    error: str | None  # 出错时的错误描述

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
        self._messages: list[dict] = []  # 消息仓库：所有消息都进这个列表
        self._lock = asyncio.Lock()  # 异步锁：并发收发消息时保护列表一致性

    async def send(self, sender: str, receiver: str, kind: str, content: str):  # sender：发送者；receiver：接收者（"*" 表示广播）；kind：消息类型；content：消息正文
        """发送消息"""
        async with self._lock:  # 持锁写：防并发写乱序
            self._messages.append({
                "sender": sender,  # 谁发的
                "receiver": receiver,  # 发给谁
                "kind": kind,  # finding / clarify / objection / delegate
                "content": content,  # 正文
                "timestamp": asyncio.get_event_loop().time()  # 时间戳：用事件循环的相对时钟（比 wall clock 更稳）
            })

    async def inbox(self, agent_name: str) -> list[dict]:  # agent_name：要收信的 Agent 名
        """接收消息（只拿发给自己的）"""
        async with self._lock:  # 持锁读：防读到一半被写线程改了
            return [
                m for m in self._messages
                if m["receiver"] in (agent_name, "*") and m["sender"] != agent_name  # 过滤：发给自己的或广播，且不回收自己发的
            ]

    async def broadcast(self, sender: str, kind: str, content: str):  # sender：广播发起者；kind：消息类型；content：广播内容
        """广播消息（发给所有人）"""
        await self.send(sender, "*", kind, content)  # 本质是 receiver="*" 的 send


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


def create_worker_subgraph(worker_type: str):  # worker_type：Worker 类型名（researcher/analyst 等）
    """创建 Worker 的子图"""

    def worker_prepare(state: WorkerState) -> dict:  # state：Worker 的独立状态
        """Worker 准备阶段：组装上下文"""
        task = state["task"]  # 取当前任务
        context = state.get("context", "")  # 取上游上下文（可能为空）

        # 组装提示词
        prompt = f"""你是{worker_type}，负责执行任务。

任务：{task['instruction']}

上下文：
{context}

请完成任务，返回结果。"""  # 三段式提示词：角色定位 + 任务 + 上下文

        return {
            "context": prompt,  # 把组装好的提示词写回 context 字段（execute 阶段要用）
            "steps": state.get("steps", 0) + 1  # 步数 +1
        }

    def worker_execute(state: WorkerState) -> dict:  # state：Worker 的独立状态
        """Worker 执行阶段：调用 LLM"""
        context = state["context"]  # 取上一步组装好的提示词

        # 模拟调用 LLM（实际会调真实的 LLM）
        result = f"[{worker_type}] 完成：{context[:100]}..."  # 截前 100 字做演示

        return {
            "result": result,  # 写回结果
            "steps": state["steps"] + 1  # 步数 +1
        }

    def worker_finalize(state: WorkerState) -> dict:  # state：Worker 的独立状态
        """Worker 收尾阶段：整理结果"""
        return {
            "result": state["result"],  # 透传结果（可在这里做格式化）
            "steps": state["steps"] + 1  # 步数 +1
        }

    # 创建子图
    subgraph = StateGraph(WorkerState)  # 建一张独立的状态图：状态与主图隔离
    subgraph.add_node("prepare", worker_prepare)  # 挂准备节点
    subgraph.add_node("execute", worker_execute)  # 挂执行节点
    subgraph.add_node("finalize", worker_finalize)  # 挂收尾节点

    subgraph.set_entry_point("prepare")  # 入口：先准备
    subgraph.add_edge("prepare", "execute")  # 准备完去执行
    subgraph.add_edge("execute", "finalize")  # 执行完去收尾
    subgraph.add_edge("finalize", END)  # 收尾完结束子图

    return subgraph.compile()  # 编译成可调用的图


# =============================================================================
# 4. 定义节点函数（Node Functions）
# =============================================================================

def supervisor_node(state: TeamState) -> dict:  # state：主图共享状态
    """主管节点：拆任务、派活。只在第一次执行时拆任务。"""
    query = state["query"]  # 取用户原始问题

    # 如果已经有 subtasks，说明已经拆过了，直接返回当前状态
    if state["subtasks"]:
        return {}  # 空 dict：LangGraph 里表示"不更新任何字段"，直接透传当前状态

    # 模拟主管拆任务（实际会调 LLM）
    # 这里简化为：根据关键词拆成查数据 + 算数据 + 写报告
    subtasks = []  # 拆出来的子任务清单

    if "营收" in query or "数据" in query:  # 命中查数关键词
        subtasks.append({
            "id": "t1",  # 任务 id
            "assignee": "researcher",  # 派给研究员
            "instruction": f"查询相关数据：{query}",  # 任务指令
            "depends_on": [],  # 无依赖：最先跑
            "status": "pending",  # 待执行
            "result": None,  # 结果还没出来
            "error": None  # 错误还没有
        })

    if "算" in query or "环比" in query or "分析" in query:  # 命中计算关键词
        subtasks.append({
            "id": "t2",  # 任务 id
            "assignee": "analyst",  # 派给分析师
            "instruction": f"分析计算：{query}",  # 任务指令
            "depends_on": ["t1"] if subtasks else [],  # 有查数任务就依赖它，没有就无依赖
            "status": "pending",  # 待执行
            "result": None,  # 结果还没出来
            "error": None  # 错误还没有
        })

    if "报告" in query or "写" in query or "总结" in query:  # 命中写报告关键词
        subtasks.append({
            "id": "t3",  # 任务 id
            "assignee": "writer",  # 派给撰写员
            "instruction": f"撰写报告：{query}",  # 任务指令
            "depends_on": ["t2"] if len(subtasks) > 1 else ["t1"] if subtasks else [],  # 优先依赖分析，其次依赖查数，都没有就无依赖
            "status": "pending",  # 待执行
            "result": None,  # 结果还没出来
            "error": None  # 错误还没有
        })

    # 如果没有拆出任务，直接回答
    if not subtasks:
        return {
            "final_answer": f"直接回答：{query}",  # 不拆任务直接答
            "status": "done",  # 状态置为完成
            "subtasks": [],  # 空任务列表
            "stats": {"total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0, "total_tokens": 0}  # 统计清零
        }

    return {
        "subtasks": subtasks,  # 写回拆好的任务清单
        "status": "dispatching",  # 状态切到派发中
        "current_task_id": None,  # 还没开始执行任何任务
        "stats": {"total_tasks": len(subtasks), "completed_tasks": 0, "failed_tasks": 0, "total_tokens": 0}  # 初始化统计
    }


def researcher_node(state: TeamState) -> dict:  # state：主图共享状态
    """研究员 Worker：查资料（同步版本）"""
    subtasks = state["subtasks"]  # 取任务清单
    worker_outputs = state["worker_outputs"]  # 取已有产出

    # 找到第一个 pending 且依赖已满足的任务
    task = None  # 目标任务：还没找到
    for t in subtasks:
        if t["status"] == "pending" and t["assignee"] == "researcher":  # 待执行且派给研究员
            deps_met = all(dep in worker_outputs for dep in t["depends_on"])  # 依赖的上游任务都有产出了吗
            if deps_met:
                task = t  # 找到可执行任务
                break  # 只取第一个，跳出循环

    if task is None:
        return {"worker_outputs": worker_outputs, "subtasks": subtasks}  # 没有可执行任务：原样透传（LangGraph 会拿去继续路由）

    task_id = task["id"]  # 记下任务 id

    # 创建 Worker 子图
    worker_graph = create_worker_subgraph("researcher")  # 建研究员专属子图

    # 组装上下文
    context = "\n".join(
        f"上游产出 {dep}: {worker_outputs.get(dep, '')[:200]}"  # 每个上游产出截前 200 字
        for dep in task["depends_on"]  # 遍历依赖的上游任务 id
    )

    # 执行 Worker 子图（同步）
    worker_state: WorkerState = {  # 组装子图初始状态
        "task": task,  # 当前任务
        "context": context,  # 上游上下文
        "result": None,  # 结果还没出来
        "error": None,  # 错误还没有
        "steps": 0  # 步数从 0 计
    }

    try:
        result = worker_graph.invoke(worker_state)  # 同步调用子图
        output = result["result"]  # 取子图最终产出
        task["status"] = "completed"  # 标完成
        task["result"] = output  # 结果写回任务
    except Exception as e:
        task["status"] = "failed"  # 标失败
        task["error"] = str(e)  # 记错误
        output = f"[研究员] 执行失败：{e}"  # 产出降级为错误文本

    return {
        "worker_outputs": {**worker_outputs, task_id: output},  # 合并新产出进总账
        "subtasks": subtasks,  # 任务状态已就地更新，原样返回
        "stats": {
            **state.get("stats", {}),  # 保留其他统计字段
            "completed_tasks": state.get("stats", {}).get("completed_tasks", 0) + 1  # 完成数 +1
        }
    }


async def researcher_node_async(state: TeamState) -> dict:  # state：主图共享状态
    """研究员 Worker：查资料（异步版本）"""
    subtasks = state["subtasks"]  # 取任务清单
    worker_outputs = state["worker_outputs"]  # 取已有产出

    # 找到第一个 pending 且依赖已满足的任务
    task = None  # 目标任务：还没找到
    for t in subtasks:
        if t["status"] == "pending" and t["assignee"] == "researcher":  # 待执行且派给研究员
            deps_met = all(dep in worker_outputs for dep in t["depends_on"])  # 依赖的上游任务都有产出了吗
            if deps_met:
                task = t  # 找到可执行任务
                break  # 只取第一个，跳出循环

    if task is None:
        return {"worker_outputs": worker_outputs, "subtasks": subtasks}  # 没有可执行任务：原样透传

    task_id = task["id"]  # 记下任务 id

    # 创建 Worker 子图
    worker_graph = create_worker_subgraph("researcher")  # 建研究员专属子图

    # 组装上下文
    context = "\n".join(
        f"上游产出 {dep}: {worker_outputs.get(dep, '')[:200]}"  # 每个上游产出截前 200 字
        for dep in task["depends_on"]  # 遍历依赖的上游任务 id
    )

    # 执行 Worker 子图（异步）
    worker_state: WorkerState = {  # 组装子图初始状态
        "task": task,  # 当前任务
        "context": context,  # 上游上下文
        "result": None,  # 结果还没出来
        "error": None,  # 错误还没有
        "steps": 0  # 步数从 0 计
    }

    try:
        result = await worker_graph.ainvoke(worker_state)  # 异步调用子图：可与其他异步节点并行
        output = result["result"]  # 取子图最终产出
        task["status"] = "completed"  # 标完成
        task["result"] = output  # 结果写回任务
    except Exception as e:
        task["status"] = "failed"  # 标失败
        task["error"] = str(e)  # 记错误
        output = f"[研究员] 执行失败：{e}"  # 产出降级为错误文本

    return {
        "worker_outputs": {**worker_outputs, task_id: output},  # 合并新产出进总账
        "subtasks": subtasks,  # 任务状态已就地更新
        "stats": {
            **state.get("stats", {}),  # 保留其他统计字段
            "completed_tasks": state.get("stats", {}).get("completed_tasks", 0) + 1  # 完成数 +1
        }
    }


def analyst_node(state: TeamState) -> dict:  # state：主图共享状态
    """分析师 Worker：算数据（同步版本）"""
    subtasks = state["subtasks"]  # 取任务清单
    worker_outputs = state["worker_outputs"]  # 取已有产出

    # 找到第一个 pending 且依赖已满足的 analyst 任务
    task = None  # 目标任务：还没找到
    for t in subtasks:
        if t["status"] == "pending" and t["assignee"] == "analyst":  # 待执行且派给分析师
            deps_met = all(dep in worker_outputs for dep in t["depends_on"])  # 依赖都满足了吗
            if deps_met:
                task = t  # 找到可执行任务
                break  # 只取第一个，跳出循环

    if task is None:
        return {"worker_outputs": worker_outputs, "subtasks": subtasks}  # 没有可执行任务：原样透传

    task_id = task["id"]  # 记下任务 id

    # 获取上游产出
    upstream = "\n".join(
        worker_outputs.get(dep, "") for dep in task["depends_on"]  # 把所有上游产出拼起来
    )

    # 模拟分析计算（同步）
    result = f"[分析师] 分析完成：基于上游数据 {upstream[:100]}... → 计算结果 YYY"  # 演示用：截前 100 字

    task["status"] = "completed"  # 标完成
    task["result"] = result  # 结果写回任务

    return {
        "worker_outputs": {**worker_outputs, task_id: result},  # 合并新产出进总账
        "subtasks": subtasks,  # 任务状态已就地更新
    }


def writer_node(state: TeamState) -> dict:  # state：主图共享状态
    """撰写员 Worker：写报告（同步版本）"""
    subtasks = state["subtasks"]  # 取任务清单
    worker_outputs = state["worker_outputs"]  # 取已有产出

    # 找到第一个 pending 且依赖已满足的 writer 任务
    task = None  # 目标任务：还没找到
    for t in subtasks:
        if t["status"] == "pending" and t["assignee"] == "writer":  # 待执行且派给撰写员
            deps_met = all(dep in worker_outputs for dep in t["depends_on"])  # 依赖都满足了吗
            if deps_met:
                task = t  # 找到可执行任务
                break  # 只取第一个，跳出循环

    if task is None:
        return {"worker_outputs": worker_outputs, "subtasks": subtasks}  # 没有可执行任务：原样透传

    task_id = task["id"]  # 记下任务 id

    upstream = "\n".join(
        worker_outputs.get(dep, "") for dep in task["depends_on"]  # 把所有上游产出拼起来
    )

    # 模拟写报告（同步）
    result = f"[撰写员] 报告完成：综合所有数据 → 最终报告 ZZZ"  # 演示用：固定文案

    task["status"] = "completed"  # 标完成
    task["result"] = result  # 结果写回任务

    return {
        "worker_outputs": {**worker_outputs, task_id: result},  # 合并新产出进总账
        "subtasks": subtasks,  # 任务状态已就地更新
        "final_answer": result  # 写报告后产出最终答案
    }


def synthesizer_node(state: TeamState) -> dict:  # state：主图共享状态
    """汇总节点：主管汇总所有产出"""
    worker_outputs = state["worker_outputs"]  # 取已有产出
    subtasks = state["subtasks"]  # 取任务清单

    # 如果所有任务都完成了，汇总答案
    all_completed = all(t["status"] in ("completed", "failed") for t in subtasks)  # 全部任务都到终态（完成或失败）了吗

    if all_completed and state["final_answer"] is None:
        # 没有 writer 节点时，由主管汇总
        answer = "【主管汇总】\n" + "\n".join(
            f"- {t['assignee']}: {t.get('result', t.get('error', '无结果'))[:200]}"  # 每个任务的结果/错误截前 200 字
            for t in subtasks
        )
        return {
            "final_answer": answer,  # 写入最终答案
            "status": "reviewing" if state["needs_review"] else "done"  # 需要审核就去审核，否则直接完成
        }

    return {
        "status": "reviewing" if state["needs_review"] else "done"  # 已有最终答案（writer 已写过）：只切状态
    }


def reviewer_node(state: TeamState) -> dict:  # state：主图共享状态
    """审核员节点：挑毛病"""
    final_answer = state["final_answer"]  # 取最终答案（本演示中未实际使用）

    # 模拟审核
    feedback = f"[审核员] 审核完成：答案基本正确，建议补充数据来源"  # 演示用：固定反馈

    return {
        "review_feedback": feedback,  # 写入审核意见
        "status": "done"  # 审核完即结束
    }


# =============================================================================
# 5. 定义路由函数（Conditional Edges）
# =============================================================================

def route_after_supervisor(state: TeamState) -> str:  # state：主图共享状态
    """主管拆完任务后，决定下一个节点"""
    if state["status"] == "done":
        return END  # 状态已经是 done：直接结束图（无需拆任务、直接回答的情况）

    # 找到下一个 pending 且依赖已满足的任务
    subtasks = state["subtasks"]  # 取任务清单
    worker_outputs = state["worker_outputs"]  # 取已有产出

    for task in subtasks:
        if task["status"] == "pending":  # 找待执行的任务
            # 检查依赖是否都完成了
            deps_met = all(
                dep in worker_outputs for dep in task["depends_on"]  # 所有依赖的 task_id 都在 worker_outputs 账本上
            )
            if deps_met:
                # 根据 assignee 路由到不同 Worker
                return task["assignee"]  # 返回值要与 add_conditional_edges 的 path_map 键对应

    # 没有可执行的任务了，去汇总
    return "synthesizer"  # 所有任务已执行完毕或无法执行：进入汇总节点


def route_after_worker(state: TeamState) -> str:  # state：主图共享状态
    """Worker 完成后，决定下一个节点"""
    # 回到 supervisor，让主管决定下一个任务
    return "supervisor"  # 固定回到 supervisor：主管节点负责调度下一条任务


def route_after_synthesizer(state: TeamState) -> str:  # state：主图共享状态
    """汇总后，决定是否需要审核"""
    if state["needs_review"]:
        return "reviewer"  # 需要审核：走审核节点
    return END  # 不需要审核：直接结束图


def route_after_reviewer(state: TeamState) -> str:  # state：主图共享状态
    """审核完成后结束"""
    return END  # 审核完直接结束


# =============================================================================
# 6. 构建 LangGraph
# =============================================================================

def build_multiagent_graph():
    """构建多 Agent 的 LangGraph"""

    # 创建主图
    workflow = StateGraph(TeamState)  # 以 TeamState 为共享状态建图

    # 添加节点
    workflow.add_node("supervisor", supervisor_node)  # 主管节点：拆任务
    workflow.add_node("researcher", researcher_node)  # 研究员节点：查数据
    workflow.add_node("analyst", analyst_node)  # 分析师节点：算数据
    workflow.add_node("writer", writer_node)  # 撰写员节点：写报告
    workflow.add_node("synthesizer", synthesizer_node)  # 汇总节点：主管汇总
    workflow.add_node("reviewer", reviewer_node)  # 审核员节点：挑毛病

    # 设置入口
    workflow.set_entry_point("supervisor")  # 图从主管开始

    # 添加条件边：主管拆完任务后的路由
    workflow.add_conditional_edges(
        "supervisor",  # 从 supervisor 节点出发
        route_after_supervisor,  # 路由判断函数
        {
            "researcher": "researcher",  # 返回 researcher 就去 researcher 节点
            "analyst": "analyst",  # 返回 analyst 就去 analyst 节点
            "writer": "writer",  # 返回 writer 就去 writer 节点
            "synthesizer": "synthesizer",  # 返回 synthesizer 就去汇总节点
            END: END  # 返回 END 就结束图
        }
    )

    # Worker 完成后回到主管
    workflow.add_conditional_edges(
        "researcher",  # 从 researcher 节点出发
        route_after_worker,  # 路由判断函数
        {"supervisor": "supervisor"}  # 只回 supervisor
    )
    workflow.add_conditional_edges(
        "analyst",  # 从 analyst 节点出发
        route_after_worker,  # 路由判断函数
        {"supervisor": "supervisor"}  # 只回 supervisor
    )
    workflow.add_conditional_edges(
        "writer",  # 从 writer 节点出发
        route_after_worker,  # 路由判断函数
        {"supervisor": "supervisor"}  # 只回 supervisor
    )

    # 汇总后的路由
    workflow.add_conditional_edges(
        "synthesizer",  # 从 synthesizer 节点出发
        route_after_synthesizer,  # 路由判断函数
        {
            "reviewer": "reviewer",  # 需要审核就去审核员
            END: END  # 不需要审核就结束
        }
    )

    # 审核后结束
    workflow.add_edge("reviewer", END)  # 审核员节点固定出边到 END

    # 编译（带 checkpoint 支持）
    memory = MemorySaver()  # 内存检查点器：支持断点续跑
    app = workflow.compile(checkpointer=memory)  # 编译成可调用图：带 checkpoint 支持

    return app  # 返回编译后的图


# =============================================================================
# 7. 运行示例
# =============================================================================

async def run_multiagent_async(query: str, needs_review: bool = False):  # query：用户问题；needs_review：是否需要审核节点
    """运行多 Agent 团队（异步版本）"""
    app = build_multiagent_graph()  # 建图

    # 初始状态
    initial_state: TeamState = {
        "query": query,  # 用户问题
        "subtasks": [],  # 空任务列表：等主管拆
        "current_task_id": None,  # 还没开始执行
        "worker_outputs": {},  # 空产出账本
        "final_answer": None,  # 还没最终答案
        "needs_review": needs_review,  # 是否走审核节点
        "review_feedback": None,  # 审核意见暂无
        "status": "planning",  # 初始状态：规划中
        "error": None,  # 错误暂无
        "messages": [],  # 空消息队列
        "stats": {"total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0, "total_tokens": 0}  # 统计清零
    }

    # 运行（带 thread_id 支持断点续跑）
    config = {"configurable": {"thread_id": "run_001"}}  # thread_id 是 checkpoint 的 key：同 id 可续跑
    result = await app.ainvoke(initial_state, config)  # 异步调用图

    return result  # 返回最终状态


def run_multiagent(query: str, needs_review: bool = False):  # query：用户问题；needs_review：是否需要审核节点
    """运行多 Agent 团队（同步版本）"""
    app = build_multiagent_graph()  # 建图

    # 初始状态
    initial_state: TeamState = {
        "query": query,  # 用户问题
        "subtasks": [],  # 空任务列表：等主管拆
        "current_task_id": None,  # 还没开始执行
        "worker_outputs": {},  # 空产出账本
        "final_answer": None,  # 还没最终答案
        "needs_review": needs_review,  # 是否走审核节点
        "review_feedback": None,  # 审核意见暂无
        "status": "planning",  # 初始状态：规划中
        "error": None,  # 错误暂无
        "messages": [],  # 空消息队列
        "stats": {"total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0, "total_tokens": 0}  # 统计清零
    }

    # 运行（带 thread_id 支持断点续跑）
    config = {"configurable": {"thread_id": "run_001"}}  # thread_id 是 checkpoint 的 key：同 id 可续跑
    result = app.invoke(initial_state, config)  # 同步调用图

    return result  # 返回最终状态


# =============================================================================
# 8. 测试
# =============================================================================

if __name__ == "__main__":
    # 测试 1：简单查询（不需要拆任务）
    print("=" * 60)  # 分隔线
    print("测试 1：简单查询")  # 测试标题
    print("=" * 60)  # 分隔线
    result = run_multiagent("你好")  # 跑同步版本：闲聊类问题，主管直接答
    print(f"答案：{result['final_answer']}")  # 打印最终答案
    print(f"状态：{result['status']}")  # 打印执行状态
    print(f"统计：{result['stats']}")  # 打印执行统计
    print()  # 空行分隔

    # 测试 2：需要拆任务的查询
    print("=" * 60)  # 分隔线
    print("测试 2：复杂查询（查数据 + 分析 + 写报告）")  # 测试标题
    print("=" * 60)  # 分隔线
    result = run_multiagent("查一下 Q3 营收，算一下环比，写个报告", needs_review=True)  # 跑同步版本：命中全部三个关键词，且走审核
    print(f"答案：{result['final_answer']}")  # 打印最终答案
    print(f"审核意见：{result['review_feedback']}")  # 打印审核反馈
    print(f"状态：{result['status']}")  # 打印执行状态
    print(f"统计：{result['stats']}")  # 打印执行统计
    print()  # 空行分隔

    # 测试 3：异步版本
    print("=" * 60)  # 分隔线
    print("测试 3：异步版本")  # 测试标题
    print("=" * 60)  # 分隔线
    result = asyncio.run(run_multiagent_async("查一下 Q4 数据，分析一下"))  # 跑异步版本：asyncio.run 起事件循环
    print(f"答案：{result['final_answer']}")  # 打印最终答案
    print(f"状态：{result['status']}")  # 打印执行状态
    print()  # 空行分隔

    # 测试 4：断点续跑
    print("=" * 60)  # 分隔线
    print("测试 4：断点续跑（用相同 thread_id）")  # 测试标题
    print("=" * 60)  # 分隔线
    app = build_multiagent_graph()  # 重建图（checkpoint 是共享的 MemorySaver 不行，这里新建实例只是演示 API 用法）
    config = {"configurable": {"thread_id": "run_001"}}  # 用同一个 thread_id：对应前面跑过的检查点

    # 获取当前状态
    state = app.get_state(config)  # 从 checkpoint 拉回该 thread 的最新状态
    print(f"当前状态：{state.values.get('status', 'N/A')}")  # 打印状态字段
    print(f"已完成任务：{len([t for t in state.values.get('subtasks', []) if t['status'] == 'completed'])}")  # 统计已完成任务数
    print(f"失败任务：{len([t for t in state.values.get('subtasks', []) if t['status'] == 'failed'])}")  # 统计失败任务数
