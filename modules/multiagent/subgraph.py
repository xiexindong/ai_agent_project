"""
LangGraph Subgraph 子图委派
============================
回答"绝大多数企业业务场景，为什么优先用 LangGraph 子图做多 Agent 委派协同"：
同进程内，管理图（Manager Graph）的节点里调用编译好的工作图（Worker Graph），
子图有自己的状态 schema，父图通过"状态映射"喂输入、收回输出。

【一句话看破】所谓子图委派 = 管理图的 delegate 节点里调一句
    worker_graph.invoke(child_state)
child_state 由父图状态映射而来，跑完的终态再映射回父图。没有魔法，
就是"函数调用"，只是被调的函数是一张编译好的 LangGraph 图：
换来了子图自己的状态流转、条件分支、以及（接上 checkpointer 后的）断点续跑。

【图结构】
    管理图:  plan → delegate → reconcile → answer → END
    工作图:  execute → (delegate? → execute 循环) → report → END
              └ 工作图自己的 delegate 节点里，再 invoke 更深一层的子图（递归）

【生产四件套（Demo 与生产的分水岭）都在本模块落地】
    ① 委派深度限制  _spawn 里 depth >= max_depth 直接拒绝并记录（防无限套娃）
    ② 子任务超时    delegate 节点用线程池 + future.result(timeout)（防一个慢任务拖死全局）
    ③ 失败降级      超时/异常的子任务记 degraded，最终答案只拼成功的部分并注明降级
    ④ 子结果冲突    reconcile 节点按 group 分组，多数决 + 平票取先完成，冲突留痕

【输入输出对照】query="查营收；算环比"（按 ；拆两个子任务）：
    初始 state = {"query": "查营收；算环比"}
    plan     → subtasks = [{"task_id":"t1","assignee":"worker-1","instruction":"查营收"},
                            {"task_id":"t2","assignee":"worker-2","instruction":"算环比"}]
    delegate → results  = [{"task_id":"t1","content":"[worker-1] 已完成: 查营收", ...},
                            {"task_id":"t2","content":"[worker-2] 已完成: 算环比", ...}]
    reconcile→ 冲突组为空（两组无同组竞争）
    answer   → final_answer = "[worker-1] 已完成: 查营收\n\n[worker-2] 已完成: 算环比"

【与既有链路的关系】
    harness（线程池编排）/ delegation（消息层委派）解决的是"多 Agent 花名册协作"；
    本模块解决的是"图套图"的委派形态——开发成本最低、状态与断点续跑原生支持，
    是面试口径里"绝大多数企业场景的首选"。三套并存，按场景选。
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from modules.common.logger import get_logger

_logger = get_logger("multiagent.subgraph")


# ==================================================================
# 状态 schema：父图与子图各一份（子图委派的"状态映射"就是它们之间的翻译）
# ==================================================================
class ManagerStateDict(TypedDict, total=False):
    """管理图状态：规划结果、子任务产出、冲突、降级、最终答案。"""

    query: str                    # 原始问题
    subtasks: list[dict]          # plan 节点拆出的子任务 [{task_id, assignee, instruction, group, depth}]
    results: list[dict]           # delegate 节点收回的子图产出（成功项）
    degraded: list[dict]          # 超时/异常被降级的子任务 [{task_id, reason}]
    conflicts: list[dict]         # reconcile 检出的冲突 [{group, variants, winner, rule}]
    final_answer: str             # answer 拼装的最终答案
    finished: bool


class WorkerStateDict(TypedDict, total=False):
    """工作图（子图）状态：一次子任务的完整生命周期。"""

    task_id: str                  # 子任务 id（父图映射进来）
    assignee: str                 # 执行者名（子图自己叫什么）
    instruction: str              # 任务指令
    depth: int                    # 当前是第几层子图（管理图的直接子图 = 第 1 层）
    chain: list[str]              # 委派链（防环：链上已有的名字不许再出现）
    context: str                  # 上下文（委派产出会追加到这里，供 execute 二次使用）
    report: str                   # execute 的产出正文（回映射给父图的就是它）
    delegate_request: Optional[dict]   # execute 想委派时填 {to, instruction, reason}
    delegations: list[dict]       # 本子图生命周期内成功转出去的委派记录
    denied: list[dict]            # 被深度/防环闸拦下的委派尝试（含原因）
    iteration: int                # execute↔delegate 循环轮数（防 execute 反复要委派绕圈）


# ==================================================================
# 配置：生产四件套的边界值
# ==================================================================
class SubgraphConfig(BaseModel):
    """子图委派的边界。默认值取"演示够用、失控可兜底"的量级。"""

    max_depth: int = Field(default=2, ge=1, description="子图最多嵌套几层；第 max_depth 层的工作图不许再委派")
    task_timeout: float = Field(default=15.0, gt=0, description="单个子任务（一次子图 invoke）的超时秒数")
    max_subtasks: int = Field(default=5, ge=1, description="单次运行最多拆几个子任务，超出截断")
    worker_max_iterations: int = Field(default=3, ge=1, description="单张工作图内 execute↔delegate 最多循环几轮")
    max_workers: int = Field(default=4, ge=1, description="子任务并行执行的线程数")


# ==================================================================
# 工作图（子图）构建
# ==================================================================
def build_worker_graph(
    name: str,
    execute_fn: Callable[[dict], dict],
    delegate_fn: Optional[Callable[[str, str, int, list[str]], Optional[dict]]] = None,
    max_iterations: int = 3,
):
    """构建一张编译好的工作图：execute → (delegate?) → report。

    【为什么 execute_fn / delegate_fn 都是注入的】
        本模块只管"图套图"的骨架；真正干什么活（调 LLM？调工具？纯模拟？）
        由装配方递进来——测试递确定性函数，生产递 ReActAgent 的执行循环。

    【delegate_fn 签名】(task_id, to_agent, depth, chain) → 记录 dict 或 None
        返回 None 表示被治理拒绝（拒绝原因由调用方记进 state.denied）。
    """

    # 【什么】执行节点：干活，可能提出委派请求
    # 【作用】execute_fn 拿到的输入是整个 worker state（dict），返回更新片段
    def _execute_node(state: dict) -> dict:
        updates = execute_fn(state) or {}
        updates["iteration"] = state.get("iteration", 0) + 1
        return updates

    # 【什么】委派节点：把 execute 提出的请求真正转出去（递归 invoke 更深一层子图）
    # 【作用】调 delegate_fn → 成功则产出并入 context，失败/被拒则记账后继续自己干
    def _delegate_node(state: dict) -> dict:
        req = state.get("delegate_request") or {}
        to_agent = str(req.get("to", ""))
        instruction = str(req.get("instruction", ""))
        depth = int(state.get("depth", 1))
        chain = list(state.get("chain", []) or [])
        rec = delegate_fn(state.get("task_id", ""), to_agent, depth, chain) if delegate_fn else None
        delegations = list(state.get("delegations", []) or [])
        denied = list(state.get("denied", []) or [])
        if rec is None:
            # 【被拒也留痕】"想转但被拦"和"根本没想转"是两件事，排查靠这条记录
            denied.append({"from": name, "to": to_agent, "depth": depth,
                           "reason": "深度/防环治理拒绝"})
            _logger.info(f"子图委派被拒: {name}→{to_agent} depth={depth}")
        else:
            delegations.append(rec)
            # 【嵌套账本向上冒泡】受托方自己转出去的记录（成功与被拒）不能
            #   消失在子图里：父层看不到它们，"为什么没人接"就没法排查
            delegations.extend(rec.pop("nested_delegations", None) or [])
            denied.extend(rec.pop("nested_denied", None) or [])
            _logger.info(f"子图委派成功: {name}→{to_agent} depth={depth}")
        return {
            "delegate_request": None,                      # 请求消费掉，防止循环回来重复委派
            "delegations": delegations,
            "denied": denied,
            # 委派产出并入上下文：execute_fn 下一轮能看到，用自己的逻辑消化它
            "context": f"{state.get('context', '')}\n[委派产出:{to_agent}] {rec.get('content', '') if rec else ''}",
        }

    # 【什么】收尾节点
    def _report_node(state: dict) -> dict:
        return {"report": state.get("report", "") or f"[{name}] 无产出", "finished": True}

    # 【什么】execute 之后走委派还是收尾
    def _after_execute(state: dict) -> str:
        if state.get("delegate_request") and delegate_fn:
            return "delegate"
        return "report"

    # 【什么】delegate 之后回 execute 还是直接收尾
    # 【作用】iteration 达到上限强制收尾：execute 若固执地反复要委派，靠这里断循环
    def _after_delegate(state: dict) -> str:
        if state.get("iteration", 0) >= max_iterations:
            return "report"
        return "execute"

    workflow = StateGraph(WorkerStateDict)
    workflow.add_node("execute", _execute_node)       # type: ignore
    workflow.add_node("delegate", _delegate_node)     # type: ignore
    workflow.add_node("report", _report_node)         # type: ignore
    workflow.set_entry_point("execute")
    workflow.add_conditional_edges("execute", _after_execute, {"delegate": "delegate", "report": "report"})
    workflow.add_conditional_edges("delegate", _after_delegate, {"execute": "execute", "report": "report"})
    workflow.add_edge("report", END)
    return workflow.compile()


# ==================================================================
# 管理图：plan → delegate → reconcile → answer
# ==================================================================
class SubgraphOrchestrator:
    """子图委派编排器：一张管理图驱动 N 张子图，四件套全程在场。

    【输入输出对照】orch = SubgraphOrchestrator()（全默认，确定性模拟执行器）：
        state = orch.run("查营收；算环比")
        state["results"]  → 2 条，content 形如 "[worker-1] 已完成: 查营收"
        state["final_answer"] → 两段产出以空行拼接
        state["degraded"] → []（没有超时/异常）

    【装配到生产】planner 换成主管 Agent 的规划调用；
        worker_execute 换成 ReActAgent 的执行循环（或直接 invoke AgentGraph）。
        骨架与四件套不用动。
    """

    def __init__(
        self,
        config: Optional[SubgraphConfig] = None,
        planner: Optional[Callable[[str], list[dict]]] = None,
        worker_execute: Optional[Callable[[dict], dict]] = None,
    ):
        self.config = config or SubgraphConfig()
        self._planner = planner or self._default_planner
        self._worker_execute = worker_execute or self._default_execute
        self._events: list[dict] = []            # 运行轨迹（给文档与测试断言用）
        self._events_lock = threading.Lock()     # delegate 节点并行收结果时并发 append

    # ------------------------------------------------------------------
    # 默认规划器：按 ；/; 切分问题（确定性，不调 LLM）
    # ------------------------------------------------------------------
    @staticmethod
    def _default_planner(query: str) -> list[dict]:
        parts = [p.strip() for p in query.replace(";", "；").split("；") if p.strip()]
        return [
            {"task_id": f"t{i+1}", "assignee": f"worker-{i+1}", "instruction": p,
             "group": "", "depth": 1}
            for i, p in enumerate(parts)
        ]

    # ------------------------------------------------------------------
    # 默认执行器：确定性模拟（生产替换点）
    # 【输入输出对照】state={"instruction":"查营收","assignee":"worker-1"} →
    #   {"report": "[worker-1] 已完成: 查营收"}
    # ------------------------------------------------------------------
    @staticmethod
    def _default_execute(state: dict) -> dict:
        return {"report": f"[{state.get('assignee', 'worker')}] 已完成: {state.get('instruction', '')}"}

    def _emit(self, kind: str, **kw) -> None:
        with self._events_lock:
            self._events.append({"kind": kind, "ts": time.time(), **kw})

    # ------------------------------------------------------------------
    # 【什么】管理图的 plan 节点
    # 【作用】拆子任务 + max_subtasks 截断（超出部分记事件，不静默丢）
    # ------------------------------------------------------------------
    def _plan_node(self, state: dict) -> dict:
        subtasks = self._planner(state.get("query", ""))
        if len(subtasks) > self.config.max_subtasks:
            dropped = len(subtasks) - self.config.max_subtasks
            subtasks = subtasks[: self.config.max_subtasks]
            self._emit("plan_truncated", dropped=dropped)
        _logger.info(f"[plan] 拆出 {len(subtasks)} 个子任务")
        return {"subtasks": subtasks}

    # ------------------------------------------------------------------
    # 【什么】管理图的 delegate 节点（子图委派核心）
    # 【作用】每个子任务映射成子图状态 → 线程池并行 invoke 子图 → 超时/异常降级
    # 【输入输出对照】subtasks=3、其中 t2 超时（timeout=0.3，执行器 sleep 2）：
    #   results → t1、t3 两条成功记录
    #   degraded → [{"task_id": "t2", "reason": "子任务超时(0.3s)"}]
    # ------------------------------------------------------------------
    def _delegate_node(self, state: dict) -> dict:
        subtasks = state.get("subtasks", [])
        results: list[dict] = []
        degraded: list[dict] = []
        timeout = self.config.task_timeout
        if subtasks:
            with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(subtasks))) as ex:
                future_map = {ex.submit(self._invoke_worker, t): t for t in subtasks}
                for fut, task in future_map.items():
                    try:
                        # 【超时为什么不叠加】futures 已并行在跑，逐个 result(timeout=)
                        #   的等待是重叠的：总等待 ≈ 最慢那个，不是各任务超时之和
                        out = fut.result(timeout=timeout)
                        results.append(out)
                    except TimeoutError:
                        fut.cancel()
                        degraded.append({"task_id": task["task_id"],
                                         "reason": f"子任务超时({timeout}s)"})
                        self._emit("task_timeout", task_id=task["task_id"])
                    except Exception as e:
                        degraded.append({"task_id": task["task_id"],
                                         "reason": f"{type(e).__name__}: {e}"})
                        self._emit("task_error", task_id=task["task_id"], error=str(e))
        _logger.info(f"[delegate] 成功 {len(results)} | 降级 {len(degraded)}")
        return {"results": results, "degraded": degraded}

    # ------------------------------------------------------------------
    # 【什么】把一个子任务映射成子图状态并 invoke（状态映射就是这几行）
    # 【作用】父图字段 → 子图字段；子图终态 → 父图结果记录
    # ------------------------------------------------------------------
    def _invoke_worker(self, task: dict) -> dict:
        assignee = task.get("assignee", "worker")
        child_state: WorkerStateDict = {
            "task_id": task["task_id"],
            "assignee": assignee,
            "instruction": task.get("instruction", ""),
            "depth": int(task.get("depth", 1)),
            "chain": [assignee],
            "context": "",
            "delegations": [],
            "denied": [],
            "iteration": 0,
        }
        worker = build_worker_graph(
            assignee,
            execute_fn=self._worker_execute,
            delegate_fn=self._spawn,
            max_iterations=self.config.worker_max_iterations,
        )
        final = worker.invoke(dict(child_state))
        return {
            "task_id": task["task_id"],
            "assignee": assignee,
            "group": task.get("group", ""),
            "content": final.get("report", ""),
            "success": True,
            "delegations": final.get("delegations", []),
            "denied": final.get("denied", []),
        }

    # ------------------------------------------------------------------
    # 【四件套①】委派深度限制 + 防环（工作图的 delegate 节点经此递归）
    # 【作用】第 max_depth 层的工作图不许再转；委派链上出现过的名字直接拒
    # 【输入输出对照】max_depth=2：worker-a(第1层)委派 → worker-b(第2层)放行；
    #   worker-b 再委派 → depth(2) >= max_depth(2) → 拒绝，记进 denied
    # ------------------------------------------------------------------
    def _spawn(self, task_id: str, to_agent: str, depth: int, chain: list[str]) -> Optional[dict]:
        if not to_agent:
            return None
        # 【四件套①的另一半：防环】A→B→A 式的互相甩锅在这里直接拦掉
        if to_agent in (chain or []):
            return None
        if depth >= self.config.max_depth:
            return None
        child_state: WorkerStateDict = {
            "task_id": f"{task_id}>{to_agent}",
            "assignee": to_agent,
            "instruction": f"承接来自 {chain[-1] if chain else '?'} 的委派",
            "depth": depth + 1,
            "chain": list(chain or []) + [to_agent],
            "context": "",
            "delegations": [],
            "denied": [],
            "iteration": 0,
        }
        worker = build_worker_graph(
            to_agent,
            execute_fn=self._worker_execute,
            delegate_fn=self._spawn,               # 受托方还能再转，直到深度闸拒绝
            max_iterations=self.config.worker_max_iterations,
        )
        final = worker.invoke(dict(child_state))
        started = time.time()
        return {
            "from": chain[-1] if chain else "?",
            "to": to_agent,
            "task_id": child_state["task_id"],
            "content": final.get("report", ""),
            "duration_ms": int((time.time() - started) * 1000),
            # 嵌套账本随记录带出，由委派方的 delegate 节点并入自己的账本
            "nested_delegations": final.get("delegations", []),
            "nested_denied": final.get("denied", []),
        }

    # ------------------------------------------------------------------
    # 【四件套④】子结果冲突处理
    # 【作用】同 group 的多条产出内容不一致时：多数决 → 平票取先完成；冲突留痕
    # 【输入输出对照】group="revenue" 三条产出 ["A","A","B"]：
    #   conflicts → [{"group":"revenue","variants":["A","A","B"],"winner":"A","rule":"多数决"}]
    #   两条 A 记录标 resolution.winner="A"；B 标 resolution.overridden=True
    # ------------------------------------------------------------------
    def _reconcile_node(self, state: dict) -> dict:
        results = list(state.get("results", []))
        conflicts: list[dict] = []
        groups: dict[str, list[dict]] = {}
        for r in results:
            if r.get("group"):
                groups.setdefault(r["group"], []).append(r)
        for group, members in groups.items():
            contents = [m.get("content", "") for m in members]
            if len(set(contents)) <= 1:
                continue                              # 全员一致，无冲突
            # 多数决：数每个内容出现次数；平票时取 duration/顺序里最先完成的那条
            counts: dict[str, int] = {}
            for c in contents:
                counts[c] = counts.get(c, 0) + 1
            best = max(counts.values())
            winners = [c for c, n in counts.items() if n == best]
            if len(winners) == 1:
                winner, rule = winners[0], "多数决"
            else:
                winner, rule = contents[0], "平票取先完成"
            conflicts.append({"group": group, "variants": contents,
                              "winner": winner, "rule": rule})
            for m in members:
                m["resolution"] = {"winner": winner, "rule": rule,
                                   "overridden": m.get("content", "") != winner}
            self._emit("conflict", group=group, winner=winner, rule=rule)
        if conflicts:
            _logger.warning(f"[reconcile] 检出 {len(conflicts)} 组冲突，已按规则收敛")
        return {"results": results, "conflicts": conflicts}

    # ------------------------------------------------------------------
    # 【四件套③】失败降级：最终答案只拼成功项，降级/冲突附注在末尾
    # 【输入输出对照】t1 成功、t2 超时：
    #   final_answer = "[worker-1] 已完成: 查营收\n\n（降级说明: t2 子任务超时(0.3s)）"
    #   全部失败时 = "（全部子任务失败，降级为直接应答）请稍后重试或缩小问题范围。"
    # ------------------------------------------------------------------
    def _answer_node(self, state: dict) -> dict:
        ok = [r for r in state.get("results", []) if r.get("success")]
        parts = [f"[{r.get('assignee', '')}] {r.get('content', '')}" for r in ok]
        answer = "\n\n".join(parts)
        degraded = state.get("degraded", [])
        conflicts = state.get("conflicts", [])
        notes = []
        if degraded:
            notes.append("（降级说明: " + "；".join(f"{d['task_id']} {d['reason']}" for d in degraded) + "）")
        for c in conflicts:
            notes.append(f"（冲突收敛: 组 {c['group']} 按「{c['rule']}」取 {c['winner'][:50]}）")
        if not answer:
            answer = "（全部子任务失败，降级为直接应答）请稍后重试或缩小问题范围。"
        if notes:
            answer = answer + "\n\n" + "\n".join(notes)
        return {"final_answer": answer, "finished": True}

    # ------------------------------------------------------------------
    # 【什么】装配并编译管理图（外部唯一入口 run 之前只跑一次）
    # ------------------------------------------------------------------
    def _build_graph(self):
        workflow = StateGraph(ManagerStateDict)
        workflow.add_node("plan", self._plan_node)             # type: ignore
        workflow.add_node("delegate", self._delegate_node)     # type: ignore
        workflow.add_node("reconcile", self._reconcile_node)   # type: ignore
        workflow.add_node("answer", self._answer_node)         # type: ignore
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "delegate")
        workflow.add_edge("delegate", "reconcile")
        workflow.add_edge("reconcile", "answer")
        workflow.add_edge("answer", END)
        return workflow.compile()

    # 【什么】跑一次完整子图委派（外部唯一入口）
    # 【作用】invoke 管理图，返回终态 dict；events 供断言/文档取轨迹
    def run(self, query: str) -> dict:
        with self._events_lock:
            self._events = []
        graph = self._build_graph()
        final = graph.invoke({"query": query})
        _logger.info(f"[subgraph] 运行完成: results={len(final.get('results', []))} "
                     f"degraded={len(final.get('degraded', []))}")
        return final

    @property
    def events(self) -> list[dict]:
        with self._events_lock:
            return list(self._events)
