"""
子图委派（Subgraph）测试
========================
覆盖：基本拆分与产出回传、四件套（深度限制 / 超时降级 / 冲突收敛 / 全失败降级）。

【怎么跑】
    python -m pytest tests/test_subgraph_delegation.py -q

【为什么不用真 LLM】默认执行器是确定性函数（"已完成: 指令"），
图套图的骨架与四件套的边界行为不依赖模型输出，测试才稳定。
"""
from __future__ import annotations

import time

from modules.multiagent.subgraph import SubgraphConfig, SubgraphOrchestrator


def test_basic_split_and_collect():
    """两个子任务（按 ；拆分）→ 两条产出 → 拼进最终答案。"""
    orch = SubgraphOrchestrator()
    state = orch.run("查营收；算环比")
    assert len(state["subtasks"]) == 2
    assert len(state["results"]) == 2
    contents = {r["content"] for r in state["results"]}
    assert contents == {"[worker-1] 已完成: 查营收", "[worker-2] 已完成: 算环比"}
    assert "查营收" in state["final_answer"] and "算环比" in state["final_answer"]
    assert state["degraded"] == []
    assert state["finished"] is True


def test_depth_limit_denies_third_layer():
    """四件套①：第 max_depth 层的工作图不许再委派，拒绝要留痕。

    场景：worker-1（第 1 层）execute 后提出委派 → worker-b（第 2 层）放行；
    worker-b 再提出委派 → depth(2) >= max_depth(2) → 拒绝。
    """
    def always_delegate(state: dict) -> dict:
        if state.get("delegate_request") is None and "委派产出" not in state.get("context", ""):
            return {"delegate_request": {"to": f"{state['assignee']}-child",
                                          "instruction": "下一层", "reason": "缺数据"}}
        return {"report": f"[{state['assignee']}] 干完了: {state['instruction']}"}

    orch = SubgraphOrchestrator(config=SubgraphConfig(max_depth=2),
                                worker_execute=always_delegate)
    state = orch.run("算环比")
    # 第 1 层成功转出 1 次（worker-1 → worker-1-child）
    assert len(state["results"][0]["delegations"]) == 1
    # 第 2 层的再委派被深度闸拒绝，denied 里必须能看到原因
    assert len(state["results"][0]["denied"]) >= 1
    assert "拒绝" in state["results"][0]["denied"][0]["reason"]
    assert state["results"][0]["success"] is True


def test_subtask_timeout_degrades():
    """四件套②③：慢子任务超时 → 进 degraded；快任务照常进最终答案。"""

    def slow_or_fast(state: dict) -> dict:
        if "慢" in state.get("instruction", ""):
            time.sleep(2.0)                       # 远超 0.3s 超时
            return {"report": "太慢了"}
        return {"report": f"[{state['assignee']}] 已完成: {state['instruction']}"}

    orch = SubgraphOrchestrator(
        config=SubgraphConfig(task_timeout=0.3),
        worker_execute=slow_or_fast,
    )
    state = orch.run("快任务；慢任务")
    assert len(state["results"]) == 1                       # 只有快任务成功
    assert state["degraded"] == [{"task_id": "t2", "reason": "子任务超时(0.3s)"}]
    assert "快任务" in state["final_answer"]
    assert "降级说明" in state["final_answer"]              # 降级必须注明，不能静默丢


def test_conflict_majority_resolution():
    """四件套④：同组三条产出 A/A/B → 多数决取 A，冲突与覆盖关系留痕。"""

    def biased(state: dict) -> dict:
        # worker-1/worker-3 给结论 A，worker-2 给结论 B（同 group="revenue"）
        verdict = "A" if state["assignee"] != "worker-2" else "B"
        return {"report": f"营收结论: {verdict}"}

    def planner(query: str) -> list[dict]:
        return [
            {"task_id": f"t{i+1}", "assignee": f"worker-{i+1}",
             "instruction": "估营收", "group": "revenue", "depth": 1}
            for i in range(3)
        ]

    orch = SubgraphOrchestrator(planner=planner, worker_execute=biased)
    state = orch.run("估营收")
    assert len(state["conflicts"]) == 1
    conflict = state["conflicts"][0]
    assert conflict["group"] == "revenue"
    assert conflict["winner"] == "营收结论: A"
    assert conflict["rule"] == "多数决"
    # 少数派 B 被标记 overridden，多数派没有
    overridden = [r for r in state["results"] if r["resolution"]["overridden"]]
    assert len(overridden) == 1 and "B" in overridden[0]["content"]
    assert "冲突收敛" in state["final_answer"]


def test_all_failed_full_degradation():
    """四件套③兜底：全部子任务异常 → 降级为直接应答话术，不抛异常。"""

    def always_fail(state: dict) -> dict:
        raise RuntimeError("模拟崩溃")

    orch = SubgraphOrchestrator(worker_execute=always_fail)
    state = orch.run("任务A；任务B")
    assert state["results"] == []
    assert len(state["degraded"]) == 2
    assert "全部子任务失败" in state["final_answer"]
    assert state["finished"] is True
