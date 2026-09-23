"""
数据契约测试：多 Agent 之间传的那些 pydantic 模型
=================================================
测的是「模型说胡话时我们能不能接住」，而不是字段默认值。
这些校验是多 Agent 链路的第一道防线：模型返回缺字段、id 重复、团队没主管，
都必须在构造时炸掉，而不是等到 harness 跑完才发现答案是空的。
"""
import pytest
from pydantic import ValidationError

from modules.multiagent.models import (
    AgentOutput, AgentRole, AgentSpec, EvalCase, EvalReport, PlanResult,
    RunResult, RunStatus, SubTask, TeamConfig, json_schema_hint, new_id,
)


# ------------------------------------------------------------------
# 【PlanResult 二选一】subtasks 与 direct_answer 必须有一个非空
#   不这条校验的后果：模型同时返回空列表和空字符串，harness 派 0 个任务
#   却继续等汇总，最终返回空答案且不报错 —— 最难排查的静默失败。
# ------------------------------------------------------------------
# 【什么】空计划必须被拒的用例
# 【作用】验：subtasks 与 direct_answer 全空时抛 ValidationError
def test_plan_must_have_subtasks_or_direct_answer():
    with pytest.raises(ValidationError):
        PlanResult()  # 两个都空 → 必须拒绝


# 【什么】只给 direct_answer 的用例
# 【作用】验：is_direct 为 True 且 subtasks 是空列表
def test_plan_direct_answer_ok():
    p = PlanResult(direct_answer="你好")
    assert p.is_direct is True
    assert p.subtasks == []


# 【什么】带子任务计划的用例
# 【作用】验：有 subtasks 时 is_direct 为 False，不走直接答路径
def test_plan_with_subtasks_not_direct():
    p = PlanResult(subtasks=[SubTask(id="t1", assignee="w1", instruction="查营收")])
    assert p.is_direct is False


# 【什么】子任务 id 重复的用例
# 【作用】验：两个 id 都叫 t1 的计划在构造时就被拒绝
def test_plan_reject_duplicate_task_ids():
    t1 = SubTask(id="t1", assignee="w1", instruction="a")
    t2 = SubTask(id="t1", assignee="w2", instruction="b")  # id 撞车
    with pytest.raises(ValidationError):
        PlanResult(subtasks=[t1, t2])


# ------------------------------------------------------------------
# 【SubTask 边界】priority 限定 1~5，越界必须在入口拒绝，
#   否则 topo_batches 的排序拿到 999 会打乱批内顺序且没人知道。
# ------------------------------------------------------------------
# 【什么】priority 取值边界的用例
# 【作用】验：5 合法，6 与 0 越界被拒（区间是 1~5）
def test_subtask_priority_bounds():
    SubTask(id="t", assignee="w", instruction="x", priority=5)   # 合法
    with pytest.raises(ValidationError):
        SubTask(id="t", assignee="w", instruction="x", priority=6)
    with pytest.raises(ValidationError):
        SubTask(id="t", assignee="w", instruction="x", priority=0)


# 【什么】Agent 名空白的用例
# 【作用】验：纯空格的 name 被拒，否则 assignee 指向谁是不确定的
def test_agent_spec_name_cannot_be_blank():
    with pytest.raises(ValidationError):
        AgentSpec(name="   ", role=AgentRole.RESEARCHER)


# 【什么】temperature 上限的用例
# 【作用】验：3.0 超出 0~2 的取值范围，构造时被拒
def test_agent_spec_temperature_bounds():
    with pytest.raises(ValidationError):
        AgentSpec(name="w1", role=AgentRole.RESEARCHER, temperature=3.0)


# ------------------------------------------------------------------
# 【TeamConfig 团队完整性】
#   1. 必须有且仅有一个 enabled 的 supervisor（否则没人拆任务）
#   2. Agent 名不能重复（否则 assignee 指向谁是不确定的）
# ------------------------------------------------------------------
# 【什么】团队缺主管的用例
# 【作用】验：没有 supervisor 的 TeamConfig 构造即失败
def test_team_requires_supervisor():
    with pytest.raises(ValidationError):
        TeamConfig(agents=[AgentSpec(name="w1", role=AgentRole.RESEARCHER)])


# 【什么】团队里 Agent 重名的用例
# 【作用】验：两个都叫 w1 的成员被拒，避免 assignee 指向不明
def test_team_reject_duplicate_names():
    agents = [
        AgentSpec(name="sup", role=AgentRole.SUPERVISOR),
        AgentSpec(name="w1", role=AgentRole.RESEARCHER),
        AgentSpec(name="w1", role=AgentRole.ANALYST),
    ]
    with pytest.raises(ValidationError):
        TeamConfig(agents=agents)


# 【什么】合法团队配置的用例
# 【作用】验：1 主管 + 1 执行者能构造成功，且默认值 max_workers=4、timeout=120 秒
def test_team_ok():
    agents = [
        AgentSpec(name="sup", role=AgentRole.SUPERVISOR),
        AgentSpec(name="w1", role=AgentRole.RESEARCHER),
    ]
    cfg = TeamConfig(agents=agents)
    assert cfg.max_workers == 4 and cfg.timeout_seconds == 120.0


# ------------------------------------------------------------------
# 【RunResult 状态语义】
#   ok = success 或 partial；failed_agents 只统计 outputs 里失败的。
#   这两条是 API 返回码与前端提示的依据，语义错了前端会把降级答案当成功。
# ------------------------------------------------------------------
# 【什么】RunResult 成功语义的用例
# 【作用】验：PARTIAL 也算 ok，failed_agents 只列 outputs 里失败的那个 Agent
def test_run_result_ok_and_failed_agents():
    r = RunResult(query="q", status=RunStatus.PARTIAL)
    r.outputs = [
        AgentOutput(agent="w1", task_id="t1", success=True),
        AgentOutput.failed("w2", "t2", "超时"),
    ]
    assert r.ok is True
    assert r.failed_agents == ["w2"]

    r2 = RunResult(query="q", status=RunStatus.FAILED)
    assert r2.ok is False


# 【什么】评测报告通过率的用例
# 【作用】验：total=4 / passed=3 渲染成字符串 "3/4"
def test_eval_report_pass_rate():
    rep = EvalReport(total=4, passed=3)
    assert rep.pass_rate == "3/4"


# ------------------------------------------------------------------
# 【json_schema_hint】喂给 LLM 的结构说明必须是「合法 JSON」（双引号）
#   不 dumps 直接用 Python dict 的 str() 会得到单引号，模型照抄后
#   json.loads 解析失败，白花一轮重试。
# ------------------------------------------------------------------
# 【什么】结构说明必须是合法 JSON 的用例
# 【作用】验：hint 能被 json.loads 解析，含两个关键字段，且不带 $defs 嵌套定义
def test_json_schema_hint_is_valid_json():
    import json
    hint = json_schema_hint(PlanResult)
    parsed = json.loads(hint)          # 能解析 = 是合法 JSON
    assert "subtasks" in parsed and "direct_answer" in parsed
    assert "$defs" not in parsed       # 已精简，不带嵌套定义（省 token）


# 【什么】ID 生成格式的用例
# 【作用】验：new_id("run") 得到 "run_" 前缀 + 8 位随机串，总长 12
def test_new_id_format():
    rid = new_id("run")
    assert rid.startswith("run_") and len(rid) == len("run_") + 8
