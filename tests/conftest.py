"""
pytest 全局夹具
===============
保证测试在「没装数据库 / 没装可选框架」的机器上也能跑起来：
    - 把项目根目录塞进 sys.path（否则 import modules.xxx 失败）
    - 给 .env 里可能缺失的配置兜底默认值（Settings 是 pydantic-settings，缺必填项会抛错）
    - 提供一批「假 Agent」，让 harness 的编排逻辑可以在无 LLM 的情况下被测到

【为什么不直接连真实 PG / Milvus 做集成测试】
    集成测试对 CI 不友好：任何一台没起容器的机器都会红。
    这里全部用假对象替换边界依赖，测的是「我们自己的编排与降级逻辑」，
    外部组件的正确性由它们自己的测试保证。
"""
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]      # tests/ 的上一层 = 项目根目录
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))               # 塞进 sys.path，否则 import modules.xxx 失败


# ------------------------------------------------------------------
# 【配置兜底】只填「未设置」的项，不覆盖用户本机已有的 .env 值。
#   不这么写的后果：CI 机器没有 .env，get_settings() 抛 ValidationError，
#   所有测试在 import 阶段就挂掉，看不到任何有用信息。
# ------------------------------------------------------------------
_DEFAULTS = {
    "DEEPSEEK_API_KEY": "test-key",       # 主模型密钥：没 .env 时的兜底值
    "ZHIPUAI_API_KEY": "test-key",        # 备用模型密钥
    "PG_HOST": "localhost",               # PG 会话库主机
    "PG_PORT": "5432",                    # PG 端口（pgvector 装在这里）
    "PG_USER": "test",                    # PG 用户名
    "PG_PASSWORD": "test",                # PG 密码
    "PG_DB": "test",                      # PG 库名
    "MILVUS_HOST": "localhost",           # Milvus 向量库主机
    "MILVUS_PORT": "19530",               # Milvus 端口
    "MYSQL_HOST": "localhost",            # MySQL 业务库主机
    "MYSQL_USER": "test",                 # MySQL 用户名
    "MYSQL_PASSWORD": "test",             # MySQL 密码
    "MYSQL_DB": "test",                   # MySQL 库名
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)


# 【什么】返回项目根目录的夹具
# 【作用】用例里拼 sql/、examples/ 等路径时不用再算相对位置
@pytest.fixture
def project_root() -> Path:
    return _ROOT


# 【什么】造子任务（SubTask）的工厂夹具
# 【作用】一行造出一个子任务，省掉每个用例重复写 4 个字段
@pytest.fixture
def task_factory():
    """造子任务的工厂：task_factory("t1", "worker_a", depends_on=["t0"])。"""
    from modules.multiagent.models import SubTask

    # 【什么】按传入参数造一个 SubTask
    # 【作用】depends_on 默认空列表、priority 默认 1，调用方只写关心的字段
    def _make(tid: str, assignee: str, depends_on=None, priority: int = 1) -> "SubTask":
        return SubTask(id=tid, assignee=assignee, instruction=f"执行 {tid}",
                       depends_on=depends_on or [], priority=priority)
    return _make


# 【什么】把 harness 的 Agent 工厂换成假实现的夹具
# 【作用】返回 inject 注入器，让用例能预设计划、汇总话术与每个 worker 的表现
@pytest.fixture
def fake_team(monkeypatch):
    """
    把 harness 的 Agent 工厂换成假实现，返回「注入器」函数。

    用法：
        inject = fake_team()
        inject(plan=..., workers={"w1": "ok"}, synth="最终答案")
        h = AgentHarness(config=team_cfg)

    【为什么用 monkeypatch 换 build_agent 而不是继承】
        harness.register 里写死了 build_agent(spec, llm=..., tool_registry=...)，
        真实 build_agent 会去 new 一个 LLMClient（要密钥、要网络）。
        换掉这一个函数就得手，比整条链路打桩干净得多。
    """
    from modules.multiagent import harness as h_module
    from modules.multiagent.agents import BaseAgent, SupervisorAgent
    from modules.multiagent.models import (
        AgentOutput, AgentRole, AgentSpec, PlanResult, SubTask,
    )

    state = {
        "plan": PlanResult(direct_answer="直接回答"),   # 主管默认计划：直接答，不派任务
        "synth": "团队汇总答案",                        # 主管汇总时的固定话术
        "workers": {},                                  # {Agent 名: 表现} 如 {"w2": "fail"}
        "plan_delay": 0.0,                              # 让主管规划慢一点，用来稳定地触发「总超时」分支
    }

    # 【什么】假主管 Agent
    # 【作用】只实现 harness 真正用到的 plan / synthesize，返回 state 里预设的值
    class _FakeSupervisor(SupervisorAgent):
        # 【什么】只补 spec 的构造旁路
        # 【作用】绕开 BaseAgent.__init__ 里 new LLMClient 的动作，避免要密钥要网络
        def __new__(cls):
            # 【为什么走 __new__ 而不是 super().__init__】
            #   BaseAgent.__init__ 会在没传 llm 时自己去 new 一个 LLMClient（要密钥、要网络），
            #   单测里那是一条死路。这里直接绕过构造，只补上 harness 真正用到的 spec。
            obj = object.__new__(cls)
            obj.spec = AgentSpec(name="supervisor", role=AgentRole.SUPERVISOR)
            return obj

        def __init__(self, *args, **kwargs):
            pass        # 屏蔽 BaseAgent.__init__（Python 会在 __new__ 之后自动调它）

        # 【什么】返回预设的拆分计划
        # 【作用】plan_delay 大于 0 时先睡一会儿，让「总超时」分支稳定命中
        def plan(self, ctx, roster):
            # 【为什么要能拖慢规划】总超时分支依赖「deadline 已过」这一条件，
            #   规划耗时不可控时该分支会时灵时不灵（flaky test）。
            #   让它可控，测试才能在任意性能的机器上都稳定命中。
            if state.get("plan_delay"):
                import time
                time.sleep(state["plan_delay"])
            return state["plan"]

        # 【什么】返回预设的汇总话术
        # 【作用】用例可以断言最终答案就是注入进来的那个字符串
        def synthesize(self, ctx, outputs):
            return state["synth"]

    # 【什么】假执行者 Agent
    # 【作用】run() 按 outcome 演四种表现：成功 / 失败 / 卡住 / 睡 N 秒
    class _FakeWorker(BaseAgent):
        """outcome: 'ok' 成功 / 'fail' 失败 / 'hang' 卡住 / 数字 = sleep 秒 / 可调用对象。"""

        # 【什么】只记 spec 与 outcome 的构造旁路
        # 【作用】同主管：不走 BaseAgent.__init__，不建 LLMClient
        def __new__(cls, spec: AgentSpec, outcome):
            obj = object.__new__(cls)
            obj.spec = spec                  # harness 靠 spec.name 认人
            obj._outcome = outcome           # 本次要演的表现，run() 里读
            return obj

        def __init__(self, *args, **kwargs):
            pass        # 同上：不走 BaseAgent.__init__

        # 【什么】按 outcome 演一次任务执行
        # 【作用】ok/可调用对象返回成功产出，fail 返回失败产出，hang/数字先睡再返回
        def run(self, task: SubTask, ctx):
            import time
            outcome = self._outcome
            if callable(outcome):
                return outcome(task)
            if outcome == "fail":
                return AgentOutput.failed(self.spec.name, task.id, "假失败")
            if outcome == "hang":
                time.sleep(5)
            elif isinstance(outcome, (int, float)):
                time.sleep(outcome)
            return AgentOutput(agent=self.spec.name, task_id=task.id,
                               content=f"{self.spec.name} 完成 {task.id}", steps=1)

    # 【什么】替换 harness 模块里的 build_agent
    # 【作用】按 role 决定造主管还是执行者；执行者表现从 state["workers"] 取，默认 "ok"
    def _build(spec: AgentSpec, llm=None, tool_registry=None):
        if spec.role == AgentRole.SUPERVISOR:
            return _FakeSupervisor()
        return _FakeWorker(spec, state["workers"].get(spec.name, "ok"))

    monkeypatch.setattr(h_module, "build_agent", _build)

    # 【什么】用例侧的注入器（fake_team() 的返回值）
    # 【作用】改 plan / synth / workers / plan_delay；传 None 的项保持原值不动
    def inject(plan=None, synth=None, workers=None, plan_delay=None):
        if plan is not None:
            state["plan"] = plan
        if synth is not None:
            state["synth"] = synth
        if workers is not None:
            state["workers"] = workers
        if plan_delay is not None:
            state["plan_delay"] = plan_delay
        return state

    return inject
