"""
Agent Harness（运行时框架）模块
==============================
Harness 是「把 Agent 跑起来」的那层底座，Agent 本身只负责思考。

【Harness 到底管什么】
    类比工厂：Agent 是工人，Harness 是车间——负责排班（编排）、
    工位分配（并发）、工时上限（超时/步数预算）、监控摄像头（trace）、
    质检（评测）。没有它，多个 Agent 就只是一堆能说话的对象，
    谁先跑、跑多久、失败了怎么办、跑得怎么样，全都无人负责。

【四个治理点，缺一个就会在生产环境出事】
    1. 编排   —— 按 depends_on 做 DAG 分批，能并行的并行
    2. 超时   —— 单次运行总 deadline + 单任务等待超时（LLM 卡住不能拖死全队）
    3. 步数   —— 总步数预算，防多 Agent 各自循环叠加成天价账单
    4. 降级   —— 单个 Agent 失败只降级不崩溃，最终答案里如实标注未完成部分

【为什么评测要内建在 harness 里】
    多 Agent 系统的回归测试最难的是"没法断言"——答案每次都不一样。
    关键词命中 + 耗时 + 完成率这套粗粒度指标，胜在能进 CI：
    改一次提示词，跑一遍 evaluate()，通过率掉了立刻知道。
"""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Callable, Optional

from modules.common.logger import get_logger
from modules.multiagent.agents import (
    BaseAgent,
    RunContext,
    SupervisorAgent,
    build_agent,
    build_default_team,
)
from modules.multiagent.models import (
    AgentOutput,
    AgentRole,
    AgentSpec,
    EvalCase,
    EvalReport,
    EvalResult,
    PlanResult,
    RunResult,
    RunStatus,
    SubTask,
    TeamConfig,
    TraceEvent,
)
from modules.multimodal.media import MultiModalInput

_logger = get_logger("harness")


class AgentHarness:
    """多 Agent 运行时：注册团队 → 跑任务 → 出结果 + trace + 评测。"""

    def __init__(
        self,
        config: Optional[TeamConfig] = None,
        llm=None,
        tool_registry=None,
        on_event: Optional[Callable[[TraceEvent], None]] = None,
    ):
        self.config = config or TeamConfig(agents=build_default_team())
        self.llm = llm
        self.tools = tool_registry
        self.on_event = on_event
        self.agents: dict[str, BaseAgent] = {}
        self.supervisor: Optional[SupervisorAgent] = None
        for spec in self.config.agents:
            if spec.enabled:
                self.register(spec)

    # ----------------------------------------------------------------------
    # 装配
    # ----------------------------------------------------------------------
    def register(self, spec: AgentSpec) -> BaseAgent:
        """注册一个 Agent。supervisor 自动成为默认主管。"""
        agent = build_agent(spec, llm=self.llm, tool_registry=self.tools)
        self.agents[spec.name] = agent
        if isinstance(agent, SupervisorAgent) and self.supervisor is None:
            self.supervisor = agent
        _logger.info(f"Agent 注册: {spec.name} ({spec.role.value})")
        return agent

    @property
    def roster(self) -> list[AgentSpec]:
        """主管选人时看到的成员清单（排除主管自己）。"""
        return [
            a.spec for a in self.agents.values()
            if a.spec.role != AgentRole.SUPERVISOR and a.spec.enabled
        ]

    def _emit(self, result: RunResult, kind: str, message: str, agent: str = "", **payload):
        """写一条 trace 事件（同时回调外部钩子，可接 SSE/审计）。"""
        event = TraceEvent(kind=kind, message=message, agent=agent, payload=payload)
        result.trace.append(event)
        if self.on_event:
            try:
                self.on_event(event)
            except Exception as e:
                _logger.debug(f"事件回调异常（忽略）: {e}")

    # ----------------------------------------------------------------------
    # 编排：DAG 分批
    # ----------------------------------------------------------------------
    @staticmethod
    def topo_batches(tasks: list[SubTask]) -> tuple[list[list[SubTask]], list[str]]:
        """
        Kahn 算法把子任务按依赖分层，同层可并行。

        【输入 → 输出对照】
            输入: [t1(无依赖), t2(depends_on=[t1]), t3(depends_on=[t1]), t4(depends_on=[t2,t3])]
            输出: ([[t1], [t2,t3], [t4]], [])
                  第 1 批并行 1 个，第 2 批并行 2 个，第 3 批 1 个

        【环怎么处理】
            输出: ([[t1]], ["t5"])  ← t5 处在环里，被剔除并记进第二个返回值
            不这么写的后果：while 循环永远排不完，整次运行挂死（不是报错，是卡住）。
        """
        done: set[str] = set()
        pending = list(tasks)
        batches: list[list[SubTask]] = []
        while pending:
            ready = [t for t in pending if all(d in done for d in t.depends_on)]
            if not ready:
                # 剩下的全都等不到上游 = 存在环或引用了不存在的 id
                return batches, [t.id for t in pending]
            ready.sort(key=lambda t: -t.priority)
            batches.append(ready)
            done.update(t.id for t in ready)
            pending = [t for t in pending if t.id not in done]
        return batches, []

    # ----------------------------------------------------------------------
    # 主流程
    # ----------------------------------------------------------------------
    def run(
        self,
        query: str,
        mm: Optional[MultiModalInput] = None,
        session_id: str = "",
        run_id: str = "",
    ) -> RunResult:
        """
        跑一次完整的多 Agent 协作。

        【流水线】
            plan → 分批 → 并行执行 → 汇总 → 终审 → 组装 RunResult
            任一环节异常都记进 trace 并降级，不向外抛异常。

        Args:
            query:      用户问题
            mm:         多模态输入（可选，带图片时只有开图能力的 Agent 会去看）
            session_id: 会话 ID（透传给工具做去重隔离）
        """
        started = time.time()
        result = RunResult(query=query, multimodal=bool(mm and mm.image_count))
        if run_id:
            result.run_id = run_id
        ctx = RunContext(query=query, mm=mm, session_id=session_id)
        deadline = started + self.config.timeout_seconds
        steps_used = 0

        try:
            if self.supervisor is None:
                result.status = RunStatus.FAILED
                result.error = "未注册 supervisor，无法编排"
                self._emit(result, "error", result.error)
                return result

            # ---- 第 1 步：主管规划 ----------------------------------------
            plan = self.supervisor.plan(ctx, self.roster)
            result.plan = plan
            self._emit(result, "plan", plan.thought or f"拆出 {len(plan.subtasks)} 个子任务",
                       self.supervisor.name, subtasks=[t.model_dump() for t in plan.subtasks])

            # ---- 第 2 步：直接答就收工 ------------------------------------
            if plan.is_direct:
                result.final_answer = plan.direct_answer
                result.status = RunStatus.SUCCESS
                result.duration_ms = int((time.time() - started) * 1000)
                self._emit(result, "done", "主管判定无需拆任务，直接作答")
                return result

            # ---- 第 3 步：DAG 分批 + 并行执行 -----------------------------
            batches, cyclic = self.topo_batches(plan.subtasks)
            if cyclic:
                self._emit(result, "error", f"存在循环依赖或无效依赖，已剔除: {cyclic}")
            self._emit(result, "dispatch", f"共 {len(batches)} 批，最大并行 {self.config.max_workers}")

            for batch_no, batch in enumerate(batches, 1):
                if time.time() > deadline:
                    for t in batch:
                        result.outputs.append(
                            AgentOutput.failed(t.assignee, t.id, "总超时，未执行")
                        )
                    result.status = RunStatus.TIMEOUT
                    self._emit(result, "error", f"第 {batch_no} 批因总超时未执行")
                    break
                if steps_used >= self.config.max_total_steps:
                    for t in batch:
                        result.outputs.append(
                            AgentOutput.failed(t.assignee, t.id, "总步数预算耗尽，未执行")
                        )
                    self._emit(result, "error", f"步数预算耗尽，第 {batch_no} 批起不再执行")
                    break
                batch_outputs = self._run_batch(batch, ctx, deadline, result, batch_no)
                for out in batch_outputs:
                    # 产出记两处：result.outputs 是对外的账单，ctx.upstream 给下游当输入。
                    # 只写 upstream 不写 outputs 的话，最终答案明明生成了，
                    # result.outputs 却是空的 —— 排查时最难发现的那种 bug。
                    result.outputs.append(out)
                    steps_used += out.steps
                    ctx.upstream[out.task_id] = out

            # ---- 第 4 步：汇总 --------------------------------------------
            if any(o.success for o in result.outputs):
                draft = self.supervisor.synthesize(ctx, result.outputs)
                result.final_answer = draft or "（团队未产出有效结论）"
                self._emit(result, "synthesize", "汇总完成", self.supervisor.name)
            elif result.status == RunStatus.TIMEOUT:
                # 【为什么不在这里把 TIMEOUT 覆盖成 FAILED】
                #   两者是截然相反的运维信号：超时要调大 deadline / 减少并行，
                #   Agent 失败要查具体执行者。一律写成 FAILED，监控上就只会
                #   看到"一直失败"，看不出该扩容，问题会一直重复到用户投诉。
                result.final_answer = "抱歉，本次任务因超时未能完成，请缩小问题范围或稍后重试。"
                result.error = result.error or "总超时，任务未执行完"
                self._emit(result, "error", result.error)
            else:
                result.status = RunStatus.FAILED
                result.final_answer = "抱歉，本次任务的所有执行者均未能产出有效结果。"
                result.error = "全部 Agent 执行失败"
                self._emit(result, "error", result.error)
                result.duration_ms = int((time.time() - started) * 1000)
                return result

            # ---- 第 5 步：终审（可选）-------------------------------------
            if self.config.review_enabled and plan.needs_review:
                reviewer = self._first_by_role(AgentRole.REVIEWER)
                if reviewer is not None:
                    result.review = reviewer.review(ctx, result.final_answer)
                    self._emit(result, "review", result.review.content[:200], reviewer.name)

            # ---- 第 6 步：定状态 ------------------------------------------
            if result.status != RunStatus.TIMEOUT:
                failed = result.failed_agents
                result.status = RunStatus.PARTIAL if failed else RunStatus.SUCCESS
                if failed:
                    result.error = f"部分 Agent 失败: {failed}"
            result.duration_ms = int((time.time() - started) * 1000)
            self._emit(result, "done", f"status={result.status.value} 耗时={result.duration_ms}ms")
            return result

        except Exception as e:
            _logger.error(f"harness 运行异常: {type(e).__name__}: {e}")
            result.status = RunStatus.FAILED
            result.error = f"{type(e).__name__}: {e}"
            result.duration_ms = int((time.time() - started) * 1000)
            self._emit(result, "error", result.error)
            return result

    def _run_batch(
        self,
        batch: list[SubTask],
        ctx: RunContext,
        deadline: float,
        result: RunResult,
        batch_no: int,
    ) -> list[AgentOutput]:
        """并行跑一批任务；单个任务超时/异常都转成 failed，不影响同批其他任务。"""
        outputs: list[AgentOutput] = []

        def _one(task: SubTask) -> AgentOutput:
            agent = self.agents.get(task.assignee)
            if agent is None:
                return AgentOutput.failed(task.assignee, task.id, f"未注册的 Agent: {task.assignee}")
            self._emit(result, "agent_start", task.instruction[:120], agent.name, task_id=task.id)
            out = agent.run(task, ctx)
            self._emit(result, "agent_end",
                       f"{'成功' if out.success else '失败'} 耗时={out.duration_ms}ms",
                       agent.name, task_id=task.id, success=out.success)
            return out

        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(batch))) as pool:
            futures = {pool.submit(_one, t): t for t in batch}
            for future, task in futures.items():
                wait = max(deadline - time.time(), 1.0)
                try:
                    outputs.append(future.result(timeout=wait))
                except FutureTimeout:
                    _logger.error(f"任务超时: {task.id}（等待 {wait:.1f}s 未返回）")
                    outputs.append(AgentOutput.failed(task.assignee, task.id, "任务执行超时"))
                except Exception as e:
                    outputs.append(
                        AgentOutput.failed(task.assignee, task.id, f"{type(e).__name__}: {e}")
                    )
        _logger.info(f"第 {batch_no} 批完成: {len(outputs)} 个产出")
        return outputs

    def _first_by_role(self, role: AgentRole) -> Optional[BaseAgent]:
        for agent in self.agents.values():
            if agent.spec.role == role and agent.spec.enabled:
                return agent
        return None

    # ----------------------------------------------------------------------
    # 评测
    # ----------------------------------------------------------------------
    def evaluate(self, cases: list[EvalCase]) -> EvalReport:
        """
        跑一批用例并打分。

        【评分口径】
            关键词命中率（0.5）+ 完成率（0.3）+ 速度分（0.2）
            速度分 = max(0, 1 - 耗时 / max_duration_ms)
            要求 require_all_agents 的用例，任一 Agent 失败直接判不通过。

        【为什么用关键词而不是 LLM 打分】
            LLM-as-judge 更"智能"，但它自己也会漂移，不适合做 CI 门禁。
            关键词是确定性的：今天通过，明天改了提示词没通过，就是真的退化了。
        """
        report = EvalReport(total=len(cases))
        total_score = 0.0
        for case in cases:
            run = self.run(case.query)
            hits = [
                kw for kw in case.expected_keywords
                if kw.lower() in run.final_answer.lower()
            ]
            kw_score = len(hits) / len(case.expected_keywords) if case.expected_keywords else 1.0
            done_ratio = (
                sum(1 for o in run.outputs if o.success) / len(run.outputs)
                if run.outputs else 0.0
            )
            speed_score = max(0.0, 1 - run.duration_ms / max(case.max_duration_ms, 1))
            score = round(kw_score * 0.5 + done_ratio * 0.3 + speed_score * 0.2, 3)

            reasons = []
            if hits != case.expected_keywords:
                miss = [k for k in case.expected_keywords if k not in hits]
                reasons.append(f"缺关键词 {miss}" if miss else "")
            if case.require_all_agents and run.failed_agents:
                reasons.append(f"有 Agent 失败: {run.failed_agents}")
                score = 0.0
            if run.duration_ms > case.max_duration_ms:
                reasons.append(f"超时 {run.duration_ms}ms")
            passed = bool(run.ok) and score >= 0.6 and not (
                case.require_all_agents and run.failed_agents
            )
            if passed:
                report.passed += 1
            total_score += score
            report.results.append(EvalResult(
                case=case.name, passed=passed, score=score,
                reason="；".join(r for r in reasons if r) or "通过",
                duration_ms=run.duration_ms, status=run.status.value,
            ))
        report.score = round(total_score / len(cases), 3) if cases else 0.0
        _logger.info(f"评测完成: 通过 {report.pass_rate}，平均分 {report.score}")
        return report


def build_harness(llm=None, tool_registry=None, **cfg_kwargs) -> AgentHarness:
    """
    一行构造默认团队的 harness（demo / API 入口用）。

    【配置优先级】显式传入的 kwargs > .env（经 Settings）> TeamConfig 默认值。
    这样同一份代码在本地（小并发、短超时）和生产（大并发）能有两套参数，
    而不用改代码。
    """
    from config import get_settings
    s = get_settings()
    cfg = TeamConfig(
        agents=build_default_team(),
        max_workers=cfg_kwargs.pop("max_workers", s.multiagent_max_workers),
        max_total_steps=cfg_kwargs.pop("max_total_steps", s.multiagent_max_total_steps),
        timeout_seconds=cfg_kwargs.pop("timeout_seconds", s.multiagent_timeout_seconds),
        review_enabled=cfg_kwargs.pop("review_enabled", s.multiagent_review_enabled),
        **cfg_kwargs,
    )
    return AgentHarness(config=cfg, llm=llm, tool_registry=tool_registry)
