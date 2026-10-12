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
import time  # 标准库时间模块：打点计时、算总超时 deadline
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout  # 线程池与「等待超时」异常：并行跑一批任务时识别单任务超时
from dataclasses import replace  # dataclass 派生工具：委派时按原 ctx 复制一份、只改委派相关字段
from typing import Any, Callable, Optional  # 类型标注：Any/Callable/Optional 用于钩子与可选参数

from modules.common.logger import get_logger  # 项目统一日志入口：按名字取 logger
from modules.multiagent.agents import (  # 角色层：Agent 基类 / 主管 / 工厂函数
    BaseAgent,  # 所有 Agent 的抽象基类，register 返回的就是它
    RunContext,  # 单次运行的共享上下文（query、上游产出、bus、gate、委派等）
    SupervisorAgent,  # 主管 Agent：负责规划与汇总，register 时自动识别为默认主管
    build_agent,  # 按 AgentSpec 造具体 Agent 实例的工厂
    build_default_team,  # 默认团队花名册：缺省 config 与 build_harness 都用它
)
from modules.multiagent.coordination import (  # 协调层：一致性 / 仲裁 / 总线 / 重规划
    Arbitrator,  # 仲裁者：多数派冲突时给一份裁决结论
    Consensus,  # 一致性检测：判断多份产出是否打架
    MessageBus,  # 成员横向消息总线：负责消息的存与取
    ReplanController,  # 重规划控制：有人失败时让主管只针对缺口补派
)
from modules.multiagent.delegation import (  # 委派层：执行者把活转给队友
    DelegationExecutor,  # 委派执行器：真正发起转派并跑受托方
    DelegationLedger,  # 委派台账：记放行/拒绝，防环防超支
    DelegationPolicy,  # 委派策略：最大深度/总次数/单人次数上限
)
from modules.multiagent.models import (  # 数据模型层：贯穿全链路的 dataclass 与枚举
    AgentOutput,  # 单个 Agent 的执行产出
    AgentRole,  # Agent 角色枚举（supervisor/researcher/...）
    AgentSpec,  # Agent 静态规格（名字/角色/能力/是否启用）
    DelegationRecord,  # 一条委派记录（谁转给谁、结果如何）
    EvalCase,  # 评测用例（query + 期望关键词 + 时间上限）
    EvalReport,  # 评测汇总报告
    EvalResult,  # 单条评测结果
    MessageKind,  # 总线消息类型枚举（finding/objection/...）
    PlanResult,  # 主管规划结果（子任务列表/是否直接答）
    RunResult,  # 一次完整运行的总结果（状态/产出/答案/trace）
    RunStatus,  # 运行状态枚举（success/partial/timeout/failed）
    SubTask,  # 拆分后的子任务（指令/受托人/依赖）
    TeamConfig,  # 团队与治理参数配置
    TraceEvent,  # trace 事件（供 SSE/审计实时推送）
    TraceKind,  # trace 事件类型枚举
)
from modules.multimodal.media import MultiModalInput  # 多模态输入封装：图片/文本等
from modules.hitl.models import HumanAction, InterruptKind  # 人工介入动作与中断类型枚举
from modules.orchestration.bus import OrchestratedBus  # 编排器门面总线：Agent 只能见到它
from modules.orchestration.control import Quota, RateLimiter  # 配额与限流组件
from modules.orchestration.guard import GuardPolicy, PolicyGuard  # 策略护栏与校验器
from modules.orchestration.orchestrator import Orchestrator, OrchestratorConfig  # 编排器与其配置
from modules.orchestration.tracer import SpanStatus, Tracer  # 追踪器与 span 状态枚举

_logger = get_logger("harness")  # 本模块 logger：所有日志统一带 harness 名字，便于过滤


# 【什么】多 Agent 运行时框架
# 【作用】负责编排、并发、超时、步数预算、trace 与评测；Agent 只管思考
class AgentHarness:  # 多 Agent 运行时框架：注册团队 → 跑任务 → 出结果 + trace + 评测
    """多 Agent 运行时：注册团队 → 跑任务 → 出结果 + trace + 评测。"""

    # 【什么】装配一个可复用的运行时实例
    # 【作用】把团队配置、LLM、工具、事件钩子、人工闸门与协调组件一次装配好
    def __init__(
        self,
        config: Optional[TeamConfig] = None,    # 团队与治理参数；不传则用默认团队
        llm=None,                               # 共享的 LLM 实现，供所有 Agent 复用
        tool_registry=None,                     # 工具注册器，为空则全程无工具
        on_event: Optional[Callable[[TraceEvent], None]] = None,   # trace 回调钩子，可接 SSE/审计
        gate=None,                              # 人工闸门；None 表示不启用 HITL
    ):
        self.config = config or TeamConfig(agents=build_default_team())  # 团队配置兜底：不传就用默认团队
        self.llm = llm                                # 保存 LLM 引用，register 时注入给每个 Agent
        self.tools = tool_registry                    # 保存工具注册器，register 时注入给每个 Agent
        self.on_event = on_event                      # 保存 trace 钩子，_emit 时回调
        # 【gate 是什么】人工闸门（modules.hitl.HumanGate）。
        #   不为 None 时，计划/子任务/敏感工具/终稿等环节会停下来等人拍板。
        #   默认 None：行为与未启用 HITL 时完全一致，不引入任何等待。
        self.gate = gate                              # 保存人工闸门，register 时注入给每个 Agent
        self.agents: dict[str, BaseAgent] = {}              # {Agent 名: Agent 实例}，派活时按名字取
        self.supervisor: Optional[SupervisorAgent] = None   # 唯一主管；为 None 时无法编排
        # ------------------------------------------------------------------
        # 【三个协调组件】
        #   consensus 判断产出是否打架，arbitrator 打架时裁决，
        #   replan 有人失败时让主管重新拆活，bus 承载成员之间的横向消息。
        #   它们都只读取配置、不持有运行状态，可以在多次 run 之间复用。
        # ------------------------------------------------------------------
        self.consensus = Consensus(threshold=self.config.consensus_threshold)   # 相似度低于阈值判冲突
        self.bus = MessageBus()                                                 # 成员横向消息总线（存储与检索）
        self.replan = ReplanController(supervisor=None, max_replans=self.config.max_replans)  # 重规划控制器，supervisor 稍后补挂
        # ------------------------------------------------------------------
        # 【委派治理】策略从团队配置搬一次；台账与执行器按「每次运行」建，
        #   原因是委派次数是"这一轮问题"的预算，不该被上一轮吃掉。
        #   默认 delegation_enabled=False：不进委派分支，行为与改造前一致。
        # ------------------------------------------------------------------
        self.delegation_policy = DelegationPolicy.from_config(self.config)  # 委派策略：从团队配置派生一份
        self._delegation_ledger: Optional[DelegationLedger] = None      # 当前运行的台账
        self._delegation_executor: Optional[DelegationExecutor] = None  # 当前运行的执行器
        # ------------------------------------------------------------------
        # 【编排治理四件套的装配】
        #   tracer  观测：每次运行一条 trace，每环节一个 span
        #   guard   安全：权限矩阵 + 敏感词 + 参数上限
        #   limiter 限流：全局一桶 + 每发送者一桶
        #   quota   配额：总量 / 单轮 / 单人 / 跳数
        #   它们都无状态地挂在 harness 上，多次 run 之间复用；
        #   每轮开始的清理在 orchestrator.reset() 里做。
        # ------------------------------------------------------------------
        cfg = self.config  # 局部别名：少敲几个 self.config，下面一长串都从它取
        self.tracer = Tracer(enabled=cfg.orchestration_trace_enabled)  # 追踪器：按配置决定是否真正记录
        self.guard = PolicyGuard(GuardPolicy(  # 策略护栏：内容长度/敏感词/接收者校验
            max_content_chars=cfg.orchestration_max_content_chars,  # 单条消息最大字符数，超出按策略处理
            on_sensitive=cfg.orchestration_on_sensitive,  # 命中敏感词时的动作（block/flag）
            sensitive_words=[  # 敏感词清单：从逗号分隔的配置串解析
                w.strip() for w in (cfg.orchestration_sensitive_words or "").split(",") if w.strip()  # 逐个去空白并过滤空串
            ],
        ))
        self.limiter = RateLimiter(  # 限流器：全局令牌桶 + 每发送者令牌桶
            capacity=cfg.orchestration_burst,  # 全局桶容量：允许的瞬间爆发消息数
            refill_per_sec=cfg.orchestration_rate_per_sec,  # 全局桶每秒补充速率
            # 【为什么单发送者的桶只给全局的 1/4】
            #   4 是默认并行度：这样即使全员一起发，也刚好把全局桶用满，
            #   不会出现"一个人吃光、其他人饿死"。
            per_sender_capacity=max(2, cfg.orchestration_burst // 4),  # 单发送者桶容量：至少 2 条，防 0
            per_sender_refill=max(1.0, cfg.orchestration_rate_per_sec / 4),  # 单发送者补充速率：至少 1 条/秒
        )
        self.quota = Quota(  # 配额器：总量/单轮/单人/跳数四道闸
            per_run=cfg.orchestration_max_per_run,  # 单次运行消息总量上限
            per_round=cfg.orchestration_max_per_round,  # 单轮消息总量上限
            per_sender_round=cfg.orchestration_max_per_sender_round,  # 单发送者单轮上限
            max_hops=cfg.orchestration_max_hops,  # 消息最大转发跳数，防无限接力
        )
        self.orchestrator = Orchestrator(  # 编排器：所有消息发送都要过它的治理流水线
            bus=self.bus, tracer=self.tracer, guard=self.guard,  # 注入总线/追踪/护栏
            limiter=self.limiter, quota=self.quota,  # 注入限流/配额
            config=OrchestratorConfig(  # 编排行为参数
                max_hops=cfg.orchestration_max_hops,  # 最大跳数（与配额一致）
                repeat_limit=cfg.orchestration_repeat_limit,  # 重复内容拦截阈值
                pingpong_limit=cfg.orchestration_pingpong_limit,  # 两人踢皮球拦截阈值
                trace_enabled=cfg.orchestration_trace_enabled,  # 是否记录消息级 span
            ),
        )
        # 【obus 是给 Agent 用的门面】
        #   Agent 只能读消息 + 经编排器发消息，拿不到裸的 MessageBus，
        #   因此不存在"绕过编排器直接塞消息"的路径。
        self.obus = OrchestratedBus(self.bus, self.orchestrator)   # 挂进 RunContext.bus：读消息原样转发，仅 post 受编排治理
        self.last_trace = None       # 最近一次运行的观测链路，供 API/排障查询
        for spec in self.config.agents:  # 遍历团队配置里的每个 Agent 规格
            if spec.enabled:  # 只注册启用状态的，禁用项不占花名册
                self.register(spec)  # 注册进运行时：造实例、挂闸门、同步花名册
        # 仲裁者要在注册完之后选（此时才能按角色找到 reviewer）
        self.arbitrator = Arbitrator(  # 装配仲裁者
            judge=self._first_by_role(AgentRole.REVIEWER) or self.supervisor  # 首选终审员当裁判，没有则主管顶上
        )
        self.replan.supervisor = self.supervisor  # 给重规划控制器补上主管引用（构造时还拿不到）
        # 【升级目标要在注册完后才能定】
        #   编排器判定 A、B 踢皮球时，消息改投给谁——这里指向主管，
        #   保证"卡住的对话"永远有个收口的人，而不是在两个人之间继续转。
        self.orchestrator.supervisor = self.supervisor.name if self.supervisor else ""  # 踢皮球升级目标：有主管用主管名，否则空串

    # ----------------------------------------------------------------------
    # 装配
    # ----------------------------------------------------------------------
    # 【什么】注册一个 Agent 进运行时
    # 【作用】把它放进 agents 字典并挂上闸门；第一个主管自动成为默认主管
    def register(self, spec: AgentSpec) -> BaseAgent:  # 注册入口：接收 AgentSpec，返回造好的 Agent
        """注册一个 Agent。supervisor 自动成为默认主管。"""
        agent = build_agent(spec, llm=self.llm, tool_registry=self.tools)  # 工厂造实例：按角色挑具体类，注入 LLM 与工具
        # ------------------------------------------------------------------
        # 【为什么 gate 用属性注入而不是当构造参数传给 build_agent】
        #   build_agent 的签名是"造一个 Agent"，gate 属于"运行时装配"。
        #   走属性赋值，任何返回 agent 对象的工厂都能接住（包括测试里
        #   被替换掉的假工厂），不用为了新增一个可选设施就去改所有工厂签名。
        #   Agent 内部只在 self.gate 不为 None 时才走审批分支，
        #   默认是 None，等价于没装 HITL。
        # ------------------------------------------------------------------
        agent.gate = self.gate  # 属性注入人工闸门：Agent 内部 gate is not None 时才走审批分支
        self.agents[spec.name] = agent  # 按名字放进字典：后续派活/发消息都按名字索引
        if isinstance(agent, SupervisorAgent) and self.supervisor is None:  # 是主管且还没主管
            self.supervisor = agent  # 第一个注册的主管自动成为默认主管
        # 【注册即同步花名册】
        #   guard 校验"接收者是否存在"时读这份名单。不同步的话，
        #   刚注册的 Agent 给别人发消息会被判 RECEIVER_UNKNOWN，
        #   表现为"明明注册了却发不出去"，且只有开 debug 日志才看得见原因。
        self.guard.set_roster(list(self.agents.keys()))  # 把当前成员名单同步给护栏做接收者校验
        _logger.info(f"Agent 注册: {spec.name} ({spec.role.value})")  # 记一条注册日志，便于核对团队构成
        return agent  # 返回造好的实例，调用方（测试）可直接用

    # 【什么】可派活的成员清单
    # 【作用】只包含启用状态的非主管 Agent，作为主管规划时的候选花名册
    @property  # 只读属性：每次访问都现算，注册/禁用后立即反映
    def roster(self) -> list[AgentSpec]:  # 返回主管可见的成员规格列表
        """主管选人时看到的成员清单（排除主管自己）。"""
        return [  # 列表推导：过滤出可派活的成员
            a.spec for a in self.agents.values()  # 遍历所有已注册 Agent，取其规格
            if a.spec.role != AgentRole.SUPERVISOR and a.spec.enabled  # 排除主管与禁用成员
        ]

    # 【什么】写一条 trace 事件
    # 【作用】追加到 result.trace 并回调外部钩子；钩子自身异常只记 debug，不影响主流程
    def _emit(self, result: RunResult, kind: TraceKind, message: str, agent: str = "", **payload):  # 事件发射器：kind/消息/来源/附载
        """写一条 trace 事件（同时回调外部钩子，可接 SSE/审计）。"""
        event = TraceEvent(kind=kind, message=message, agent=agent, payload=payload)  # 组装事件对象
        result.trace.append(event)  # 追加到本次运行的事件流（最终随 RunResult 返回）
        if self.on_event:  # 外部钩子存在才回调
            try:  # 钩子异常不能炸主流程
                self.on_event(event)  # 回调外部钩子（如 SSE 推送/审计落库）
            except Exception as e:  # 吞掉钩子异常
                _logger.debug(f"事件回调异常（忽略）: {e}")  # 只记 debug，让主流程继续跑

    # ----------------------------------------------------------------------
    # 委派（Delegation）：执行者把活转给别人
    # ----------------------------------------------------------------------
    # 【什么】为一次运行新建委派台账与执行器
    # 【作用】每次 run 都要新建：委派次数是"这一轮问题"的预算，不能跨轮累计
    def _new_delegation(self, ctx: RunContext, result: Optional[RunResult] = None) -> DelegationExecutor:  # 建台账+执行器，挂到 ctx 上
        """建本次运行的委派台账与执行器，并挂到 ctx 上供所有 Agent 使用。

        【为什么执行器里跑子任务用的是 replace(ctx)】
            受托方要知道"我现在在第几层委派、委派链上都有谁"，
            这是递归防环的依据；用 replace 复制一份 ctx 改这三个字段，
            既共享上游产出（同一个 upstream 引用），又不会污染委派方的上下文。
        """
        roster = {a.spec.name: a.spec for a in self.agents.values()}  # 成员花名册字典：按名字索引规格，供台账校验受托方
        ledger = DelegationLedger(self.delegation_policy, roster=roster)  # 新建台账：策略 + 花名册
        self._delegation_ledger = ledger  # 挂到实例上，finally 里收口进 RunResult

        # 【委派事件写进 trace】放行与拒绝都记：
        #   "根本没转"和"想转但被治理拦了"在排查时是两件事
        def _on_event(rec: Optional[DelegationRecord], note: str) -> None:  # 委派事件回调：rec 为 None 表示被治理拒绝
            if result is None:  # 无结果对象可写时直接返回（离线调用场景）
                return  # 什么都不记
            if rec is None:  # 委派被治理拒绝（超深度/成环/超预算）
                self._emit(result, "delegate", f"委派未放行: {note}")  # 记一条拒绝事件
                return  # 结束
            self._emit(result, "delegate", f"{rec.label} {note}", rec.from_agent,  # 记一条放行事件
                       to=rec.to_agent, depth=rec.depth, success=rec.success,  # 附载：受托方/深度/成败
                       task_id=rec.parent_task_id)  # 附载：父任务 id
            # 【委派产出共享到总线】让队友看到"这块已经有人做了"，
            #   否则同一件事可能被第二个人重复查一遍
            if rec.success and self.config.share_findings and rec.content:  # 成功且开了共享且有内容
                self.orchestrator.send(  # 经编排器发消息：受限流/配额管着
                    rec.from_agent, f"（委派 {rec.to_agent} 的产出）{rec.content[:300]}",  # 截 300 字：共享线索不共享全文
                    kind=MessageKind.FINDING, sender_role="worker",  # 消息类型：发现
                    ref_task_id=rec.parent_task_id,  # 关联到父任务
                    run_id=result.run_id if result is not None else "",  # 关联到本次运行
                )

        def _run_fn(agent_name: str, task: SubTask, extra: dict) -> AgentOutput:  # 执行器回调：真正跑受托方
            agent = self.agents.get(agent_name)  # 按名字取受托方实例
            if agent is None:  # 受托方未注册
                return AgentOutput.failed(agent_name, task.id, f"未注册的 Agent: {agent_name}")  # 返回失败产出
            info = extra.get("extra", {})  # 从执行器透传的扩展信息里取委派元数据
            sub_ctx = replace(  # 派生新上下文：共享上游产出，但更新委派三字段
                ctx,  # 原上下文
                delegation_depth=int(info.get("delegation_depth", 0)),  # 当前委派深度
                delegation_chain=list(info.get("delegation_chain", []) or []),  # 委派链名单
                delegated_by=str(info.get("delegated_by", "") or ""),  # 委派方名字
            )
            # ------------------------------------------------------------------
            # 【委派也要进 span 树】挂在根 span 下：
            #   直接跑 agent.run 的话，链路里只看得到"多了一个 agent 在跑"，
            #   看不出它是主管派的还是中途被转来的、转了几层；
            #   包一层 kind="delegate" 的 span 后，
            #   前端按 trace_id 画出的时间线上能看到"这里发生了一次转派"。
            # ------------------------------------------------------------------
            trace = self.last_trace  # 取当前运行的观测链路
            if trace is None:                       # 没进 run（显式委派 API）时没有 trace 可挂
                return agent.run(task, sub_ctx)  # 直接跑，不记 span
            with self.tracer.span(  # 包一层 delegate span
                trace, f"delegate.{info.get('delegated_by', '')}→{agent_name}",  # span 名：谁转给谁
                kind="delegate", parent_id=getattr(trace, "root_id", ""), task_id=task.id,  # 挂根 span 下，关联任务
            ) as dsp:  # 拿到 span 句柄写 IO
                dsp.inputs.update({  # 记录输入：指令/委派方/深度/链
                    "instruction": task.instruction[:200],  # 指令截 200 字
                    "from": str(info.get("delegated_by", "") or ""),  # 委派方
                    "depth": int(info.get("delegation_depth", 0)),  # 深度
                    "chain": list(info.get("delegation_chain", []) or []),  # 委派链
                })
                out = agent.run(task, sub_ctx)  # 真正执行受托方
                dsp.set_io(outputs={"success": out.success, "chars": len(out.content or "")})  # 记录输出：成败/长度
                return out  # 返回产出

        self._delegation_executor = DelegationExecutor(  # 组装委派执行器
            run_fn=_run_fn, ledger=ledger, roster=roster, on_event=_on_event,  # 注入执行回调/台账/花名册/事件回调
        )
        ctx.delegation = self._delegation_executor  # 挂到共享上下文上，Agent.run 里能拿到
        if self.delegation_policy.enabled:  # 委派开关打开才记日志
            _logger.info("委派能力已启用: "  # 记一条启用日志，带三道上限
                         f"depth<={self.delegation_policy.max_depth} "  # 最大深度
                         f"总次数<={self.delegation_policy.max_per_run} "  # 单次运行总次数
                         f"单人<={self.delegation_policy.max_per_agent}")  # 单人次数
        return self._delegation_executor  # 返回执行器，方便调用方直接用

    # 【什么】显式委派一次（API / 主管 / 重规划后都能调）
    # 【作用】给"代码决定转派"的场景一个正式入口，治理与 run 内委派完全一致
    def delegate(self, from_agent: str, to_agent: str, instruction: str, *,  # 显式委派入口：from→to + 指令
                 query: str = "", session_id: str = "", parent_task_id: str = "",  # 可选：原问题/会话/父任务
                 reason: str = "") -> Optional[DelegationRecord]:  # 返回记录；被治理拒绝时返回 None
        """显式委派一次。被治理拒绝时返回 None（调用方照常自己处理）。

        【输入输出对照】
            h.delegate("analyst_1", "researcher_1", "查 Q3/Q4 营收", query="营收环比多少")
            → DelegationRecord(from_agent="analyst_1", to_agent="researcher_1", success=True, ...)
            → 受托方不存在 / 超深度 / 成环 → None
        """
        if self._delegation_executor is None:  # 还没建执行器（run 之外直接调 API 的场景）
            ctx = RunContext(query=query, session_id=session_id, bus=self.obus, gate=self.gate)  # 临时造一个上下文
            self._new_delegation(ctx)  # 建台账与执行器
        assert self._delegation_executor is not None  # 断言：上面一定建好了
        return self._delegation_executor.delegate(  # 交给执行器走统一治理流程
            from_agent=from_agent, to_agent=to_agent, instruction=instruction,  # 谁转给谁、转什么
            parent_task_id=parent_task_id, depth=0, reason=reason,  # 显式委派深度恒为 0（起点）
        )

    # ----------------------------------------------------------------------
    # 编排：DAG 分批
    # ----------------------------------------------------------------------
    # 【什么】把子任务按依赖分成若干批
    # 【作用】Kahn 拓扑分层，同批可并行；处在环里的任务被剔除并作为第二返回值给出
    @staticmethod  # 静态方法：不依赖实例状态，纯函数式分批
    def topo_batches(tasks: list[SubTask]) -> tuple[list[list[SubTask]], list[str]]:  # 返回 (分批结果, 环中任务id)
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
        done: set[str] = set()  # 已完成任务的 id 集合：判断依赖是否就绪
        pending = list(tasks)  # 待分批任务列表：复制一份避免改调用方的列表
        batches: list[list[SubTask]] = []  # 分批结果：每批是一个可并行的任务列表
        while pending:  # 还有任务没分完就继续
            ready = [t for t in pending if all(d in done for d in t.depends_on)]  # 找出依赖全部就绪的任务
            if not ready:  # 一个就绪的都没有 = 存在环或引用了不存在的 id
                # 剩下的全都等不到上游 = 存在环或引用了不存在的 id
                return batches, [t.id for t in pending]  # 返回已分好的批 + 环中任务 id
            ready.sort(key=lambda t: -t.priority)  # 同批内按优先级降序排：重要的先跑
            batches.append(ready)  # 当前就绪任务作为新一批
            done.update(t.id for t in ready)  # 把这批任务标记为已完成
            pending = [t for t in pending if t.id not in done]  # 从待分列表剔除已完成的
        return batches, []  # 全部排完，无环

    # ----------------------------------------------------------------------
    # 主流程
    # ----------------------------------------------------------------------
    # 【什么】跑一次完整的多 Agent 协作
    # 【作用】按「规划→汇派活→总→终审」流水线执行，任一环节异常都降级不抛错
    def run(
        self,
        query: str,  # 用户问题原文
        mm: Optional[MultiModalInput] = None,  # 多模态输入（可选）
        session_id: str = "",  # 会话 ID（透传给工具做隔离）
        run_id: str = "",  # 运行 ID（空则自动生成）
    ) -> RunResult:  # 返回完整运行结果
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
        started = time.time()  # 记录开始时刻：算总耗时与总超时
        result = RunResult(query=query, multimodal=bool(mm and mm.image_count))  # 初始化结果对象：标记是否带图
        if run_id:  # 调用方显式传了 run_id
            result.run_id = run_id  # 覆盖默认生成的 id，便于外部串联日志

        # ------------------------------------------------------------------
        # 【观测链路先起，再挂给编排器】
        #   顺序反过来的话，第一批消息会挂到一个空 trace_id 上，
        #   事后按 trace_id 查链路时，最早那几条消息永远查不到。
        # ------------------------------------------------------------------
        trace = self.tracer.new_trace("multiagent.run", run_id=result.run_id)  # 起一条新 trace，名字定死便于检索
        self.last_trace = trace  # 挂到实例上：委派/显式 API 要靠它挂 span
        result.trace_id = trace.trace_id  # 把 trace_id 写进结果，调用方按它查链路
        self.orchestrator.reset()  # 清空编排器状态：上一轮的消息计数不能带进本轮
        self.orchestrator.bind_trace(trace)  # 把 trace 挂给编排器：消息 span 知道往哪写
        root = self.tracer.start_span(trace, "run", kind="run", query=query[:200])  # 开根 span，记录截断的 query

        # 【bus / gate 挂到 ctx 上】Agent 内部靠它们做横向对话与人工批准，
        #   方法签名就不用层层透传这两个"全队共享设施"了。
        #   注意这里给的是 obus（编排器门面）而不是裸总线：
        #   Agent 读消息走总线，发消息必经编排器。
        ctx = RunContext(query=query, mm=mm, session_id=session_id,  # 组装共享上下文
                         bus=self.obus, gate=self.gate)  # 注入编排门面总线与人工闸门
        # 【委派装配】必须在派活之前：Agent.run 里要靠 ctx.delegation 转派
        self._new_delegation(ctx, result)  # 建本轮委派台账与执行器
        self.bus.clear()      # 上一轮的消息不跨运行，避免旧异议影响本轮
        deadline = started + self.config.timeout_seconds   # 整次运行的绝对截止时刻（秒级 epoch）
        steps_used = 0                                      # 累计已消耗的推理步数，用于卡 max_total_steps

        try:  # 主流程整体包 try：任何环节异常都降级成 FAILED 结果，不向外抛
            if self.supervisor is None:  # 没有主管就无法编排
                result.status = RunStatus.FAILED  # 直接判失败
                result.error = "未注册 supervisor，无法编排"  # 写明原因
                self._emit(result, "error", result.error)  # 记 trace 事件
                return result  # 提前返回

            # ---- 第 1 步：主管规划 ----------------------------------------
            with self.tracer.span(trace, "plan", kind="plan", parent_id=root.span_id) as sp:  # 规划 span，挂根下
                sp.inputs.update({"query": query[:200],  # 记录输入：截断 query
                                  "roster": [a.name for a in self.roster]})  # 与候选成员名单
                plan = self.supervisor.plan(ctx, self.roster)  # 主管拆任务：返回 PlanResult
                sp.set_io(outputs={"subtasks": len(plan.subtasks),  # 记录输出：子任务数
                                   "is_direct": plan.is_direct,  # 是否直接答
                                   "thought": (plan.thought or "")[:120]})  # 思考摘要截 120 字
            result.plan = plan  # 计划写进结果，供 API/前端展示
            self._emit(result, "plan", plan.thought or f"拆出 {len(plan.subtasks)} 个子任务",  # 记规划事件
                       self.supervisor.name, subtasks=[t.model_dump() for t in plan.subtasks])  # 附载完整子任务

            # ---- 第 2 步：直接答就收工 ------------------------------------
            if plan.is_direct:  # 主管判定无需拆任务（寒暄/单步可答）
                result.final_answer = plan.direct_answer  # 直接用主管给的答案
                result.status = RunStatus.SUCCESS  # 标记成功
                result.duration_ms = int((time.time() - started) * 1000)  # 算耗时（毫秒）
                self._emit(result, "done", "主管判定无需拆任务，直接作答")  # 记完成事件
                return result  # 提前返回

            # ---- 第 2.5 步：任务分解的人工确认（HITL，可选）---------------
            #   放在 is_direct 之后：寒暄这类"主管直接答"的场景不该打扰人，
            #   只有真的要派活了才需要人确认"这么拆合理吗"。
            if self.gate is not None and self.gate.policy.kind_enabled(InterruptKind.BEFORE_PLAN):  # 闸门存在且开了计划前审批
                reply = self.gate.ask(  # 发起人工询问（阻塞等人回）
                    kind=InterruptKind.BEFORE_PLAN,  # 中断类型：计划前
                    title=f"确认任务分解（{len(plan.subtasks)} 个子任务）",  # 审批卡片标题
                    reason="计划阶段人工确认已开启",  # 发起原因
                    context={"thought": plan.thought,  # 上下文：主管思考
                             "subtasks": [t.model_dump() for t in plan.subtasks]},  # 与子任务列表
                    run_id=result.run_id, session_id=ctx.session_id,  # 关联运行与会话
                    agent=self.supervisor.name if self.supervisor else "",  # 发起人：主管
                )
                result.interrupt_count += 1  # 累计中断次数：衡量人工介入成本
                self._emit(result, "hitl", f"计划审批: {reply.action.value}", "human",  # 记审批结果事件
                           action=reply.action.value)  # 附载动作类型
                if reply.action == HumanAction.ABORT:  # 人工终止
                    result.status = RunStatus.FAILED  # 判失败
                    result.error = "人工终止（计划阶段）"  # 写明阶段
                    result.final_answer = "本次任务已由人工终止。"  # 给用户的话
                    result.duration_ms = int((time.time() - started) * 1000)  # 算耗时
                    return result  # 提前返回
                if not reply.is_go:  # 未批准（reject/超时默认拒绝）
                    result.status = RunStatus.FAILED  # 判失败
                    result.error = "人工未批准任务分解，未执行"  # 写明原因
                    result.final_answer = "本次任务分解未获人工批准，未执行。"  # 给用户的话
                    result.duration_ms = int((time.time() - started) * 1000)  # 算耗时
                    return result  # 提前返回

            # ---- 第 3 步：DAG 分批 + 并行执行 -----------------------------
            batches, cyclic = self.topo_batches(plan.subtasks)  # 按依赖分批；cyclic 是环中任务 id
            if cyclic:  # 有环或无效依赖
                self._emit(result, "error", f"存在循环依赖或无效依赖，已剔除: {cyclic}")  # 记事件但不中断
            self._emit(result, "dispatch", f"共 {len(batches)} 批，最大并行 {self.config.max_workers}")  # 记派活事件


            for batch_no, batch in enumerate(batches, 1):  # 逐批执行，batch_no 从 1 起
                if time.time() > deadline:  # 撞总超时
                    for t in batch:  # 本批剩余任务全部标记未执行
                        result.outputs.append(  # 追加失败产出
                            AgentOutput.failed(t.assignee, t.id, "总超时，未执行")  # 原因写明总超时
                        )
                    result.status = RunStatus.TIMEOUT  # 状态改为超时
                    self._emit(result, "error", f"第 {batch_no} 批因总超时未执行")  # 记事件
                    break  # 跳出批次循环
                if steps_used >= self.config.max_total_steps:  # 步数预算耗尽
                    for t in batch:  # 本批剩余任务全部标记未执行
                        result.outputs.append(  # 追加失败产出
                            AgentOutput.failed(t.assignee, t.id, "总步数预算耗尽，未执行")  # 原因写明预算耗尽
                        )
                    self._emit(result, "error", f"步数预算耗尽，第 {batch_no} 批起不再执行")  # 记事件
                    break  # 跳出批次循环
                with self.tracer.span(trace, f"dispatch.batch_{batch_no}", kind="dispatch",  # 批次 span
                                      parent_id=root.span_id) as bsp:  # 挂根 span 下
                    bsp.inputs.update({"tasks": [t.id for t in batch], "size": len(batch)})  # 记录输入：任务 id 与批大小
                    batch_outputs = self._run_batch(batch, ctx, deadline, result, batch_no,  # 并行跑本批
                                                    trace=trace, parent_span=bsp.span_id)  # 传入 trace 与父 span
                    bsp.set_io(outputs={  # 记录输出
                        "ok": sum(1 for o in batch_outputs if o.success),  # 成功数
                        "failed": sum(1 for o in batch_outputs if not o.success),  # 失败数
                    })
                for out in batch_outputs:  # 处理本批每个产出
                    # 产出记两处：result.outputs 是对外的账单，ctx.upstream 给下游当输入。
                    # 只写 upstream 不写 outputs 的话，最终答案明明生成了，
                    # result.outputs 却是空的 —— 排查时最难发现的那种 bug。
                    result.outputs.append(out)  # 写进对外账单
                    steps_used += out.steps  # 累计推理步数
                    ctx.upstream[out.task_id] = out  # 挂到上游产出表，下游任务可见
                    # 【把成功产出共享到总线】
                    #   下一批成员能在 prompt 的"同伴消息"里看到前人的发现，
                    #   避免重复查同一件事。只截 400 字：共享的是线索不是全文。
                    if self.config.share_findings and out.success and out.content:  # 开了共享且成功且有内容
                        # 【共享发现也走编排器】
                        #   这是系统自动发的消息，同样要受限流与配额管着：
                        #   一批 10 个 Agent 各贴一条就是瞬间 10 条，
                        #   不过编排器的话，几轮下来总线就被自动消息灌满。
                        self.orchestrator.send(  # 经编排器发消息
                            out.agent, out.content[:400], kind=MessageKind.FINDING,  # 发送者/截断内容/类型
                            sender_role=out.role.value,  # 发送者角色
                            ref_task_id=out.task_id, run_id=result.run_id,  # 关联任务与运行
                        )

            # ---- 第 3.5 步：重规划（可选）---------------------------------
            #   有人失败时才触发；默认关闭（要多调一次 LLM，是真金白银）。
            if self.config.replan_enabled and self.replan.need_replan(result.outputs):  # 开关打开且有失败
                self._try_replan(ctx, result, deadline)  # 进重规划补做

            # ------------------------------------------------------------------
            # 【第 4 步之前：一致性判定 / 仲裁 / 人工裁决】
            #   synth_inputs 是喂给主管汇总的清单：默认是全部执行产出，
            #   出了冲突就把仲裁结论也加进去——这样主管看到的是
            #   "三份矛盾产出 + 一份裁决"，而不是自己在矛盾里硬选一个。
            # ------------------------------------------------------------------
            synth_inputs = list(result.outputs)  # 汇总输入清单：先复制全部产出
            human_final = ""  # 人工裁决的最终文本：非空则跳过机器汇总
            if self.config.consensus_enabled and len(result.outputs) >= 2:  # 开了一致性且产出≥2 份才值得比
                with self.tracer.span(trace, "consensus", kind="consensus",  # 一致性 span
                                      parent_id=root.span_id) as csp:  # 挂根 span 下
                    csp.inputs.update({"outputs": len(result.outputs),  # 记录输入：产出数
                                       "threshold": self.config.consensus_threshold})  # 与相似度阈值
                    cons = self.consensus.detect(result.outputs)  # 跑一致性检测
                    csp.set_io(outputs={"has_conflict": cons.has_conflict,  # 记录输出：是否冲突
                                        "similarity": cons.similarity,  # 相似度
                                        "needs_arbitration": cons.needs_arbitration})  # 是否需仲裁
                result.consensus = cons  # 一致性结论写进结果
                self._emit(result, "consensus", cons.note)  # 记一致性事件
                if cons.needs_arbitration:  # 确实打起来了
                    majority = set(cons.majority)  # 多数派名单转集合：查找快
                    for o in result.outputs:  # 遍历所有产出
                        if o.success and o.agent not in majority:  # 成功的少数派
                            self.orchestrator.send(  # 给少数派发异议消息
                                "consensus",  # 发送者：一致性组件
                                f"你的结论与多数派不一致（相似度低于阈值 {self.config.consensus_threshold}）",  # 告知原因
                                receiver=o.agent, kind=MessageKind.OBJECTION,  # 定向发给少数派，类型异议
                                sender_role="supervisor", ref_task_id=o.task_id,  # 角色主管，关联任务
                                run_id=result.run_id,  # 关联运行
                            )
                    arb = None  # 仲裁结论占位：可能不开仲裁
                    if self.config.arbitration_enabled:  # 开了仲裁
                        with self.tracer.span(trace, "arbitration", kind="arbitration",  # 仲裁 span
                                              parent_id=root.span_id) as arsp:  # 挂根 span 下
                            arb = self.arbitrator.arbitrate(ctx.query, result.outputs, cons)  # 裁判出结论
                            if arb is not None:  # 裁判给出了结论
                                arsp.set_io(outputs={"judge": arb.agent,  # 记录谁裁的
                                                      "chars": len(arb.content or "")})  # 与结论长度
                        if arb is not None:  # 仲裁成功
                            result.arbitration = arb  # 写进结果
                            synth_inputs.append(arb)  # 仲裁结论加进汇总输入：让主管看到"官方结论"
                            self._emit(result, "arbitration", arb.content[:200], arb.agent)  # 记仲裁事件
                    # 【人工裁决】人可以直接改写最终答案（modify），
                    #   也可以只点批准（采纳仲裁结论）。
                    if self.gate is not None and self.gate.policy.kind_enabled(InterruptKind.ON_DISPUTE):  # 闸门存在且开了冲突审批
                        reply = self.gate.ask(  # 发起人工裁决
                            kind=InterruptKind.ON_DISPUTE,  # 中断类型：冲突时
                            title="多个 Agent 结论冲突，请裁决",  # 审批卡片标题
                            reason=cons.note,  # 发起原因：一致性组件的说明
                            context={  # 上下文：给人看足够的信息才能裁
                                "consensus": cons.model_dump(),  # 一致性结论全量
                                "arbitration": arb.content[:800] if arb else "",  # 仲裁结论截 800 字
                                "outputs": [  # 各方观点
                                    {o.agent: o.content[:300]}  # 每个成功产出截 300 字
                                    for o in result.outputs if o.success
                                ],
                            },
                            run_id=result.run_id, session_id=ctx.session_id, agent="consensus",  # 关联运行/会话/发起人
                        )
                        result.interrupt_count += 1  # 累计中断次数
                        self._emit(result, "hitl", f"冲突裁决: {reply.action.value}",  # 记裁决事件
                                   "human", action=reply.action.value)  # 附载动作
                        if reply.action == HumanAction.ABORT:  # 人工终止
                            result.status = RunStatus.FAILED  # 判失败
                            result.error = "人工终止（冲突裁决阶段）"  # 写明阶段
                            result.final_answer = "本次任务已由人工终止。"  # 给用户的话
                            result.duration_ms = int((time.time() - started) * 1000)  # 算耗时
                            return result  # 提前返回
                        if reply.is_go and reply.action == HumanAction.MODIFY and reply.edited.strip():  # 人改写了答案
                            human_final = reply.edited.strip()  # 用人改的版本当最终答案

            # ---- 第 4 步：汇总 --------------------------------------------
            if human_final:  # 人工裁决给了最终文本
                result.final_answer = human_final  # 直接采纳，跳过机器汇总
                self._emit(result, "synthesize", "采用人工裁决结论（跳过汇总）", "human")  # 记事件
            elif any(o.success for o in result.outputs):  # 至少一份成功产出，可以让主管汇总
                with self.tracer.span(trace, "synthesize", kind="synthesize",  # 汇总 span
                                      parent_id=root.span_id) as ssp:  # 挂根 span 下
                    ssp.inputs.update({"inputs": len(synth_inputs)})  # 记录输入份数
                    draft = self.supervisor.synthesize(ctx, synth_inputs)  # 主管汇总出草稿
                    ssp.set_io(outputs={"chars": len(draft or "")})  # 记录草稿长度
                result.final_answer = draft or "（团队未产出有效结论）"  # 草稿为空时给兜底文案
                self._emit(result, "synthesize", "汇总完成", self.supervisor.name)  # 记汇总事件
            elif result.status == RunStatus.TIMEOUT:  # 没有任何成功产出且是超时
                # 【为什么不在这里把 TIMEOUT 覆盖成 FAILED】
                #   两者是截然相反的运维信号：超时要调大 deadline / 减少并行，
                #   Agent 失败要查具体执行者。一律写成 FAILED，监控上就只会
                #   看到"一直失败"，看不出该扩容，问题会一直重复到用户投诉。
                result.final_answer = "抱歉，本次任务因超时未能完成，请缩小问题范围或稍后重试。"  # 给用户的超时话术
                result.error = result.error or "总超时，任务未执行完"  # 错误信息兜底
                self._emit(result, "error", result.error)  # 记错误事件
            else:  # 没有任何成功产出且不是超时 = 全部执行失败
                result.status = RunStatus.FAILED  # 判失败
                result.final_answer = "抱歉，本次任务的所有执行者均未能产出有效结果。"  # 给用户的失败话术
                result.error = "全部 Agent 执行失败"  # 写明原因
                self._emit(result, "error", result.error)  # 记错误事件
                result.duration_ms = int((time.time() - started) * 1000)  # 算耗时
                return result  # 提前返回：失败就不到终审了

            # ---- 第 5 步：终审（可选）-------------------------------------
            if self.config.review_enabled and plan.needs_review:  # 开了终审且计划要求终审
                reviewer = self._first_by_role(AgentRole.REVIEWER)  # 找第一个终审员
                if reviewer is not None:  # 找到了才终审
                    # reviewer 由 _first_by_role 按角色取出，静态类型是 BaseAgent；
                    #   只有 ReviewerAgent 才有 review()，这里按角色已确保是终审员
                    with self.tracer.span(trace, "review", kind="review",  # 终审 span
                                          parent_id=root.span_id) as rsp:  # 挂根 span 下
                        result.review = reviewer.review(ctx, result.final_answer)   # type: ignore  # 调终审员 review；静态类型不认，按角色已确保安全
                        rsp.set_io(outputs={"chars": len(result.review.content or "")})  # 记录评审结论长度
                    self._emit(result, "review", result.review.content[:200], reviewer.name)  # 记终审事件

            # ------------------------------------------------------------------
            # 【第 5.5 步：终稿人工确认】最后一道关卡。
            #   默认兜底是"放行"：终稿已经过汇总与终审，卡住比放行更糟。
            #   人点了 modify 就用改后的文本，点了 reject 就不发出去。
            # ------------------------------------------------------------------
            if self.gate is not None and self.gate.policy.kind_enabled(InterruptKind.BEFORE_FINAL):  # 闸门存在且开了终稿审批
                go, text = self.gate.ask_final(  # 发起终稿审批：返回 (是否放行, 最终文本)
                    result.final_answer,  # 待审批的终稿
                    agent=self.supervisor.name if self.supervisor else "",  # 发起人：主管
                    run_id=result.run_id, session_id=ctx.session_id,  # 关联运行与会话
                )
                result.interrupt_count += 1  # 累计中断次数
                self._emit(result, "hitl", f"终稿审批: {'放行' if go else '未放行'}",  # 记审批事件
                           "human", action="approve" if go else "reject")  # 附载动作
                result.final_answer = text if go else "（该答案未通过人工审核，未向用户发送）"  # 放行用审批文本，否则给未发送话术

            # ---- 第 6 步：定状态 ------------------------------------------
            if result.status != RunStatus.TIMEOUT:  # 超时的状态前面已定，不覆盖
                failed = result.failed_agents  # 取失败 Agent 名单
                result.status = RunStatus.PARTIAL if failed else RunStatus.SUCCESS  # 有失败=部分成功，无=全成功
                if failed:  # 有失败
                    result.error = f"部分 Agent 失败: {failed}"  # 写明失败名单
            result.duration_ms = int((time.time() - started) * 1000)  # 算总耗时
            self._emit(result, "done", f"status={result.status.value} 耗时={result.duration_ms}ms")  # 记完成事件
            return result  # 正常返回

        except Exception as e:  # 主流程任何异常都接住
            _logger.error(f"harness 运行异常: {type(e).__name__}: {e}")  # 记错误日志
            result.status = RunStatus.FAILED  # 判失败
            result.error = f"{type(e).__name__}: {e}"  # 错误信息带异常类型
            result.duration_ms = int((time.time() - started) * 1000)  # 算耗时
            self._emit(result, "error", result.error)  # 记错误事件
            return result  # 返回失败结果
        finally:  # 无论走哪条路径退出都要做的清理
            # ------------------------------------------------------------------
            # 【为什么退出前要作废所有还在等的中断请求】
            #   运行提前结束（abort / 异常 / 超时）时，可能还有请求停在 WAITING。
            #   不唤醒的话，等待它们的线程要一直挂到超时才释放，
            #   团队并行度被白白占住，审批页上还挂着一条永远不会生效的卡片。
            # ------------------------------------------------------------------
            if self.gate is not None:  # 闸门存在才需要清理
                self.gate.cancel_waiting()  # 作废所有还在等的审批请求，释放等待线程
            # ------------------------------------------------------------------
            # 【委派台账收口】无论从哪条路径退出，本次转派过的活都要进结果。
            #   放在 finally 里是因为 run 有多个 return（直接答、人工终止、超时、异常），
            #   和 span 快照同理：失败的那次恰恰最需要看"活被转给了谁"。
            # ------------------------------------------------------------------
            if self._delegation_ledger is not None:  # 本轮建过台账
                result.delegations = list(self._delegation_ledger.records)  # 把委派记录复制进结果
                if self._delegation_ledger.records or self._delegation_ledger.denials:  # 有记录才记日志
                    _logger.info(f"委派汇总: {self._delegation_ledger.summary()}")  # 记委派汇总日志
            # ------------------------------------------------------------------
            # 【收口：关根 span、封 trace、把 span 快照写进结果】
            #   放在 finally 里是因为 run 有多处 return（直接答、人工终止、
            #   超时、异常），任何一条路径退出都要留下完整链路，
            #   否则"失败的那次"反而没有 trace —— 恰恰是最需要 trace 的那些。
            # ------------------------------------------------------------------
            final_status = {  # 运行状态 → span 状态映射
                RunStatus.TIMEOUT: SpanStatus.TIMEOUT,  # 超时映射超时
                RunStatus.FAILED: SpanStatus.ERROR,  # 失败映射错误
            }.get(result.status, SpanStatus.OK)  # 其余默认 OK
            self.tracer.end_span(  # 关闭根 span
                root, final_status,  # span 句柄与最终状态
                outputs={"status": result.status.value, "outputs": len(result.outputs),  # 记录输出：状态/产出数
                         "escalated": len(self.orchestrator.escalations)},  # 与踢皮球升级次数
                error=result.error[:200],  # 错误信息截 200 字
            )
            self.tracer.finish(trace)  # 封 trace：标记结束时刻
            # 【为什么按开始顺序而不是字典顺序导出】
            #   span_id 是随机串，字典顺序毫无意义；
            #   按 start_ms 排序后，前端不用排序就能画出时间线。
            result.spans = [  # 把 span 快照写进结果
                sp.model_dump() for sp in  # 每个 span 转 dict
                sorted(trace.spans.values(), key=lambda s: (s.start_ms, s.span_id))  # 按开始时刻排序，同刻按 id 兜底
            ]

    # ----------------------------------------------------------------------
    # 重规划：一批人没干成时，让主管只针对未完成的部分重新拆活
    # ----------------------------------------------------------------------
    # 【什么】失败后的重规划补做
    # 【作用】每轮重算还失败的任务并补派新任务；撞上 deadline 或没人补位即停
    def _try_replan(self, ctx: RunContext, result: RunResult, deadline: float) -> None:  # 重规划补做入口
        """最多重规划 max_replans 次。

        【为什么不把失败产出从 outputs 里删掉】
            删掉会让"曾经失败过"这件事从结果里消失，
            排查时看到的就是一份干净但失真的记录。
            保留 + 追加重试产出，才能还原"第一次谁失败了、换谁重试、成了没有"。

        【为什么每轮都重新算 failed】
            上一轮失败的任务可能这一轮已经由别人完成，
            重新算才能保证"只补还缺的"，不做重复功。
        """
        for attempt in range(1, self.config.max_replans + 1):  # 最多重规划 max_replans 次
            failed = [o for o in result.outputs if not o.success]  # 重算当前还失败的任务
            if not failed:  # 没有失败的了
                break  # 收工
            if time.time() > deadline:  # 撞总超时
                self._emit(result, "replan", "已达总超时，停止重规划")  # 记事件
                break  # 收工

            plan2 = self.replan.build_replan(ctx, failed, self.roster)  # 让主管只针对失败任务重新拆活
            tasks = list(plan2.subtasks) if plan2 and plan2.subtasks else []  # 取新任务列表
            if not tasks:  # 主管没给出新计划（可能规划调用失败）：退化为换个人重试
                fallback = self.replan.fallback_task(failed, self.roster)  # 挑一个换人的兜底任务
                tasks = [fallback] if fallback else []  # 有就用，没有就空
            if not tasks:  # 连兜底都没有
                break  # 收工

            result.replan_count += 1  # 累计重规划次数
            self._emit(result, "replan",  # 记重规划事件
                       f"第 {attempt} 次重规划：新增 {len(tasks)} 个任务，"  # 第几次、新增几个
                       f"待补 {len(failed)} 项失败")  # 还缺几项
            batches, cyclic = self.topo_batches(tasks)  # 新任务按依赖分批
            if cyclic:  # 有环
                self._emit(result, "replan", f"重规划任务存在无效依赖，已剔除: {cyclic}")  # 记事件
            for i, batch in enumerate(batches, 1):  # 逐批执行新任务
                if time.time() > deadline:  # 撞总超时
                    self._emit(result, "replan", "重规划批次因总超时未执行")  # 记事件
                    break  # 跳出批次循环
                outs = self._run_batch(batch, ctx, deadline, result, batch_no=100 * attempt + i)  # 跑本批；批次号避开主流程
                for out in outs:  # 处理每个产出
                    result.outputs.append(out)  # 追加进对外账单（保留失败记录，不删）
                    ctx.upstream[out.task_id] = out  # 挂到上游产出表

    # 【什么】并行跑一批任务
    # 【作用】用线程池并发执行，单个任务的超时或异常都转成 failed，不拖垮同批其他任务
    def _run_batch(
        self,
        batch: list[SubTask],  # 本批任务列表
        ctx: RunContext,  # 共享上下文
        deadline: float,  # 总超时时刻
        result: RunResult,  # 结果对象（写 trace/中断计数）
        batch_no: int,  # 批次号（日志用）
        trace: Optional[Any] = None,      # 本次运行的观测链路；为 None 时不记 span（离线调用场景）
        parent_span: str = "",            # 本批的 span id，Agent 的 span 挂在它下面
    ) -> list[AgentOutput]:  # 返回本批产出列表
        """并行跑一批任务；单个任务超时/异常都转成 failed，不影响同批其他任务。"""
        outputs: list[AgentOutput] = []  # 本批产出收集器

        # 【什么】单个任务的执行包装
        # 【作用】先做子任务级人工批准（可选），再记首尾两个 trace 事件并执行
        def _one(task: SubTask) -> AgentOutput:  # 单个任务执行函数（给线程池 submit 用）
            agent = self.agents.get(task.assignee)  # 按受托人名字取 Agent 实例
            if agent is None:  # 受托人未注册
                return AgentOutput.failed(task.assignee, task.id, f"未注册的 Agent: {task.assignee}")  # 返回失败产出

            # ------------------------------------------------------------------
            # 【子任务级人工批准】只在策略开启 BEFORE_TASK 时生效。
            #   modify 时人改写的是"指令"（不是答案），
            #   所以用改后的 instruction 重新构造一个 task 副本再执行，
            #   不修改原对象 —— 原计划要留着进 trace 做前后对比。
            # ------------------------------------------------------------------
            if self.gate is not None and self.gate.policy.kind_enabled(InterruptKind.BEFORE_TASK):  # 闸门存在且开了任务前审批
                reply = self.gate.ask(  # 发起人工询问
                    kind=InterruptKind.BEFORE_TASK,  # 中断类型：任务前
                    title=f"执行子任务 {task.id}",  # 审批卡片标题
                    reason="子任务执行前的人工确认已开启",  # 发起原因
                    context={"instruction": task.instruction, "assignee": task.assignee},  # 上下文：指令与受托人
                    run_id=result.run_id, session_id=ctx.session_id, agent=agent.name,  # 关联运行/会话/执行者
                )
                result.interrupt_count += 1  # 累计中断次数
                self._emit(result, "hitl", f"任务审批 {task.id}: {reply.action.value}",  # 记审批事件
                           "human", action=reply.action.value, task_id=task.id)  # 附载动作与任务 id
                if not reply.is_go:  # 未批准
                    note = "人工终止" if reply.action == HumanAction.ABORT else "人工未批准执行"  # 区分终止与拒绝
                    return AgentOutput.failed(task.assignee, task.id, note)  # 返回失败产出
                if reply.action == HumanAction.MODIFY and reply.edited.strip():  # 人改写了指令
                    task = task.model_copy(update={"instruction": reply.edited.strip()})  # 用改后指令造副本，不动原对象
                    _logger.info(f"子任务 {task.id} 指令已被人工修改")  # 记日志

            self._emit(result, "agent_start", task.instruction[:120], agent.name, task_id=task.id)  # 记开始事件
            if trace is not None:  # 有观测链路才记 span
                # 【为什么要把一次 Agent 执行单独包一个 span】
                #   Agent 内部有人工批准、工具调用、多步推理，
                #   出问题时第一件事是判断"慢在哪一环"——
                #   有这个 span 就能直接看出是模型慢、工具慢，还是卡在等人批准。
                with self.tracer.span(trace, f"agent.{agent.name}", kind="agent",  # Agent span
                                      parent_id=parent_span or trace.root_id,  # 挂批次 span 下，没传就挂根
                                      task_id=task.id) as asp:  # 关联任务 id
                    asp.inputs.update({"instruction": task.instruction[:200]})  # 记录输入：截断指令
                    out = agent.run(task, ctx)  # 真正执行
                    asp.set_io(outputs={"success": out.success, "steps": out.steps,  # 记录输出：成败/步数
                                        "chars": len(out.content or ""),  # 内容长度
                                        "error": out.error[:120]})  # 错误截 120 字
            else:  # 无观测链路（离线调用）
                out = agent.run(task, ctx)  # 直接执行
            self._emit(result, "agent_end",  # 记结束事件
                       f"{'成功' if out.success else '失败'} 耗时={out.duration_ms}ms",  # 成败与耗时
                       agent.name, task_id=task.id, success=out.success)  # 执行者/任务 id/成败
            return out  # 返回产出

        # 线程数取「配置并发上限」与「本批任务数」的较小值：不多开空转线程
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(batch))) as pool:  # 建线程池
            futures = {pool.submit(_one, t): t for t in batch}  # 全部提交：future → 任务 映射
            for future, task in futures.items():  # 按提交顺序收结果（注：不是完成顺序）
                wait = max(deadline - time.time(), 1.0)   # 剩余时间不足 1 秒时也至少等 1 秒，避免 0 秒立刻超时
                try:  # 收单个 future 的结果
                    outputs.append(future.result(timeout=wait))  # 带超时等待
                except FutureTimeout:  # 等待超时（任务还在跑）
                    _logger.error(f"任务超时: {task.id}（等待 {wait:.1f}s 未返回）")  # 记错误日志
                    outputs.append(AgentOutput.failed(task.assignee, task.id, "任务执行超时"))  # 转成失败产出
                except Exception as e:  # 任务内部抛异常
                    outputs.append(  # 转成失败产出
                        AgentOutput.failed(task.assignee, task.id, f"{type(e).__name__}: {e}")  # 带异常类型
                    )
        _logger.info(f"第 {batch_no} 批完成: {len(outputs)} 个产出")  # 记批次完成日志
        return outputs  # 返回本批产出

    # 【什么】按角色找第一个启用的 Agent
    # 【作用】用于定位终审员或仲裁者；找不到返回 None，由调用方决定是否跳过该环节
    def _first_by_role(self, role: AgentRole) -> Optional[BaseAgent]:  # 按角色找 Agent
        for agent in self.agents.values():  # 遍历所有已注册 Agent
            if agent.spec.role == role and agent.spec.enabled:  # 角色匹配且启用
                return agent  # 返回第一个命中的
        return None  # 没找到

    # ----------------------------------------------------------------------
    # 评测
    # ----------------------------------------------------------------------
    # 【什么】跑一批评测用例并生成报告
    # 【作用】用关键词命中率、完成率、速度分合成得分，作为回归测试门禁
    def evaluate(self, cases: list[EvalCase]) -> EvalReport:  # 评测入口：给一批用例
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
        report = EvalReport(total=len(cases))  # 初始化报告：总用例数
        total_score = 0.0  # 总分累加器
        for case in cases:  # 逐个用例跑
            run = self.run(case.query)  # 跑一次完整协作
            hits = [  # 命中的关键词列表
                kw for kw in case.expected_keywords  # 遍历期望关键词
                if kw.lower() in run.final_answer.lower()  # 大小写不敏感地匹配最终答案
            ]
            kw_score = len(hits) / len(case.expected_keywords) if case.expected_keywords else 1.0  # 关键词命中率；没期望词给满分
            done_ratio = (                                          # 完成率：成功产出数 / 总产出数
                sum(1 for o in run.outputs if o.success) / len(run.outputs)  # 成功数除以总数
                if run.outputs else 0.0  # 没产出给 0
            )
            speed_score = max(0.0, 1 - run.duration_ms / max(case.max_duration_ms, 1))  # 速度分：越快越高，超时到 0
            score = round(kw_score * 0.5 + done_ratio * 0.3 + speed_score * 0.2, 3)  # 加权合成总分，保留 3 位

            reasons = []  # 不通过原因收集器
            if hits != case.expected_keywords:  # 关键词没全中
                miss = [k for k in case.expected_keywords if k not in hits]  # 缺哪些词
                reasons.append(f"缺关键词 {miss}" if miss else "")  # 记原因
            if case.require_all_agents and run.failed_agents:  # 要求全员成功但有失败
                reasons.append(f"有 Agent 失败: {run.failed_agents}")  # 记原因
                score = 0.0  # 直接判 0 分
            if run.duration_ms > case.max_duration_ms:  # 超时
                reasons.append(f"超时 {run.duration_ms}ms")  # 记原因
            passed = bool(run.ok) and score >= 0.6 and not (   # 通过线：运行 ok 且综合分不低于 0.6
                case.require_all_agents and run.failed_agents  # 且全员成功要求未被破坏
            )
            if passed:  # 通过了
                report.passed += 1  # 通过数加一
            total_score += score  # 累加总分
            report.results.append(EvalResult(  # 记单条结果
                case=case.name, passed=passed, score=score,  # 用例名/是否通过/得分
                reason="；".join(r for r in reasons if r) or "通过",  # 原因拼接；空则写"通过"
                duration_ms=run.duration_ms, status=run.status.value,  # 耗时与运行状态
            ))
        report.score = round(total_score / len(cases), 3) if cases else 0.0  # 平均分；没用例给 0
        _logger.info(f"评测完成: 通过 {report.pass_rate}，平均分 {report.score}")  # 记评测汇总日志
        return report  # 返回报告


# 【什么】一行构造默认团队的 harness
# 【作用】给 demo 与 API 入口用；治理参数优先取显式 kwargs，其次 .env，最后 TeamConfig 默认值
def build_harness(llm=None, tool_registry=None, gate=None, **cfg_kwargs) -> AgentHarness:  # 一行构造函数：LLM/工具/闸门 + 治理参数
    """
    一行构造默认团队的 harness（demo / API 入口用）。

    【配置优先级】显式传入的 kwargs > .env（经 Settings）> TeamConfig 默认值。
    这样同一份代码在本地（小并发、短超时）和生产（大并发）能有两套参数，
    而不用改代码。

    Args:
        gate: 人工闸门。不传时按配置决定是否自动建一个：
              HITL_ENABLED=true 才建；否则保持 None（完全不引入等待）。
              测试与 CI 里推荐显式传带 auto_resolver 的 gate，避免真的阻塞。
    """
    from config import get_settings  # 延迟导入：避免模块加载时就读 .env
    s = get_settings()  # 取全局配置单例
    cfg = TeamConfig(  # 组装团队配置
        agents=build_default_team(),  # 默认团队花名册
        max_workers=cfg_kwargs.pop("max_workers", s.multiagent_max_workers),  # 并发上限：显式 > .env
        max_total_steps=cfg_kwargs.pop("max_total_steps", s.multiagent_max_total_steps),  # 总步数预算
        timeout_seconds=cfg_kwargs.pop("timeout_seconds", s.multiagent_timeout_seconds),  # 总超时秒数
        review_enabled=cfg_kwargs.pop("review_enabled", s.multiagent_review_enabled),  # 是否开终审
        consensus_enabled=cfg_kwargs.pop("consensus_enabled", s.multiagent_consensus_enabled),  # 是否开一致性检测
        consensus_threshold=cfg_kwargs.pop(  # 一致性相似度阈值
            "consensus_threshold", s.multiagent_consensus_threshold),  # 显式 > .env
        arbitration_enabled=cfg_kwargs.pop(  # 是否开仲裁
            "arbitration_enabled", s.multiagent_arbitration_enabled),  # 显式 > .env
        share_findings=cfg_kwargs.pop("share_findings", s.multiagent_share_findings),  # 是否共享发现到总线
        replan_enabled=cfg_kwargs.pop("replan_enabled", s.multiagent_replan_enabled),  # 是否开重规划
        max_replans=cfg_kwargs.pop("max_replans", s.multiagent_max_replans),  # 最大重规划次数
        # ---- 委派：与上面同一套优先级（显式 > .env > 默认值），默认关闭 ----
        delegation_enabled=cfg_kwargs.pop("delegation_enabled", s.multiagent_delegation_enabled),  # 是否开委派
        delegation_max_depth=cfg_kwargs.pop("delegation_max_depth", s.multiagent_delegation_max_depth),  # 委派最大深度
        delegation_max_per_run=cfg_kwargs.pop(  # 委派单次运行总次数
            "delegation_max_per_run", s.multiagent_delegation_max_per_run),  # 显式 > .env
        delegation_max_per_agent=cfg_kwargs.pop(  # 委派单人次数上限
            "delegation_max_per_agent", s.multiagent_delegation_max_per_agent),  # 显式 > .env
        hitl_enabled=cfg_kwargs.pop("hitl_enabled", s.hitl_enabled),  # 是否开人工介入
        # ---- 编排治理：与上面的协调参数同一套优先级（显式 > .env > 默认值）----
        orchestration_enabled=cfg_kwargs.pop("orchestration_enabled", s.orchestration_enabled),  # 是否开编排治理
        orchestration_max_hops=cfg_kwargs.pop("orchestration_max_hops", s.orchestration_max_hops),  # 消息最大跳数
        orchestration_max_per_run=cfg_kwargs.pop(  # 单次运行消息总量
            "orchestration_max_per_run", s.orchestration_max_per_run),  # 显式 > .env
        orchestration_max_per_round=cfg_kwargs.pop(  # 单轮消息总量
            "orchestration_max_per_round", s.orchestration_max_per_round),  # 显式 > .env
        orchestration_max_per_sender_round=cfg_kwargs.pop(  # 单发送者单轮上限
            "orchestration_max_per_sender_round", s.orchestration_max_per_sender_round),  # 显式 > .env
        orchestration_rate_per_sec=cfg_kwargs.pop(  # 全局限流速率
            "orchestration_rate_per_sec", s.orchestration_rate_per_sec),  # 显式 > .env
        orchestration_burst=cfg_kwargs.pop("orchestration_burst", s.orchestration_burst),  # 全局突发容量
        orchestration_repeat_limit=cfg_kwargs.pop(  # 重复内容拦截阈值
            "orchestration_repeat_limit", s.orchestration_repeat_limit),  # 显式 > .env
        orchestration_pingpong_limit=cfg_kwargs.pop(  # 踢皮球拦截阈值
            "orchestration_pingpong_limit", s.orchestration_pingpong_limit),  # 显式 > .env
        orchestration_sensitive_words=cfg_kwargs.pop(  # 敏感词清单
            "orchestration_sensitive_words", s.orchestration_sensitive_words),  # 显式 > .env
        orchestration_on_sensitive=cfg_kwargs.pop(  # 命中敏感词时的动作
            "orchestration_on_sensitive", s.orchestration_on_sensitive),  # 显式 > .env
        orchestration_max_content_chars=cfg_kwargs.pop(  # 单条消息最大字符数
            "orchestration_max_content_chars", s.orchestration_max_content_chars),  # 显式 > .env
        orchestration_trace_enabled=cfg_kwargs.pop(  # 是否开消息级 trace
            "orchestration_trace_enabled", s.orchestration_trace_enabled),  # 显式 > .env
        **cfg_kwargs,  # 其余未识别的 kwargs 原样透传给 TeamConfig（会报错的字段由 pydantic 拦）
    )
    # ------------------------------------------------------------------
    # 【什么时候自动建闸门】只在配置明确开启时建。
    #   不传 gate 且配置关闭 → 保持 None，链路里所有 HITL 分支整块跳过，
    #   连一次布尔判断的开销都省了（gate is not None 直接短路）。
    # ------------------------------------------------------------------
    if gate is None and cfg.hitl_enabled:  # 没显式传闸门且配置开了 HITL
        # ------------------------------------------------------------------
        # 【为什么用进程级共享闸门而不是在这里 new 一个】
        #   审批端点（/hitl/reply）必须往同一个 gate 提交，
        #   人点了批准链路才收得到。各自 new 一个就会各等各的，
        #   表现为"点了没反应"，而且很难查 —— 两边日志都正常。
        # ------------------------------------------------------------------
        from modules.hitl import get_global_gate  # 延迟导入：避免循环依赖
        gate = get_global_gate(enabled=True)  # 取进程级共享闸门
        _logger.info(f"HITL 已启用：中断点={s.hitl_kinds or '全部'} "  # 记启用日志：中断点
                     f"敏感工具={s.hitl_sensitive_tools or '无'}（进程级共享闸门）")  # 与敏感工具
    return AgentHarness(config=cfg, llm=llm, tool_registry=tool_registry, gate=gate)  # 装配完成，返回运行时
