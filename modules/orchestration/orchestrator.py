"""
中心化编排器
============
所有 Agent 通信的唯一入口。发出去的每条消息都要经过：
    校验（权限/敏感/参数） → 限流 → 配额 → 防环 → 投递 → 上报

【为什么必须中心化】
    点对点自由通信在生产上会出三类事故：
        1. 死循环：A 问 B、B 回问 A，谁也不收口，token 一直烧
        2. 不可观测：消息散在各 Agent 的调用里，出问题时拼不出链路
        3. 无法管控：想加一条"不许外发客户手机号"的规则，得改所有 Agent
    中心化之后，这三件事各有一个唯一的落点：本文件。

【防死循环的三道闸】
    1. 跳数上限（max_hops）      一条对话链最多几跳
    2. 每轮配额（per_sender_round）  同一个人在一轮里最多发几条
    3. 重复/乒乓检测             内容重复或 A→B→A→B 模式直接升级给主管
"""
from __future__ import annotations  # 注解延迟求值：允许类型标注前置引用尚未定义的名字

import time  # 计时：给每次裁决算耗时（elapsed_ms）
from collections import defaultdict  # 带默认值的字典：计数器免初始化、免判空
from typing import Optional  # 可选类型标注

from pydantic import BaseModel, Field  # 数据模型基类与字段声明：配置对象靠它做启动期校验

from modules.common.logger import get_logger  # 统一日志入口
from modules.multiagent.models import MessageKind  # 消息类型枚举（finding/objection/clarify/delegate）
from modules.orchestration.control import Quota, RateLimiter  # 配额与限流：负载控制的两道闸门
from modules.orchestration.guard import GuardPolicy, PolicyGuard  # 权限/敏感词/参数校验
from modules.orchestration.protocol import (  # 信封与裁决协议
    DeliveryStatus,  # 投递状态枚举（pending/delivered/rejected/dropped/escalated）
    Envelope,  # 标准化消息信封
    RouteAction,  # 裁决动作枚举（deliver/reject/drop/escalate）
    RouteDecision,  # 裁决结果对象（本文件所有出口的返回类型）
    RejectCode,  # 拒绝原因码（监控按它聚合）
    envelope_to_log_dict,  # 信封压平成日志字典的工具函数
)
from modules.orchestration.tracer import SpanStatus, Trace, Tracer  # 观测链路：span 状态/链路/写入器

_logger = get_logger("orchestrator")  # 本模块日志器：名字即模块名，日志可溯源


# ---------------------------------------------------------------------------
# 【什么】编排器的治理参数（pydantic 模型）
# 【作用】把"防死循环三道闸 + 升级对象 + 观测开关"集中成一份可校验的配置
# 【为什么用 pydantic】构造时自动做类型/取值校验（ge=0 等），
#   配错参数在启动期就报错，而不是跑到一半才行为诡异
# ---------------------------------------------------------------------------
class OrchestratorConfig(BaseModel):  # 编排器配置：全部字段有默认值，不传也能跑
    """编排器的治理参数。"""

    max_hops: int = Field(default=2, ge=0, description="一条对话链最多几跳，超出升级主管")  # 闸①：跳数上限
    repeat_limit: int = Field(default=2, ge=1,
                              description="同一对收发方的相同内容最多出现几次，超出判定为无效重复")  # 闸②：重复内容上限
    pingpong_limit: int = Field(default=2, ge=1,
                                description="A→B 与 B→A 各出现几次判定为踢皮球")  # 闸③：乒乓次数上限
    escalate_to: str = Field(default="", description="升级给谁；为空则用主管名")  # 升级目标：空则回落到 supervisor
    trace_enabled: bool = Field(default=True, description="是否为每条消息写观测 span")  # 观测开关：关掉可省写 span 的开销


# ---------------------------------------------------------------------------
# 【什么】中心化编排器本体
# 【作用】所有消息的唯一出入口：校验 → 限流 → 配额 → 防环 → 投递 → 上报，六道工序在这里串完
# 【为什么 Agent 不能直接拿总线】见模块 docstring；Agent 拿到的 bus 只是本类的门面（OrchestratedBus）
# ---------------------------------------------------------------------------
class Orchestrator:  # 消息的唯一出入口
    """消息的唯一出入口。Agent 拿到的 bus 只是它的门面。"""

    # 【什么】构造：装配总线、观测、校验、限流、配额五件套，并建两本防环账
    # 【作用】没传的组件一律用默认实现兜底，保证"空手 new 一个 Orchestrator(bus) 就能跑"
    # 【为什么】编排器是必经之路，必须做到零配置可用；要定制时再逐个注入
    def __init__(
        self,
        bus,  # 消息总线（存储侧）：投递与回查父消息都靠它
        tracer: Optional[Tracer] = None,  # 观测写入器；None = 不记 span
        guard: Optional[PolicyGuard] = None,  # 校验器；None = 用默认策略现场建一个
        limiter: Optional[RateLimiter] = None,  # 限流器；None = 用默认规格现场建一个
        quota: Optional[Quota] = None,  # 配额器；None = 按 config 的 max_hops 现场建一个
        config: Optional[OrchestratorConfig] = None,  # 治理参数；None = 全默认
        supervisor: str = "",  # 主管名：升级没指定目标时的兜底收口人
    ):
        self.bus = bus                                        # 消息总线（存储侧）
        self.tracer = tracer                                  # 观测（可为空=不记 span）
        self.guard = guard or PolicyGuard(GuardPolicy())  # 校验器兜底：没传就建默认策略的
        self.limiter = limiter or RateLimiter()  # 限流器兜底：没传就建默认规格的
        self.quota = quota or Quota(max_hops=(config.max_hops if config else 2))  # 配额器兜底：跳数先对齐 config
        self.config = config or OrchestratorConfig()  # 配置兜底：没传就用全默认配置
        self.supervisor = supervisor  # 记住主管名：升级时用
        # 【为什么配额的 max_hops 要与配置对齐一次】
        #   Quota 自己有 max_hops 字段，构造时若没传就用自己的默认 2。
        #   这里显式同步一次，避免"改了 config 但配额没跟上"的静默错位。
        self.quota.max_hops = self.config.max_hops  # 把配置的跳数上限写进配额器，保证两处一致

        self.round_no: int = 0                     # 当前协作轮次
        self.trace: Optional[Trace] = None         # 当前运行绑定的观测链路
        self.escalations: list[dict] = []          # 被判定为无效交互的记录，供主管/人工查看
        # 内容去重：{("发送者","接收者","内容"): 次数}
        self._content_hits: dict[tuple[str, str, str], int] = defaultdict(int)  # 三元组计数账：查"重复发同一内容"用
        # 方向计数：{("发送者","接收者"): 次数}（不分轮次，专抓乒乓）
        self._pair_hits: dict[tuple[str, str], int] = defaultdict(int)  # 二元组计数账：抓"A→B→A→B 踢皮球"用

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def bind_trace(self, trace: Optional[Trace]) -> None:  # 绑定/解绑观测链路（传 None 即解绑）
        """绑定本次运行的观测链路，之后每条消息都挂到它的根 span 下。"""
        self.trace = trace  # 之后 send 里每条消息的 span 都挂到这条 trace 的根下

    def new_round(self) -> int:  # 轮次号 +1 并返回新值
        """进入下一轮协作，返回新的轮次号。

        【为什么要有轮次概念】
            上一轮的异议、求助如果不随轮次失效，新一批 Agent 开工时
            会读到与当前任务无关的旧消息——表现为"莫名其妙被批评"。
        """
        self.round_no += 1  # 轮次自增：之后发出的消息 round 字段跟着变
        return self.round_no  # 把新轮次号还给调用方（harness 记日志/传给其他组件）

    def reset(self) -> None:  # 一次新运行开始前清零全部运行时状态
        """开始一次新运行：清空计数与升级记录。"""
        self.round_no = 0  # 轮次归零
        self.escalations.clear()  # 清空升级记录：不带上次运行的包袱
        self._content_hits.clear()  # 清空内容去重账
        self._pair_hits.clear()  # 清空方向计数账
        self.quota.reset_run()  # 配额三本账（总/轮/人轮）也归零
        self.trace = None  # 解绑旧 trace：防新运行的消息挂到旧链路上

    # ------------------------------------------------------------------
    # 核心：发一条消息
    # ------------------------------------------------------------------
    # 【什么】消息的唯一发送入口（六道工序串行：校验→限流→配额→防环→投递→上报）
    # 【作用】任何 Agent 的任何消息都必须从这里过，每一关不过都有明确的裁决返回
    # 【为什么六道工序是这个顺序】先便宜后贵：校验查表最快、防环要查账、投递最贵放最后
    def send(
        self,
        sender: str,  # 发送者名
        content: str,  # 消息正文
        receiver: str = "*",  # 接收者；"*" = 广播给所有人
        kind: MessageKind = MessageKind.FINDING,  # 消息类型，默认"发现/结论"类
        *,
        sender_role: str = "",  # 发送者角色：权限矩阵按它判定
        parent_id: str = "",  # 回复的上一条消息 id；空 = 主动发起
        ref_task_id: str = "",  # 关联的任务 id
        meta: Optional[dict] = None,  # 结构化附加数据
        run_id: str = "",  # 所属运行 id
    ) -> RouteDecision:
        """发一条消息，返回编排器的裁决。

        【返回值为什么必须是 RouteDecision 而不是 bool】
            调用方需要区分"被权限拒了"（要去申请权限）和"被限流丢了"
            （等一会儿重试即可）。布尔值把这两种情况压成一个 False，
            线上就只能靠猜。
        """
        t0 = time.time()  # 裁决计时起点：返回值里要带 elapsed_ms
        trace = self.trace  # 取出当前绑定的观测链路（可能为 None）
        parent = self._find(parent_id)  # 回查父消息：算 hop、挂 parent_id 都要它

        env = Envelope(  # 把散参数打包成标准信封：之后所有关卡只对信封说话
            sender=sender, receiver=receiver, kind=kind, content=content or "",  # 基本信息；None 正文兜底成空串
            sender_role=sender_role, ref_task_id=ref_task_id, meta=meta or {},  # 角色/任务/附加数据；meta 兜底成空 dict
            round=self.round_no,  # 打上当前轮次：配额的"每轮上限"按它记账
            # 【hop 怎么算】回复别人的消息才算一跳；主动发起的消息是第 0 跳。
            hop=(parent.hop + 1) if parent is not None else 0,  # 有父消息则跳数 +1，否则从 0 起步
            parent_id=parent.id if parent is not None else "",  # 挂上父消息 id：拼得出对话链
            trace_id=trace.trace_id if trace else "",  # 挂观测链路 id：没绑 trace 则为空
            run_id=run_id,  # 挂运行 id：与 RunResult 对上号
        )

        # ---- 第 1 关：权限 / 敏感词 / 参数 --------------------------------
        verdict = self.guard.check(env)  # 跑校验五连：协议→身份→权限→敏感→参数
        if not verdict.passed:  # 校验没过：这条消息到此为止
            env.mark(DeliveryStatus.REJECTED, verdict.reason)  # 信封打上"被拒"状态与理由（留痕）
            self._record(env, verdict.code, SpanStatus.ERROR)  # 写一条 ERROR span：监控能看到被拒事件
            return RouteDecision.blocked(RouteAction.REJECT, verdict.code, verdict.reason,
                                         env, self._ms(t0))  # 裁决=拒绝：调用方可见，要去改权限/内容

        # ---- 第 2 关：限流 ------------------------------------------------
        ok, by = self.limiter.acquire(sender)  # 过两道令牌闸：全局闸 + 发送者个人闸
        if not ok:  # 任一闸没过：静默丢弃（不打断发送方）
            env.mark(DeliveryStatus.DROPPED, f"限流（{by}）")  # 信封打"被丢弃"，理由带被哪道闸拦的
            self._record(env, RejectCode.RATE_LIMITED, SpanStatus.TIMEOUT)  # 记 span：TIMEOUT 表示"负载原因"
            return RouteDecision.blocked(RouteAction.DROP, RejectCode.RATE_LIMITED,
                                         f"被 {by} 限流，消息未投递", env, self._ms(t0))  # 裁决=丢弃

        # ---- 第 3 关：配额与跳数 ------------------------------------------
        allowed, why = self.quota.check(sender, env.round, env.hop)  # 四道配额闸只查不扣
        if not allowed:  # 配额没过
            # 【跳数超限为什么要升级而不是丢弃】
            #   跳数超限意味着"两个人已经来回好几轮还没结论"，
            #   这是个需要人/主管知道的信号；丢掉就永远发现不了这对组合有问题。
            if env.hop > self.config.max_hops:  # 超的是跳数闸：升级为无效交互
                env.mark(DeliveryStatus.ESCALATED, why)  # 信封打"已升级"状态
                self._escalate(env, RejectCode.MAX_HOPS, why)  # 改投主管：消息不丢，换收口人
                return RouteDecision.blocked(RouteAction.ESCALATE, RejectCode.MAX_HOPS,
                                             why, env, self._ms(t0))  # 裁决=升级
            env.mark(DeliveryStatus.DROPPED, why)  # 其他配额闸超了：丢弃
            self._record(env, RejectCode.QUOTA_EXCEEDED, SpanStatus.TIMEOUT)  # 记 span：负载原因用 TIMEOUT
            return RouteDecision.blocked(RouteAction.DROP, RejectCode.QUOTA_EXCEEDED,
                                         why, env, self._ms(t0))  # 裁决=丢弃

        # ---- 第 4 关：防死循环（重复 / 乒乓）------------------------------
        loop = self._detect_loop(env)  # 查内容重复与乒乓；返回 None = 正常
        if loop is not None:  # 判定为无效来回
            env.mark(DeliveryStatus.ESCALATED, loop)  # 信封打"已升级"
            self._escalate(env, RejectCode.LOOP_DETECTED, loop)  # 交给主管收口
            return RouteDecision.blocked(RouteAction.ESCALATE, RejectCode.LOOP_DETECTED,
                                         loop, env, self._ms(t0))  # 裁决=升级

        # ---- 投递 --------------------------------------------------------
        env.mark(DeliveryStatus.DELIVERED)  # 六道工序全过：信封打"已投递"
        self.bus.post_envelope(env)  # 真正进总线：接收方现在能读到了
        self.quota.consume(sender, env.round)  # 配额记账：消息真发出去才算消费
        self._content_hits[(sender, env.receiver, content or "")] += 1  # 内容去重账 +1（下次重复能查到）
        self._pair_hits[(sender, env.receiver)] += 1  # 方向计数账 +1（下次乒乓能查到）
        self._record(env, RejectCode.OK, SpanStatus.OK)  # 记一条 OK span
        _logger.debug(f"消息已投递: {sender} → {env.receiver} [{kind.value}] hop={env.hop}")  # 调试日志：谁发给谁、什么类型、第几跳
        return RouteDecision.ok(env, "已投递", self._ms(t0))  # 裁决=投递，附耗时

    # ------------------------------------------------------------------
    # 防死循环
    # ------------------------------------------------------------------
    # 【什么】无效来回检测器（重复内容 + 乒乓两个视角）
    # 【作用】返回 None 表示正常；返回字符串表示判定原因（直接用作升级理由）
    # 【为什么】重复与乒乓都不是"错"，是"卡住了"——要升级收口而不是简单拦截
    def _detect_loop(self, env: Envelope) -> Optional[str]:  # 无效来回检测：None=正常，字符串=判定原因
        """判定这条消息是否属于无效来回。返回 None 表示正常。"""
        key = (env.sender, env.receiver, env.content)  # 三元组键：同一个人对同一个人说同一段话
        if self._content_hits[key] >= self.config.repeat_limit:  # 这段话已出现过 limit 次
            return (f"{env.sender} 重复发送相同内容给 {env.receiver} "
                    f"（第 {self._content_hits[key] + 1} 次），判定为无效重复")  # 说明第几次，供主管判断

        # 【乒乓怎么判】
        #   同一对收发方，正反两个方向都发过至少 pingpong_limit 次，
        #   说明他们在互相推而不是推进任务。这不是"错"，是"卡住了"，
        #   所以要升级给主管收口，而不是简单地不让发。
        a2b = self._pair_hits.get((env.sender, env.receiver), 0)  # 正向已发次数（没发过记 0）
        b2a = self._pair_hits.get((env.receiver, env.sender), 0)  # 反向已发次数（没发过记 0）
        if env.receiver != "*" and a2b >= self.config.pingpong_limit and b2a >= self.config.pingpong_limit:  # 广播不算乒乓；双向都到限才算
            return (f"{env.sender} 与 {env.receiver} 已来回 {a2b}/{b2a} 次，"
                    f"判定为踢皮球，升级给主管")  # 带双方次数，主管一眼看出卡了多久
        return None  # 两道检测都过：这条消息是正常交互

    # 【什么】升级执行器：留痕 + 改投 + 记账
    # 【作用】把无效交互转投给主管，消息不丢但换了收口人
    # 【为什么配额照记】升级也算发送者发出去了一条，不记账就能靠"被升级"绕过配额
    def _escalate(self, env: Envelope, code: RejectCode, why: str) -> None:  # 升级执行：留痕 + 改投主管 + 配额照记
        """把无效交互交给主管，而不是投给原接收方。"""
        target = self.config.escalate_to or self.supervisor or "supervisor"  # 升级目标三级兜底：配置→主管名→字面量
        item = {**envelope_to_log_dict(env), "code": code.value, "escalate_to": target}  # 留痕条目：信封摘要+原因码+投给谁
        self.escalations.append(item)  # 进升级列表：主管/人工事后可审
        env.receiver = target          # 改投主管：消息不丢，但换了个收口的人
        self.bus.post_envelope(env)  # 投递给主管：他能读到这条无效交互
        self.quota.consume(env.sender, env.round)  # 配额照记：升级也算发送者发出去了一条
        _logger.warning(f"消息升级给 {target}: {why}")  # 告警日志：升级是异常信号，用 warning 级

    # ------------------------------------------------------------------
    # 观测与工具
    # ------------------------------------------------------------------
    # 【什么】把一次收发写进观测链路
    # 【作用】每条消息一个 span：名字带 类型.发送者→接收者，树上一眼可读
    # 【为什么三个前置条件任一不满足就静默返回】观测是附加能力，不能影响主流程
    def _record(self, env: Envelope, code: RejectCode, status: SpanStatus) -> None:  # 写一条消息 span
        """把这次收发写进观测链路。"""
        if self.tracer is None or self.trace is None or not self.config.trace_enabled:  # 没 tracer/没绑 trace/关了开关：不记
            return
        span = self.tracer.start_span(  # 开 span：挂到当前运行的链路上
            self.trace,  # 本次运行的 trace
            name=f"msg.{env.kind.value}.{env.sender}→{env.receiver}",  # span 名：一眼看出谁在干什么
            kind="message",  # span 类别：监控按 message 类聚合
            parent_id=self.trace.root_id or "",  # 统一挂根下：消息之间是平级协作关系
        )
        span.inputs.update(envelope_to_log_dict(env, content_limit=120))  # 输入=信封摘要（正文截 120 字防膨胀）
        self.tracer.end_span(  # 立即结束：消息收发是瞬时动作，span 只记这一次裁决的结果
            span, status=status,  # OK / ERROR / TIMEOUT（限流配额用 TIMEOUT 表示负载原因）
            outputs={"delivery": env.delivery.value, "code": code.value,  # 输出=最终状态+原因码
                     "hop": env.hop, "round": env.round,  # 附跳数与轮次：排查死循环直接看这里
                     "size_bytes": env.size_bytes},  # 附正文大小：排查超长消息用
            error=env.reason if status == SpanStatus.ERROR else "",  # 只有 ERROR 才把理由填进 error 字段
        )

    # 【什么】按 id 回查消息
    # 【作用】给 send 算 hop / 挂 parent_id 用；空 id 或查不到都返回 None（按"无父"处理）
    # 【为什么只看最近 200 条】父消息不会太久远，全量扫太贵
    def _find(self, msg_id: str) -> Optional[Envelope]:  # 按 id 回查消息；查不到返回 None
        if not msg_id:  # 空 id 直接 None：主动发起的消息没有父
            return None
        for m in self.bus.all(limit=200):  # 只回看最近 200 条
            if m.id == msg_id:  # 命中
                return m  # 返回父消息信封
        return None  # 没找到：父消息可能已被清掉，按"无父"处理

    # 【什么】毫秒耗时工具
    # 【作用】给裁决算 elapsed_ms，监控看"编排器本身慢不慢"
    @staticmethod
    def _ms(t0: float) -> int:  # 从 t0 到现在过了几毫秒
        return int((time.time() - t0) * 1000)  # 秒差 × 1000 取整

    # 【什么】运行统计快照
    # 【作用】给监控大盘与排障用：轮次、已投递数、升级明细、配额与限流水位
    def stats(self) -> dict:  # 运行统计快照
        """编排器的运行统计：给监控大盘与排障用。"""
        return {
            "round": self.round_no,  # 当前轮次
            "delivered": len([1 for m in self.bus.all(limit=500)  # 已投递条数：扫最近 500 条数 DELIVERED
                              if m.delivery == DeliveryStatus.DELIVERED]),
            "escalated": len(self.escalations),  # 升级总数
            "escalations": self.escalations[-10:],  # 最近 10 条升级明细：大盘只放最新的
            "quota": self.quota.stats(),  # 配额账本快照
            "limiter": self.limiter.stats(),  # 限流水位快照
        }
