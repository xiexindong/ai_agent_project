"""
Human Gate（人工闸门）模块
==========================
链路代码唯一需要打交道的对象：一句话问"这步能不能过"。

【为什么链路里只该看到一个 ask】
    调用方不关心请求怎么存、人在哪个界面点、等多久超时、
    超时了该放行还是该拒绝 —— 这些都是 gate 内部的事。
    链路里只要：

        go, args = gate.ask_tool("data_query", {"id": "1"})
        if not go: 跳过这一步

    这样 HITL 可以整块开关（关掉时 ask 立即返回放行，零开销），
    也可以换成共享存储做多实例审批，链路一行不改。

【resolver 是什么，为什么要有它】
    等待真人是有阻塞代价的。三种场合不能真的等：
        1. 单元测试 / CI —— 没有人，必须自动答复
        2. 无人值守时段  —— 需要按预设策略自动放行或拒绝
        3. 演示脚本      —— 想演示"人改了一下参数"而不想真开个网页
    resolver 就是一个 (InterruptRequest) -> Optional[HumanReply] 的函数：
        返回回复  → 立即采用，不阻塞
        返回 None → 走正常等待（等人或超时）

【超时兜底的顺序不能反】
    先构造超时回复并提交（status: WAITING → RESOLVED），
    再把状态改成 EXPIRED。反过来的话 submit 会因"请求已结束"拒收，
    审计里就只剩"超时"、看不到"最终按什么动作执行了"。
"""
from __future__ import annotations  # 启用延迟注解求值，允许类型提示前向引用

import json  # 解析人工修改后的 JSON 参数（ask_tool 的 modify 分支用）
from typing import Callable, Optional  # 类型标注：可调用类型与可选类型

from modules.common.logger import get_logger  # 项目统一日志器工厂
from modules.hitl.models import (  # 导入 HITL 数据契约模型
    HumanAction,  # 人工决策动作枚举（approve / modify / reject / abort）
    HumanReply,  # 人工回复模型
    InterruptKind,  # 中断点类型枚举
    InterruptRequest,  # 中断请求模型
    InterruptStatus,  # 中断请求生命周期状态枚举
)
from modules.hitl.policy import HITLPolicy  # 触发策略：决定何时中断、超时按什么动作兜底
from modules.hitl.store import InMemoryInterruptStore, InterruptStore  # 存储协议与进程内默认实现

_logger = get_logger("hitl_gate")  # 本模块日志器

# resolver 的类型：拿到请求，返回回复或 None（None = 走正常等待）
Resolver = Callable[[InterruptRequest], Optional[HumanReply]]  # 自动答复器类型别名：请求进、回复或 None 出


# 【什么】固定动作自动答复器的闭包工厂
# 【作用】生成一个对任何请求都返回同一预设答复的 resolver
# 【为什么】单元测试 / CI / 无人值守场景没有真人，必须用确定性答复避免阻塞
def auto_resolver(
    action: HumanAction = HumanAction.APPROVE,  # 固定返回的决策动作，默认放行
    edited: str = "",  # action 为 modify 时携带的修改内容
    operator: str = "auto",  # 操作人标识，写入审计记录
    note: str = "自动答复（测试 / CI / 无人值守）",  # 答复备注，说明答复来源
) -> Resolver:  # 返回值：符合 Resolver 签名的闭包
    """生成一个固定动作的自动答复器。

    【输入 → 输出对照】
        输入: resolver = auto_resolver(HumanAction.REJECT)
              resolver(InterruptRequest(id="hitl_1", ...))
        输出: HumanReply(request_id="hitl_1", action=reject, operator="auto")
    """

    def _resolve(req: InterruptRequest) -> HumanReply:  # 内层闭包：忽略请求内容，给出预设答复
        return HumanReply(  # 构造人工回复对象
            request_id=req.id, action=action, edited=edited, operator=operator, note=note  # 关联请求 ID 并填入预设字段
        )

    return _resolve  # 把闭包作为答复器返回


# 【什么】按中断点类型编排答复的脚本化答复器
# 【作用】不同 kind 给不同动作，未编排的 kind 用 fallback
# 【为什么】演示"这里批准、那里拒绝、另一处改参数"时无需真人逐条点击
def scripted_resolver(script: dict[str, HumanAction], fallback: HumanAction = HumanAction.APPROVE) -> Resolver:  # script: kind→动作 的编排表；fallback: 未命中时的默认动作
    """按中断点类型编排答复：适合演示"这里批准、那里拒绝、另一处改参数"。

    【输入 → 输出对照】
        输入: script = {"before_tool": "reject", "before_final": "approve"}
             请求 kind=before_tool  → reject
             请求 kind=before_plan  → approve（fallback）
    """

    def _resolve(req: InterruptRequest) -> HumanReply:  # 内层闭包：按请求 kind 查脚本
        action = script.get(req.kind.value, fallback)  # 命中脚本用脚本动作，未命中用 fallback
        return HumanReply(  # 构造人工回复对象
            request_id=req.id, action=action, operator="script",  # 操作人标记为 script
            note=f"按脚本对 {req.kind.value} 答复 {action.value}",  # 备注记录命中的编排规则
        )

    return _resolve  # 把闭包作为答复器返回


# 【什么】人工闸门
# 【作用】发起中断 → 等人（或自动答复）→ 返回决策；链路代码唯一需要打交道的对象
# 【为什么】把策略判断、存储、通知、超时兜底全部封装在内部，链路只需一句 ask
class HumanGate:
    """人工闸门：发起中断 → 等人（或自动答复）→ 返回决策。"""

    def __init__(
        self,
        policy: Optional[HITLPolicy] = None,  # 触发策略；None 时用默认策略（总开关关闭）
        store: Optional[InterruptStore] = None,  # 存储实现；None 时用进程内存储
        resolver: Optional[Resolver] = None,  # 自动答复器；None 表示真人等待模式
        on_event: Optional[Callable[[InterruptRequest], None]] = None,  # 请求登记后的事件钩子（挂通知器）
    ):
        self.policy = policy or HITLPolicy()  # 策略缺省时给默认值（HITL 关闭）
        self.store: InterruptStore = store or InMemoryInterruptStore()  # 存储缺省时用内存实现
        self.resolver = resolver  # 保存自动答复器
        self.on_event = on_event  # 保存事件钩子

    # 【什么】闸门是否启用的只读属性
    # 【作用】对外暴露 enabled，调用方不用深入 policy 对象
    # 【为什么】enabled 的判定权在策略，属性透传避免两处真相
    @property  # 以只读属性形式暴露
    def enabled(self) -> bool:  # 返回闸门总开关状态
        return self.policy.enabled  # 以策略的 enabled 为准

    # ------------------------------------------------------------------
    # 【核心】发起一次中断并等到结果
    #   三条出口：
    #     ① 策略未启用     → 立即 approve（零开销，等价于没装 HITL）
    #     ② 有 resolver    → 用它的答复，不阻塞
    #     ③ 等人           → 等到回复；超时按 default_action 兜底
    # ------------------------------------------------------------------
    # 【什么】发起一次中断请求并等待最终决策
    # 【作用】gate 的唯一核心入口：登记 → 自动答复 / 等人 → 超时兜底
    # 【为什么】三条出口收拢在一处，链路对 HITL 的内部机制完全无感知
    def ask(
        self,
        *,
        kind: InterruptKind,  # 中断点类型（Agent 在哪个环节停下来）
        title: str,  # 一句话标题，直接显示在审批页
        reason: str = "",  # 触发原因：命中了哪条策略
        context: Optional[dict] = None,  # 给人看的上下文（工具名 / 参数 / 草稿等）
        options: Optional[list[str]] = None,  # 可选动作；None 表示四种全开放
        run_id: str = "",  # 所属运行 ID，便于与 RunResult 关联
        session_id: str = "",  # 会话 ID，前端按会话展示审批卡片
        agent: str = "",  # 抛出中断的 Agent 名
        timeout_seconds: Optional[float] = None,  # 单次等待上限；None 用策略默认值
    ) -> HumanReply:  # 返回最终决策回复
        # ① 策略没开：直接放行。这样 HITL 可以整块开关而不改动调用点。
        if not self.policy.kind_enabled(kind):  # 该中断点未启用
            return HumanReply(  # 构造放行回复，零开销返回
                request_id="", action=HumanAction.APPROVE, operator="policy",  # 无对应请求；动作=放行；来源=策略
                note=f"中断点 {kind.value} 未启用，自动放行",  # 备注说明放行原因
            )

        req = InterruptRequest(  # 构造中断请求对象
            run_id=run_id, session_id=session_id, agent=agent,  # 关联运行 / 会话 / Agent 标识
            kind=kind, title=title, reason=reason,  # 中断点 / 标题 / 原因
            context=dict(context or {}),  # 拷贝上下文，避免外部原字典被后续改动
            options=list(options or [a.value for a in HumanAction]),  # 缺省时四种动作全开放
            default_action=self.policy.default_action_for(kind),  # 按中断点取超时兜底动作
            timeout_seconds=float(timeout_seconds or self.policy.timeout_seconds),  # 缺省用策略等待上限
        )
        self.store.save(req)  # 登记请求（同时预热等待 Event）
        if self.on_event:  # 配了事件钩子（通常是通知器）
            try:
                self.on_event(req)  # 同步触发通知
            except Exception as e:  # 通知异常绝不能打断审批主链路
                _logger.debug(f"中断事件回调异常（忽略）: {e}")  # 只记 debug 日志

        # ② 自动答复器：无人值守 / 测试 / 演示
        if self.resolver is not None:  # 注入了自动答复器
            auto = self.resolver(req)  # 请求自动答复
            if auto is not None:  # 答复器给出了确定回复（None 表示仍走真人等待）
                self.store.submit(auto)  # 提交回复，走正常留痕
                return auto  # 立即返回，不阻塞

        # ③ 等人
        reply = self.store.wait(req.id, req.timeout_seconds)  # 阻塞等待真人回复，超时返回 None
        if reply is not None:  # 人在超时前回复了
            return reply  # 采用真人回复

        # 【超时兜底】顺序见文件头注释：先提交再置 EXPIRED
        timeout_reply = HumanReply(  # 构造超时兜底回复
            request_id=req.id, action=req.default_action, operator="system:timeout",  # 按请求自带的兜底动作执行
            note=f"超时 {req.timeout_seconds}s 无人处理，按默认动作 {req.default_action.value} 兜底",  # 备注记录超时事实
        )
        self.store.submit(timeout_reply)  # 先提交兜底回复（WAITING → RESOLVED），保证审计能看到最终动作
        req.status = InterruptStatus.EXPIRED  # 再把请求状态置为已超时
        _logger.warning(  # 记警告日志
            f"中断超时兜底: {req.id} kind={kind.value} 按 {req.default_action.value} 执行"  # 说明哪个请求按什么动作兜底
        )
        return timeout_reply  # 返回兜底回复

    # ------------------------------------------------------------------
    # 【工具调用前的批准】链路里最常用的一处
    #   返回 (是否放行, 最终参数)
    #   modify 时 edited 应是 JSON 字符串，解析成功就替换参数；
    #   解析失败保留原参数 —— 不能因为人写错格式就把整次调用搞挂。
    # ------------------------------------------------------------------
    # 【什么】工具调用前的人工批准入口
    # 【作用】返回 (是否放行, 最终参数)，modify 时解析人工改后的 JSON 参数
    # 【为什么】链路最常用的审批点；参数替换必须容错，格式错了不能把调用搞挂
    def ask_tool(
        self,
        tool_name: str,  # 待调用的工具名
        arguments: dict,  # 工具调用参数
        *,
        agent: str = "",  # 发起调用的 Agent 名
        run_id: str = "",  # 所属运行 ID
        session_id: str = "",  # 会话 ID
    ) -> tuple[bool, dict]:  # 返回 (是否放行, 最终使用的参数)
        reason = self.policy.check_tool(tool_name)  # 查工具是否命中敏感规则，命中返回原因
        if reason is None:  # 非敏感工具
            return True, dict(arguments or {})  # 直接放行，参数原样拷贝返回

        reply = self.ask(  # 发起 before_tool 中断
            kind=InterruptKind.BEFORE_TOOL,  # 中断点：调用工具前
            title=f"调用工具：{tool_name}",  # 审批页标题
            reason=reason,  # 命中的敏感规则说明
            context={"tool": tool_name, "arguments": dict(arguments or {})},  # 给人看的工具名与参数
            agent=agent, run_id=run_id, session_id=session_id,  # 透传来源标识
        )
        if not reply.is_go:  # reject / abort：不继续
            return False, dict(arguments or {})  # 不放行，参数原样返回（调用方应跳过这一步）
        if reply.action == HumanAction.MODIFY and reply.edited.strip():  # 人工修改且给了修改内容
            try:
                parsed = json.loads(reply.edited)  # 尝试把修改内容解析为 JSON
                if isinstance(parsed, dict):  # 解析结果是字典才能作为参数
                    _logger.info(f"人工修改工具参数: {tool_name} {arguments} → {parsed}")  # 记参数变更日志
                    return True, parsed  # 放行并使用修改后的参数
            except json.JSONDecodeError as e:  # 修改内容不是合法 JSON
                _logger.warning(f"人工修改内容不是合法 JSON，沿用原参数: {e}")  # 记警告并降级沿用原参数
        return True, dict(arguments or {})  # approve 或修改无效：放行且沿用原参数

    # ------------------------------------------------------------------
    # 【出最终答案前的确认】返回 (是否放行, 最终文本)
    # ------------------------------------------------------------------
    # 【什么】出最终答案前的人工确认入口
    # 【作用】返回 (是否放行, 最终文本)；modify 时采用人工改后的文本
    # 【为什么】最终答案是发给用户前的最后一道人工关卡
    def ask_final(
        self, draft: str, *, agent: str = "", run_id: str = "", session_id: str = ""  # draft: 待确认草稿；其余为来源标识
    ) -> tuple[bool, str]:  # 返回 (是否放行, 最终发出的文本)
        reply = self.ask(  # 发起 before_final 中断
            kind=InterruptKind.BEFORE_FINAL,  # 中断点：出最终答案前
            title="确认最终答案",  # 审批页标题
            reason="最终答案发出前的人工确认",  # 固定原因说明
            context={"draft": draft[:2000]},  # 草稿截断 2000 字，避免上下文过大
            agent=agent, run_id=run_id, session_id=session_id,  # 透传来源标识
        )
        if not reply.is_go:  # reject / abort：不继续
            return False, draft  # 不放行，返回原草稿（调用方应停止发送）
        return True, reply.content_or(draft)  # 放行；modify 用改后文本，否则用原草稿

    # ------------------------------------------------------------------
    # 【人这一侧的接口】提交答复 / 看待办 / 看归档
    # ------------------------------------------------------------------
    # 【什么】提交人工答复
    # 【作用】人这一侧（API / 审批页 / CLI）把决策写回闸门
    # 【为什么】人侧只认 request_id 和动作，拼装 HumanReply 的细节收敛在闸门内
    def submit(self, request_id: str, action: HumanAction, edited: str = "",  # 请求 ID / 决策动作 / 修改内容
               operator: str = "", note: str = "") -> bool:  # 操作人 / 备注；返回是否受理成功
        return self.store.submit(HumanReply(  # 构造回复并交给存储层
            request_id=request_id, action=action, edited=edited,  # 关联请求与决策
            operator=operator, note=note,  # 审计字段
        ))

    # 【什么】看待办清单
    # 【作用】审批页轮询当前还在等待的请求
    # 【为什么】人侧需要知道"现在有哪些事等我拍板"
    def pending(self, limit: int = 50) -> list[InterruptRequest]:  # limit: 最多返回条数
        return self.store.pending(limit)  # 透传给存储层

    # 【什么】看归档记录
    # 【作用】返回完整留痕（请求 + 回复 成对）
    # 【为什么】审计与排查需要"谁在什么时候、为什么被打断、人怎么回的"
    def records(self) -> list:  # 返回归档列表；存储不支持时给空列表
        return self.store.records() if hasattr(self.store, "records") else []  # 兼容没有 records 能力的存储实现

    # 【什么】作废所有还在等待的请求
    # 【作用】流程提前结束时唤醒所有等待线程
    # 【为什么】不唤醒的话等待线程要挂到超时才释放，白白占住并行度
    def cancel_waiting(self) -> int:  # 返回实际作废的请求数
        """流程提前结束时作废所有还在等的请求，避免等待线程挂住。

        【什么时候需要】运行超时 / abort / 主流程异常退出时，
        可能还有请求停在 WAITING。不唤醒的话，
        等待它们的线程要等到 timeout 才释放，团队并行度被白白占住。
        """
        n = 0  # 作废计数
        for req in self.pending(limit=200):  # 遍历当前待办（上限 200 防失控）
            cancel = getattr(self.store, "cancel", None)  # 存储可能没有 cancel 能力，动态取
            if cancel and cancel(req.id):  # 有能力且作废成功
                n += 1  # 计数 +1
        return n  # 返回作废总数
