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
from __future__ import annotations

import json
from typing import Callable, Optional

from modules.common.logger import get_logger
from modules.hitl.models import (
    HumanAction,
    HumanReply,
    InterruptKind,
    InterruptRequest,
    InterruptStatus,
)
from modules.hitl.policy import HITLPolicy
from modules.hitl.store import InMemoryInterruptStore, InterruptStore

_logger = get_logger("hitl_gate")

# resolver 的类型：拿到请求，返回回复或 None（None = 走正常等待）
Resolver = Callable[[InterruptRequest], Optional[HumanReply]]


def auto_resolver(
    action: HumanAction = HumanAction.APPROVE,
    edited: str = "",
    operator: str = "auto",
    note: str = "自动答复（测试 / CI / 无人值守）",
) -> Resolver:
    """生成一个固定动作的自动答复器。

    【输入 → 输出对照】
        输入: resolver = auto_resolver(HumanAction.REJECT)
              resolver(InterruptRequest(id="hitl_1", ...))
        输出: HumanReply(request_id="hitl_1", action=reject, operator="auto")
    """

    def _resolve(req: InterruptRequest) -> HumanReply:
        return HumanReply(
            request_id=req.id, action=action, edited=edited, operator=operator, note=note
        )

    return _resolve


def scripted_resolver(script: dict[str, HumanAction], fallback: HumanAction = HumanAction.APPROVE) -> Resolver:
    """按中断点类型编排答复：适合演示"这里批准、那里拒绝、另一处改参数"。

    【输入 → 输出对照】
        输入: script = {"before_tool": "reject", "before_final": "approve"}
             请求 kind=before_tool  → reject
             请求 kind=before_plan  → approve（fallback）
    """

    def _resolve(req: InterruptRequest) -> HumanReply:
        action = script.get(req.kind.value, fallback)
        return HumanReply(
            request_id=req.id, action=action, operator="script",
            note=f"按脚本对 {req.kind.value} 答复 {action.value}",
        )

    return _resolve


class HumanGate:
    """人工闸门：发起中断 → 等人（或自动答复）→ 返回决策。"""

    def __init__(
        self,
        policy: Optional[HITLPolicy] = None,
        store: Optional[InterruptStore] = None,
        resolver: Optional[Resolver] = None,
        on_event: Optional[Callable[[InterruptRequest], None]] = None,
    ):
        self.policy = policy or HITLPolicy()
        self.store: InterruptStore = store or InMemoryInterruptStore()
        self.resolver = resolver
        self.on_event = on_event

    @property
    def enabled(self) -> bool:
        return self.policy.enabled

    # ------------------------------------------------------------------
    # 【核心】发起一次中断并等到结果
    #   三条出口：
    #     ① 策略未启用     → 立即 approve（零开销，等价于没装 HITL）
    #     ② 有 resolver    → 用它的答复，不阻塞
    #     ③ 等人           → 等到回复；超时按 default_action 兜底
    # ------------------------------------------------------------------
    def ask(
        self,
        *,
        kind: InterruptKind,
        title: str,
        reason: str = "",
        context: Optional[dict] = None,
        options: Optional[list[str]] = None,
        run_id: str = "",
        session_id: str = "",
        agent: str = "",
        timeout_seconds: Optional[float] = None,
    ) -> HumanReply:
        # ① 策略没开：直接放行。这样 HITL 可以整块开关而不改动调用点。
        if not self.policy.kind_enabled(kind):
            return HumanReply(
                request_id="", action=HumanAction.APPROVE, operator="policy",
                note=f"中断点 {kind.value} 未启用，自动放行",
            )

        req = InterruptRequest(
            run_id=run_id, session_id=session_id, agent=agent,
            kind=kind, title=title, reason=reason,
            context=dict(context or {}),
            options=list(options or [a.value for a in HumanAction]),
            default_action=self.policy.default_action_for(kind),
            timeout_seconds=float(timeout_seconds or self.policy.timeout_seconds),
        )
        self.store.save(req)
        if self.on_event:
            try:
                self.on_event(req)
            except Exception as e:
                _logger.debug(f"中断事件回调异常（忽略）: {e}")

        # ② 自动答复器：无人值守 / 测试 / 演示
        if self.resolver is not None:
            auto = self.resolver(req)
            if auto is not None:
                self.store.submit(auto)
                return auto

        # ③ 等人
        reply = self.store.wait(req.id, req.timeout_seconds)
        if reply is not None:
            return reply

        # 【超时兜底】顺序见文件头注释：先提交再置 EXPIRED
        timeout_reply = HumanReply(
            request_id=req.id, action=req.default_action, operator="system:timeout",
            note=f"超时 {req.timeout_seconds}s 无人处理，按默认动作 {req.default_action.value} 兜底",
        )
        self.store.submit(timeout_reply)
        req.status = InterruptStatus.EXPIRED
        _logger.warning(
            f"中断超时兜底: {req.id} kind={kind.value} 按 {req.default_action.value} 执行"
        )
        return timeout_reply

    # ------------------------------------------------------------------
    # 【工具调用前的批准】链路里最常用的一处
    #   返回 (是否放行, 最终参数)
    #   modify 时 edited 应是 JSON 字符串，解析成功就替换参数；
    #   解析失败保留原参数 —— 不能因为人写错格式就把整次调用搞挂。
    # ------------------------------------------------------------------
    def ask_tool(
        self,
        tool_name: str,
        arguments: dict,
        *,
        agent: str = "",
        run_id: str = "",
        session_id: str = "",
    ) -> tuple[bool, dict]:
        reason = self.policy.check_tool(tool_name)
        if reason is None:
            return True, dict(arguments or {})

        reply = self.ask(
            kind=InterruptKind.BEFORE_TOOL,
            title=f"调用工具：{tool_name}",
            reason=reason,
            context={"tool": tool_name, "arguments": dict(arguments or {})},
            agent=agent, run_id=run_id, session_id=session_id,
        )
        if not reply.is_go:
            return False, dict(arguments or {})
        if reply.action == HumanAction.MODIFY and reply.edited.strip():
            try:
                parsed = json.loads(reply.edited)
                if isinstance(parsed, dict):
                    _logger.info(f"人工修改工具参数: {tool_name} {arguments} → {parsed}")
                    return True, parsed
            except json.JSONDecodeError as e:
                _logger.warning(f"人工修改内容不是合法 JSON，沿用原参数: {e}")
        return True, dict(arguments or {})

    # ------------------------------------------------------------------
    # 【出最终答案前的确认】返回 (是否放行, 最终文本)
    # ------------------------------------------------------------------
    def ask_final(
        self, draft: str, *, agent: str = "", run_id: str = "", session_id: str = ""
    ) -> tuple[bool, str]:
        reply = self.ask(
            kind=InterruptKind.BEFORE_FINAL,
            title="确认最终答案",
            reason="最终答案发出前的人工确认",
            context={"draft": draft[:2000]},
            agent=agent, run_id=run_id, session_id=session_id,
        )
        if not reply.is_go:
            return False, draft
        return True, reply.content_or(draft)

    # ------------------------------------------------------------------
    # 【人这一侧的接口】提交答复 / 看待办 / 看归档
    # ------------------------------------------------------------------
    def submit(self, request_id: str, action: HumanAction, edited: str = "",
               operator: str = "", note: str = "") -> bool:
        return self.store.submit(HumanReply(
            request_id=request_id, action=action, edited=edited,
            operator=operator, note=note,
        ))

    def pending(self, limit: int = 50) -> list[InterruptRequest]:
        return self.store.pending(limit)

    def records(self) -> list:
        return self.store.records() if hasattr(self.store, "records") else []

    def cancel_waiting(self) -> int:
        """流程提前结束时作废所有还在等的请求，避免等待线程挂住。

        【什么时候需要】运行超时 / abort / 主流程异常退出时，
        可能还有请求停在 WAITING。不唤醒的话，
        等待它们的线程要等到 timeout 才释放，团队并行度被白白占住。
        """
        n = 0
        for req in self.pending(limit=200):
            cancel = getattr(self.store, "cancel", None)
            if cancel and cancel(req.id):
                n += 1
        return n
