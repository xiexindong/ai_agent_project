"""
HITL 通知器：把「有人在等你拍板」这件事主动推出去
==================================================

【为什么需要它】
    gate.ask() 只干三件事：登记请求 → 等人 → 超时兜底。
    store.save(req) 之后，请求只是躺在内存里等人来取：
    人这一侧如果没人盯着 GET /hitl/pending，这条请求就会安静地等到超时，
    然后按 default_action 兜底 —— 对工具调用来说等于直接拒绝。
    表现出来就是"系统自己把事拒了，人从头到尾不知情"。

    本模块补上"请求抛出的那一刻就通知到人"这一环，
    挂在 HumanGate.on_event 上（gate.py:149），零侵入。

【三条通道，为什么是三条】
    console  终端打印   —— 本地 / CLI / CI 跑脚本时立刻看得见（默认开）
    webhook  机器人     —— 钉钉 / 飞书 / 企业微信 / 自定义 URL，真正通知到人（配 URL 才开）
    SSE      订阅队列   —— 审批页用 GET /hitl/stream 长连接收推送，省掉轮询

【通知失败会怎样】
    什么都不发生。通知是增强不是依赖：
    webhook 超时 / 报错 / 没配，都只记 warning，
    Agent 继续按原来的节奏等人、超时、兜底。

【最小用法】
    notifier = build_notifier()                  # 按配置造
    gate = build_gate(enabled=True, on_event=notifier.notify)

    # 只想在终端看到（默认就是这样）
    notifier = InterruptNotifier(console=True)

    # 推钉钉
    notifier = InterruptNotifier(webhook="https://oapi.dingtalk.com/robot/send?access_token=xxx")
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import urllib.error
import urllib.request
from typing import Callable, Optional
from urllib.parse import urlparse

from modules.common.logger import get_logger
from modules.hitl.models import InterruptRequest

_logger = get_logger("hitl_notify")

# webhook 形态：auto=按域名猜，其余按平台拼消息体
WEBHOOK_KINDS = ("auto", "dingtalk", "feishu", "wecom", "generic")

_DEFAULT_OPTIONS = "approve / modify / reject / abort"


# ---------------------------------------------------------------------------
# 【什么】把一次中断请求格式化成人能读的一段话
# 【作用】终端、钉钉、飞书共用同一份文本，避免各渠道文案漂移
# ---------------------------------------------------------------------------
def format_interrupt_text(req: InterruptRequest, max_context_chars: int = 800) -> str:
    """把 InterruptRequest 渲染成给人看的文本。

    【输入 → 输出对照】
        输入: InterruptRequest(kind=BEFORE_TOOL, title="调用工具：send_notice",
                              reason="工具 send_notice 在人工批准白名单内",
                              context={"tool": "send_notice", "arguments": {"to": "U007"}},
                              default_action=reject, timeout_seconds=60)
        输出:
            [待人工拍板] before_tool hitl_3f9a2c11
            标题：调用工具：send_notice
            原因：工具 send_notice 在人工批准白名单内
            上下文：{"tool": "send_notice", "arguments": {"to": "U007"}}
            可选：approve / modify / reject / abort
            60s 内无人处理 → 按 reject 兜底
            处理：POST /hitl/reply  {"request_id": "hitl_3f9a2c11", "action": "approve"}
    """
    ctx = ""
    if req.context:
        try:
            ctx = json.dumps(req.context, ensure_ascii=False, default=str)
        except Exception:                                   # 上下文里可能有不可序列化的对象
            ctx = str(req.context)
        limit = max(0, int(max_context_chars))
        if len(ctx) > limit:
            ctx = ctx[:limit] + "...（已截断）"

    options = " / ".join(str(o) for o in req.options) if req.options else _DEFAULT_OPTIONS
    parts = [f"agent={req.agent}" if req.agent else "",
             f"session={req.session_id}" if req.session_id else "",
             f"run={req.run_id}" if req.run_id else ""]
    source = " / ".join(p for p in parts if p)

    lines = [f"[待人工拍板] {req.kind.value} {req.id}", f"标题：{req.title}"]
    if req.reason:
        lines.append(f"原因：{req.reason}")
    if source:
        lines.append(f"来源：{source}")
    if ctx:
        lines.append(f"上下文：{ctx}")
    lines.append(f"可选：{options}")
    lines.append(f"{float(req.timeout_seconds):.0f}s 内无人处理 → 按 {req.default_action.value} 兜底")
    lines.append('处理：POST /hitl/reply  {"request_id": "' + req.id + '", "action": "approve"}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 【什么】按 webhook 域名猜平台
# 【作用】配置里写 auto 就行，不用让人记住该填 dingtalk 还是 feishu
# ---------------------------------------------------------------------------
def detect_webhook_kind(url: str) -> str:
    """按 URL 域名判断机器人类型，认不出就当通用 JSON。

    【输入 → 输出对照】
        "https://oapi.dingtalk.com/robot/send?access_token=x" → "dingtalk"
        "https://open.feishu.cn/open-apis/bot/v2/hook/x"      → "feishu"
        "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?x"  → "wecom"
        "https://internal.example.com/hitl"                   → "generic"
    """
    host = urlparse(url or "").netloc.lower()
    if "dingtalk" in host:
        return "dingtalk"
    if "feishu" in host or "larksuite" in host:
        return "feishu"
    if "weixin.qq.com" in host:
        return "wecom"
    return "generic"


class InterruptNotifier:
    """把中断请求推给人：终端 / 机器人 / SSE 订阅者。

    Attributes:
        console:         是否打印到终端（stderr）
        webhook:         机器人地址；留空=不推送
        webhook_kind:    auto / dingtalk / feishu / wecom / generic
        webhook_timeout: 单次推送的 HTTP 超时（秒）
        sink:            自定义输出函数，测试用来捕获终端内容
    """

    def __init__(
        self,
        console: bool = True,
        webhook: str = "",
        webhook_kind: str = "auto",
        webhook_timeout: float = 3.0,
        max_context_chars: int = 800,
        sink: Optional[Callable[[str], None]] = None,
    ):
        self.console = bool(console)
        self.webhook = str(webhook or "").strip()
        self.webhook_kind = str(webhook_kind or "auto").strip().lower() or "auto"
        self.webhook_timeout = float(webhook_timeout or 3.0)
        self.max_context_chars = int(max_context_chars or 800)
        self.sink = sink

        self._subscribers: list = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 【核心入口】签名与 HumanGate.on_event 一致：拿一个 req，不返回
    #   【为什么不抛出】on_event 是在 gate.ask() 里同步调用的，
    #     通知器一抛异常就会打断"登记 → 等待"这条主链路，
    #     等于把"通知没发出去"升级成"这次审批整个失败"。
    # ------------------------------------------------------------------
    def notify(self, req: InterruptRequest) -> None:
        try:
            text = format_interrupt_text(req, self.max_context_chars)
            self._to_console(text)
            self._fanout(req)
            self._post_async(req, text)
        except Exception as e:                              # 兜底：通知绝不能影响审批本身
            _logger.warning(f"HITL 通知异常（已忽略，Agent 仍在等待人工）: {e}")

    # ------------------------------------------------------------------
    # ① console：写 stderr 而不是 stdout，免得污染 CLI 的正常输出
    # ------------------------------------------------------------------
    def _to_console(self, text: str) -> None:
        if not self.console:
            return
        try:
            if self.sink is not None:
                self.sink(text)
            else:
                print(text, file=sys.stderr, flush=True)
        except Exception as e:
            _logger.debug(f"终端通知输出失败（忽略）: {e}")

    # ------------------------------------------------------------------
    # ② SSE：推给所有订阅者（/hitl/stream）；队列满就丢，绝不阻塞 Agent
    # ------------------------------------------------------------------
    def _fanout(self, req: InterruptRequest) -> None:
        with self._lock:
            if not self._subscribers:
                return
            targets = list(self._subscribers)
        dead = []
        for q in targets:
            try:
                q.put_nowait(req)
            except Exception:
                dead.append(q)                              # 队列满或订阅者已失效
        if dead:
            with self._lock:
                for q in dead:
                    if q in self._subscribers:
                        self._subscribers.remove(q)

    # ------------------------------------------------------------------
    # ③ webhook：开后台线程发，不让网络耗时算进"等人的时间"
    #   【为什么必须异步】Agent 线程随后要 event.wait(60s)；
    #     若同步 POST 且网络慢到 3s 超时，人可用的审批时间就被白白吃掉 3s。
    # ------------------------------------------------------------------
    def _post_async(self, req: InterruptRequest, text: str) -> None:
        if not self.webhook:
            return
        threading.Thread(
            target=self._post_sync, args=(req, text),
            name=f"hitl-notify-{req.id}", daemon=True,
        ).start()

    def _post_sync(self, req: InterruptRequest, text: str) -> None:
        try:
            payload = self.payload_for(self.webhook_kind, req, text)
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                self.webhook, data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.webhook_timeout) as resp:
                _logger.info(f"HITL 通知已推送: {req.id} → HTTP {resp.status}")
        except Exception as e:                              # 含 URLError / 超时 / 非 2xx
            _logger.warning(
                f"HITL 通知推送失败（Agent 仍在等待人工，不影响流程）: {req.id} | {e}"
            )

    # ------------------------------------------------------------------
    # 【什么】按平台拼消息体
    # 【作用】各机器人只认自己的字段；generic 给自建系统用，信息最全
    # ------------------------------------------------------------------
    def payload_for(self, kind: str, req: InterruptRequest, text: str) -> dict:
        """拼出该平台认得的 JSON body。

        【输入 → 输出对照】
            kind="dingtalk" → {"msgtype": "text", "text": {"content": "..."}}
            kind="feishu"   → {"msg_type": "text", "content": {"text": "..."}}
            kind="wecom"    → {"msgtype": "text", "text": {"content": "..."}}
            kind="generic"  → {"event": "hitl.pending", "request": {...}, "text": "..."}
        """
        k = (kind or "auto").strip().lower()
        if k == "auto":
            k = detect_webhook_kind(self.webhook)
        if k == "dingtalk":
            return {"msgtype": "text", "text": {"content": text}}
        if k == "feishu":
            return {"msg_type": "text", "content": {"text": text}}
        if k == "wecom":
            return {"msgtype": "text", "text": {"content": text}}
        return {
            "event": "hitl.pending",
            "text": text,
            "request": {
                "id": req.id, "kind": req.kind.value, "title": req.title,
                "reason": req.reason, "agent": req.agent,
                "session_id": req.session_id, "run_id": req.run_id,
                "context": req.context, "options": req.options,
                "default_action": req.default_action.value,
                "timeout_seconds": req.timeout_seconds,
                "created_at": req.created_at,
            },
        }

    # ------------------------------------------------------------------
    # 【SSE 订阅】/hitl/stream 用；unsubscribe 必须配对调用，否则队列泄漏
    # ------------------------------------------------------------------
    def subscribe(self, maxsize: int = 200) -> queue.Queue:
        """登记一个订阅者，返回它专属的队列。"""
        q: queue.Queue = queue.Queue(maxsize=max(1, int(maxsize)))
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


# ---------------------------------------------------------------------------
# 【什么】按配置造通知器
# 【作用】调用方不用自己读 settings；字段缺失一律退回默认（向后兼容）
# ---------------------------------------------------------------------------
def build_notifier(settings=None) -> InterruptNotifier:
    """按配置构造通知器。

    【输入 → 输出对照】
        HITL_NOTIFY_WEBHOOK 留空           → 只走终端 + SSE
        HITL_NOTIFY_WEBHOOK=钉钉地址        → 终端 + SSE + 钉钉（kind=auto 自动识别）
        HITL_NOTIFY_CONSOLE=false          → 静默，只走 SSE
    """
    if settings is None:
        from config import get_settings
        settings = get_settings()

    def _get(name: str, default):
        return getattr(settings, name, default)

    return InterruptNotifier(
        console=bool(_get("hitl_notify_console", True)),
        webhook=str(_get("hitl_notify_webhook", "") or "").strip(),
        webhook_kind=str(_get("hitl_notify_webhook_kind", "auto") or "auto").strip().lower(),
        webhook_timeout=float(_get("hitl_notify_webhook_timeout", 3.0) or 3.0),
    )


# ---------------------------------------------------------------------------
# 【什么】进程级共享通知器
# 【作用】和 gate 同理：链路抛中断与 /hitl/stream 订阅必须是同一个对象，
#         否则订阅者挂在一个通知器上，收到的是另一个通知器的事件。
# ---------------------------------------------------------------------------
_GLOBAL_NOTIFIER: Optional[InterruptNotifier] = None


def get_global_notifier() -> InterruptNotifier:
    """取进程级共享通知器（第一次调用时按配置建造）。"""
    global _GLOBAL_NOTIFIER
    if _GLOBAL_NOTIFIER is None:
        _GLOBAL_NOTIFIER = build_notifier()
    return _GLOBAL_NOTIFIER


def set_global_notifier(notifier: Optional[InterruptNotifier]) -> None:
    """替换共享通知器（测试注入 sink / 演示时用）。"""
    global _GLOBAL_NOTIFIER
    _GLOBAL_NOTIFIER = notifier
