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
from __future__ import annotations  # 启用延迟注解求值，允许类型提示前向引用

import json  # 序列化上下文与 webhook 消息体
import queue  # SSE 订阅者的消息队列
import sys  # 写 stderr 用
import threading  # 订阅者列表锁与 webhook 后台线程
import urllib.error  # 网络错误类型（标准库）
import urllib.request  # 发送 webhook POST（标准库）
from typing import Callable, Optional  # 类型标注：可调用与可选
from urllib.parse import urlparse  # 解析 webhook 域名以识别平台

from modules.common.logger import get_logger  # 项目统一日志器工厂
from modules.hitl.models import InterruptRequest  # 中断请求模型

_logger = get_logger("hitl_notify")  # 本模块日志器

# webhook 形态：auto=按域名猜，其余按平台拼消息体
WEBHOOK_KINDS = ("auto", "dingtalk", "feishu", "wecom", "generic")  # 支持的机器人平台标识

_DEFAULT_OPTIONS = "approve / modify / reject / abort"  # 请求未带可选动作时的默认展示文案


# ---------------------------------------------------------------------------
# 【什么】把一次中断请求格式化成人能读的一段话
# 【作用】终端、钉钉、飞书共用同一份文本，避免各渠道文案漂移
# 【为什么】文案只有一处源头，改一处全渠道生效
# ---------------------------------------------------------------------------
def format_interrupt_text(req: InterruptRequest, max_context_chars: int = 800) -> str:  # 输入请求与上下文截断长度，返回多行文本
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
    ctx = ""  # 上下文 JSON 文本，默认空
    if req.context:  # 有上下文才序列化
        try:
            ctx = json.dumps(req.context, ensure_ascii=False, default=str)  # 保留中文；不可序列化对象转 str
        except Exception:                                   # 上下文里可能有不可序列化的对象
            ctx = str(req.context)  # 兜底用 repr 形式
        limit = max(0, int(max_context_chars))  # 截断长度下限为 0
        if len(ctx) > limit:  # 超长则截断，避免通知消息过长
            ctx = ctx[:limit] + "...（已截断）"  # 截断并标注

    options = " / ".join(str(o) for o in req.options) if req.options else _DEFAULT_OPTIONS  # 可选动作拼成展示串
    parts = [f"agent={req.agent}" if req.agent else "",  # 来源字段：agent（可空）
             f"session={req.session_id}" if req.session_id else "",  # 来源字段：session（可空）
             f"run={req.run_id}" if req.run_id else ""]  # 来源字段：run（可空）
    source = " / ".join(p for p in parts if p)  # 过滤空段后拼成来源行

    lines = [f"[待人工拍板] {req.kind.value} {req.id}", f"标题：{req.title}"]  # 首两行：标识行 + 标题行
    if req.reason:  # 有原因才输出
        lines.append(f"原因：{req.reason}")  # 追加原因行
    if source:  # 有来源才输出
        lines.append(f"来源：{source}")  # 追加来源行
    if ctx:  # 有上下文才输出
        lines.append(f"上下文：{ctx}")  # 追加上下文行
    lines.append(f"可选：{options}")  # 追加可选动作行
    lines.append(f"{float(req.timeout_seconds):.0f}s 内无人处理 → 按 {req.default_action.value} 兜底")  # 追加超时兜底说明
    lines.append('处理：POST /hitl/reply  {"request_id": "' + req.id + '", "action": "approve"}')  # 追加处理方式提示
    return "\n".join(lines)  # 多行拼成最终文本


# ---------------------------------------------------------------------------
# 【什么】按 webhook 域名猜平台
# 【作用】配置里写 auto 就行，不用让人记住该填 dingtalk 还是 feishu
# 【为什么】平台域名是稳定特征，自动识别减少配置出错
# ---------------------------------------------------------------------------
def detect_webhook_kind(url: str) -> str:  # 输入 webhook URL，返回平台标识
    """按 URL 域名判断机器人类型，认不出就当通用 JSON。

    【输入 → 输出对照】
        "https://oapi.dingtalk.com/robot/send?access_token=x" → "dingtalk"
        "https://open.feishu.cn/open-apis/bot/v2/hook/x"      → "feishu"
        "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?x"  → "wecom"
        "https://internal.example.com/hitl"                   → "generic"
    """
    host = urlparse(url or "").netloc.lower()  # 取域名并转小写；空 URL 容错为空串
    if "dingtalk" in host:  # 钉钉机器人域名
        return "dingtalk"  # 判定为钉钉
    if "feishu" in host or "larksuite" in host:  # 飞书国内版 / 海外版域名
        return "feishu"  # 判定为飞书
    if "weixin.qq.com" in host:  # 企业微信机器人域名
        return "wecom"  # 判定为企业微信
    return "generic"  # 认不出就当通用 JSON 接口


# 【什么】中断通知器
# 【作用】把"有人在等你拍板"通过终端 / 机器人 / SSE 三条通道推出去
# 【为什么】请求只躺在内存里等人来取会安静超时，必须主动通知到人
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
        console: bool = True,  # 是否打印到终端（stderr）
        webhook: str = "",  # 机器人地址；留空=不推送
        webhook_kind: str = "auto",  # 平台形态：auto=按域名自动识别
        webhook_timeout: float = 3.0,  # 单次推送的 HTTP 超时（秒）
        max_context_chars: int = 800,  # 上下文展示的最大字符数
        sink: Optional[Callable[[str], None]] = None,  # 自定义输出函数，测试用来捕获终端内容
    ):
        self.console = bool(console)  # 归一化为布尔
        self.webhook = str(webhook or "").strip()  # 归一化并去空白
        self.webhook_kind = str(webhook_kind or "auto").strip().lower() or "auto"  # 归一化；空值回退 auto
        self.webhook_timeout = float(webhook_timeout or 3.0)  # 归一化；0 / None 回退 3 秒
        self.max_context_chars = int(max_context_chars or 800)  # 归一化；缺省 800 字
        self.sink = sink  # 保存自定义输出

        self._subscribers: list = []  # SSE 订阅者队列列表
        self._lock = threading.RLock()  # 保护订阅者列表的可重入锁

    # ------------------------------------------------------------------
    # 【核心入口】签名与 HumanGate.on_event 一致：拿一个 req，不返回
    #   【为什么不抛出】on_event 是在 gate.ask() 里同步调用的，
    #     通知器一抛异常就会打断"登记 → 等待"这条主链路，
    #     等于把"通知没发出去"升级成"这次审批整个失败"。
    # ------------------------------------------------------------------
    # 【什么】通知核心入口
    # 【作用】格式化文本后扇出到终端 / SSE / webhook 三条通道
    # 【为什么】签名与 HumanGate.on_event 一致，可直接挂为事件钩子
    def notify(self, req: InterruptRequest) -> None:  # 接收请求，无返回
        try:
            text = format_interrupt_text(req, self.max_context_chars)  # 渲染成人能读的文本
            self._to_console(text)  # 通道①：终端
            self._fanout(req)  # 通道②：SSE 订阅者
            self._post_async(req, text)  # 通道③：webhook（后台线程）
        except Exception as e:                              # 兜底：通知绝不能影响审批本身
            _logger.warning(f"HITL 通知异常（已忽略，Agent 仍在等待人工）: {e}")  # 只记警告

    # ------------------------------------------------------------------
    # ① console：写 stderr 而不是 stdout，免得污染 CLI 的正常输出
    # ------------------------------------------------------------------
    # 【什么】终端通道
    # 【作用】把通知文本写到 stderr
    # 【为什么】写 stderr 而不是 stdout，免得污染 CLI 的正常输出
    def _to_console(self, text: str) -> None:  # 输出一段通知文本
        if not self.console:  # 终端通道被关闭
            return  # 直接跳过
        try:
            if self.sink is not None:  # 测试注入了自定义输出
                self.sink(text)  # 走自定义输出
            else:
                print(text, file=sys.stderr, flush=True)  # 写 stderr 并立即刷出
        except Exception as e:  # 输出失败（如流被关闭）
            _logger.debug(f"终端通知输出失败（忽略）: {e}")  # 只记 debug

    # ------------------------------------------------------------------
    # ② SSE：推给所有订阅者（/hitl/stream）；队列满就丢，绝不阻塞 Agent
    # ------------------------------------------------------------------
    # 【什么】SSE 扇出通道
    # 【作用】把请求推给所有订阅者队列
    # 【为什么】审批页用长连接收推送省掉轮询；队列满就丢，绝不阻塞 Agent
    def _fanout(self, req: InterruptRequest) -> None:  # 扇出一个请求
        with self._lock:  # 加锁读订阅者列表
            if not self._subscribers:  # 没有订阅者
                return  # 直接返回
            targets = list(self._subscribers)  # 拷贝快照，避免迭代中被修改
        dead = []  # 失效订阅者收集
        for q in targets:  # 逐个推送
            try:
                q.put_nowait(req)  # 非阻塞入队；满了立即抛异常
            except Exception:
                dead.append(q)                              # 队列满或订阅者已失效
        if dead:  # 有失效订阅者
            with self._lock:  # 加锁清理
                for q in dead:  # 逐个摘除
                    if q in self._subscribers:  # 防重复移除
                        self._subscribers.remove(q)  # 从订阅列表移除

    # ------------------------------------------------------------------
    # ③ webhook：开后台线程发，不让网络耗时算进"等人的时间"
    #   【为什么必须异步】Agent 线程随后要 event.wait(60s)；
    #     若同步 POST 且网络慢到 3s 超时，人可用的审批时间就被白白吃掉 3s。
    # ------------------------------------------------------------------
    # 【什么】webhook 异步发送入口
    # 【作用】开后台线程发 HTTP，不把网络耗时算进等人的时间
    # 【为什么】同步 POST 慢到会吃掉人的审批时间窗口
    def _post_async(self, req: InterruptRequest, text: str) -> None:  # 异步推送
        if not self.webhook:  # 未配置 webhook
            return  # 跳过
        threading.Thread(  # 创建后台守护线程
            target=self._post_sync, args=(req, text),  # 在线程里跑同步发送
            name=f"hitl-notify-{req.id}", daemon=True,  # 命名便于排查；daemon 不挡进程退出
        ).start()  # 立即启动

    # 【什么】webhook 同步发送实现
    # 【作用】拼消息体 → 序列化 → POST → 记录结果
    # 【为什么】失败只记警告：通知是增强不是依赖
    def _post_sync(self, req: InterruptRequest, text: str) -> None:  # 在后台线程中执行
        try:
            payload = self.payload_for(self.webhook_kind, req, text)  # 按平台拼消息体
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")  # 序列化为 UTF-8 JSON 字节
            request = urllib.request.Request(  # 构造 HTTP 请求
                self.webhook, data=data,  # POST 到机器人地址
                headers={"Content-Type": "application/json"}, method="POST",  # JSON 头部与 POST 方法
            )
            with urllib.request.urlopen(request, timeout=self.webhook_timeout) as resp:  # 发送并等待响应
                _logger.info(f"HITL 通知已推送: {req.id} → HTTP {resp.status}")  # 记录推送成功
        except Exception as e:                              # 含 URLError / 超时 / 非 2xx
            _logger.warning(  # 失败只记警告
                f"HITL 通知推送失败（Agent 仍在等待人工，不影响流程）: {req.id} | {e}"  # 说明不影响主流程
            )

    # ------------------------------------------------------------------
    # 【什么】按平台拼消息体
    # 【作用】各机器人只认自己的字段；generic 给自建系统用，信息最全
    # 【为什么】消息格式收敛在一个方法里，加新平台只改这一处
    # ------------------------------------------------------------------
    def payload_for(self, kind: str, req: InterruptRequest, text: str) -> dict:  # 返回该平台认得的 JSON body
        """拼出该平台认得的 JSON body。

        【输入 → 输出对照】
            kind="dingtalk" → {"msgtype": "text", "text": {"content": "..."}}
            kind="feishu"   → {"msg_type": "text", "content": {"text": "..."}}
            kind="wecom"    → {"msgtype": "text", "text": {"content": "..."}}
            kind="generic"  → {"event": "hitl.pending", "request": {...}, "text": "..."}
        """
        k = (kind or "auto").strip().lower()  # 归一化平台标识
        if k == "auto":  # 自动识别模式
            k = detect_webhook_kind(self.webhook)  # 按域名猜平台
        if k == "dingtalk":  # 钉钉机器人
            return {"msgtype": "text", "text": {"content": text}}  # 钉钉文本消息体
        if k == "feishu":  # 飞书机器人
            return {"msg_type": "text", "content": {"text": text}}  # 飞书文本消息体
        if k == "wecom":  # 企业微信机器人
            return {"msgtype": "text", "text": {"content": text}}  # 企业微信文本消息体（与钉钉同构）
        return {  # generic：自建系统，信息最全
            "event": "hitl.pending",  # 事件类型
            "text": text,  # 人读文本
            "request": {  # 完整请求字段
                "id": req.id, "kind": req.kind.value, "title": req.title,  # 标识 / 类型 / 标题
                "reason": req.reason, "agent": req.agent,  # 原因 / 来源 Agent
                "session_id": req.session_id, "run_id": req.run_id,  # 会话 / 运行标识
                "context": req.context, "options": req.options,  # 上下文 / 可选动作
                "default_action": req.default_action.value,  # 超时兜底动作
                "timeout_seconds": req.timeout_seconds,  # 等待上限
                "created_at": req.created_at,  # 创建时间戳
            },
        }

    # ------------------------------------------------------------------
    # 【SSE 订阅】/hitl/stream 用；unsubscribe 必须配对调用，否则队列泄漏
    # ------------------------------------------------------------------
    # 【什么】登记 SSE 订阅者
    # 【作用】返回订阅者专属的消息队列
    # 【为什么】每个长连接一个队列互不干扰；maxsize 防内存被慢消费者撑爆
    def subscribe(self, maxsize: int = 200) -> queue.Queue:  # maxsize: 队列容量上限
        """登记一个订阅者，返回它专属的队列。"""
        q: queue.Queue = queue.Queue(maxsize=max(1, int(maxsize)))  # 建容量至少为 1 的有界队列
        with self._lock:  # 加锁
            self._subscribers.append(q)  # 登记到订阅列表
        return q  # 返回队列给订阅方

    # 【什么】注销 SSE 订阅者
    # 【作用】从订阅列表移除队列
    # 【为什么】必须与 subscribe 配对调用，否则队列泄漏
    def unsubscribe(self, q: queue.Queue) -> None:  # 注销指定队列
        with self._lock:  # 加锁
            if q in self._subscribers:  # 确认还在列表里
                self._subscribers.remove(q)  # 移除

    # 【什么】当前订阅者数量
    # 【作用】监控 / 调试时看有多少长连接挂着
    # 【为什么】读列表长度也要加锁，避免读到中间态
    @property  # 只读属性
    def subscriber_count(self) -> int:  # 返回订阅者数量
        with self._lock:  # 加锁
            return len(self._subscribers)  # 返回列表长度


# ---------------------------------------------------------------------------
# 【什么】按配置造通知器
# 【作用】调用方不用自己读 settings；字段缺失一律退回默认（向后兼容）
# 【为什么】配置读取细节收敛一处，新增配置项不影响调用方
# ---------------------------------------------------------------------------
def build_notifier(settings=None) -> InterruptNotifier:  # settings 缺省时从全局配置读
    """按配置构造通知器。

    【输入 → 输出对照】
        HITL_NOTIFY_WEBHOOK 留空           → 只走终端 + SSE
        HITL_NOTIFY_WEBHOOK=钉钉地址        → 终端 + SSE + 钉钉（kind=auto 自动识别）
        HITL_NOTIFY_CONSOLE=false          → 静默，只走 SSE
    """
    if settings is None:  # 调用方没传配置对象
        from config import get_settings  # 延迟导入，避免模块级循环依赖
        settings = get_settings()  # 读全局配置

    def _get(name: str, default):  # 内层工具函数：安全取配置字段
        return getattr(settings, name, default)  # 字段缺失一律退回默认（向后兼容）

    return InterruptNotifier(  # 按配置构造通知器
        console=bool(_get("hitl_notify_console", True)),  # 终端通道开关，默认开
        webhook=str(_get("hitl_notify_webhook", "") or "").strip(),  # 机器人地址，默认空
        webhook_kind=str(_get("hitl_notify_webhook_kind", "auto") or "auto").strip().lower(),  # 平台形态，默认 auto
        webhook_timeout=float(_get("hitl_notify_webhook_timeout", 3.0) or 3.0),  # 推送超时，默认 3 秒
    )


# ---------------------------------------------------------------------------
# 【什么】进程级共享通知器
# 【作用】和 gate 同理：链路抛中断与 /hitl/stream 订阅必须是同一个对象，
#         否则订阅者挂在一个通知器上，收到的是另一个通知器的事件。
# 【为什么】进程内单例保证"谁抛出"与"谁订阅"对上同一个事件源
# ---------------------------------------------------------------------------
_GLOBAL_NOTIFIER: Optional[InterruptNotifier] = None  # 进程级共享通知器实例，首次使用时建造


# 【什么】取进程级共享通知器
# 【作用】首次调用按配置建造，之后返回同一实例
# 【为什么】抛中断与 SSE 订阅必须挂在同一个通知器上，否则订阅者收不到事件
def get_global_notifier() -> InterruptNotifier:  # 返回共享实例
    """取进程级共享通知器（第一次调用时按配置建造）。"""
    global _GLOBAL_NOTIFIER  # 声明读写模块级变量
    if _GLOBAL_NOTIFIER is None:  # 尚未建造
        _GLOBAL_NOTIFIER = build_notifier()  # 按配置建造
    return _GLOBAL_NOTIFIER  # 返回共享实例


# 【什么】替换进程级共享通知器
# 【作用】测试注入 sink / 演示时换实现
# 【为什么】不替换的话测试捕获不到终端输出
def set_global_notifier(notifier: Optional[InterruptNotifier]) -> None:  # 传 None 可清空
    """替换共享通知器（测试注入 sink / 演示时用）。"""
    global _GLOBAL_NOTIFIER  # 声明写模块级变量
    _GLOBAL_NOTIFIER = notifier  # 替换共享实例
