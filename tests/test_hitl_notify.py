"""
HITL 通知器单元测试
====================
全部离线：不连网络、不连数据库、不需要真人。

【为什么这些用例只验"通知侧"，不验等待】
    等待与超时是 test_hitl.py 的事。这里只验一件事：
    请求登记后，通知器有没有把它变成人能看见的东西（文本 / 消息体 / SSE 队列）。
"""
import os

os.environ.setdefault("LOG_LEVEL", "ERROR")

import pytest  # noqa: E402

from modules.hitl import (  # noqa: E402
    InterruptNotifier,
    auto_resolver,
    build_gate,
    format_interrupt_text,
)
from modules.hitl.models import HumanAction, InterruptKind, InterruptRequest  # noqa: E402
from modules.hitl.notify import detect_webhook_kind  # noqa: E402


def _req(**kw) -> InterruptRequest:
    base = dict(kind=InterruptKind.BEFORE_TOOL, title="调用工具：send_notice")
    base.update(kw)
    return InterruptRequest(**base)


def test_文本包含人做决定需要的全部要素():
    """标题/原因/上下文/可选动作/超时兜底/处理入口，缺一样人都没法拍板。"""
    text = format_interrupt_text(_req(
        reason="工具 send_notice 在人工批准白名单内",
        context={"tool": "send_notice", "arguments": {"to": "U007"}},
        default_action=HumanAction.REJECT, timeout_seconds=60,
    ))
    assert "调用工具：send_notice" in text
    assert "send_notice 在人工批准白名单内" in text
    assert '"to": "U007"' in text
    assert "approve / modify / reject / abort" in text
    assert "按 reject 兜底" in text
    assert "/hitl/reply" in text


def test_上下文过长会截断():
    """超长上下文不能把终端/机器人消息刷爆。"""
    text = format_interrupt_text(_req(context={"blob": "x" * 5000}), max_context_chars=100)
    assert "已截断" in text
    assert "x" * 5000 not in text


def test_终端通道输出到_sink_且不回退到_stdout():
    """sink 用于捕获；console=False 时一条都不出。"""
    got = []
    InterruptNotifier(console=True, sink=got.append).notify(_req())
    assert len(got) == 1 and "待人工拍板" in got[0]

    silent = []
    InterruptNotifier(console=False, sink=silent.append).notify(_req())
    assert silent == []


def test_webhook_按平台拼消息体():
    """钉钉/飞书/企业微信只认自己的字段；generic 信息最全。"""
    n = InterruptNotifier(console=False, webhook="https://oapi.dingtalk.com/robot/send")
    r, text = _req(), "T"
    assert n.payload_for("dingtalk", r, text) == {"msgtype": "text", "text": {"content": "T"}}
    assert n.payload_for("feishu", r, text) == {"msg_type": "text", "content": {"text": "T"}}
    assert n.payload_for("wecom", r, text) == {"msgtype": "text", "text": {"content": "T"}}

    generic = n.payload_for("generic", r, text)
    assert generic["event"] == "hitl.pending"
    assert generic["request"]["kind"] == "before_tool"
    assert generic["request"]["title"] == "调用工具：send_notice"


def test_auto_按域名识别平台():
    """配置里写 auto 就该认出来，认不出退 generic 而不是报错。"""
    assert detect_webhook_kind("https://oapi.dingtalk.com/robot/send?access_token=x") == "dingtalk"
    assert detect_webhook_kind("https://open.feishu.cn/open-apis/bot/v2/hook/x") == "feishu"
    assert detect_webhook_kind("https://qyapi.weixin.qq.com/cgi-bin/webhook/send") == "wecom"
    assert detect_webhook_kind("https://internal.example.com/hitl") == "generic"

    n = InterruptNotifier(console=False, webhook="https://open.feishu.cn/x", webhook_kind="auto")
    assert "msg_type" in n.payload_for("auto", _req(), "T")


def test_SSE_订阅者能收到_退订后不再收():
    """连上就有、断开就摘——队列不泄漏。"""
    n = InterruptNotifier(console=False)
    q = n.subscribe()
    assert n.subscriber_count == 1

    n.notify(_req())
    assert q.get_nowait().title == "调用工具：send_notice"

    n.unsubscribe(q)
    assert n.subscriber_count == 0
    n.notify(_req())
    assert q.empty()


def test_通知失败不影响审批流程():
    """webhook 是个不可达地址时，notify 不抛出，请求照常登记照常等人。"""
    n = InterruptNotifier(console=False, webhook="http://127.0.0.1:1/nope", webhook_timeout=0.2)
    gate = build_gate(enabled=True, sensitive_tools=["send_notice"],
                      timeout_seconds=0.3, on_event=n.notify)
    # 没有人答复 → 超时按 reject 兜底；关键是这一路没有异常冒出来
    go, _args = gate.ask_tool("send_notice", {"to": "U007"})
    assert go is False
    assert len(gate.records()) == 1


def test_闸门登记后确实触发了通知():
    """端到端：ask 一次 → 通知器收到一次（这是整条链补全的验收点）。

    【为什么注入 auto_resolver】不注入就会真等 60 秒；
    通知发生在 resolver 之前，所以注入自动答复不影响这里要验的东西。
    """
    seen = []
    n = InterruptNotifier(console=True, sink=seen.append)
    gate = build_gate(enabled=True, sensitive_tools=["data_query"], on_event=n.notify,
                      resolver=auto_resolver(HumanAction.APPROVE))
    go, args = gate.ask_tool("data_query", {"id": "1"})

    assert go is True and args == {"id": "1"}
    assert len(seen) == 1 and "data_query" in seen[0]
