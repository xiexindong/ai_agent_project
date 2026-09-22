"""
熔断 + 降级测试：错误分类 / 三态流转 / 端到端接线
=================================================
全部用假 SDK，不联网、不读真实密钥 —— 测的是「我们自己写的编排逻辑」。

【为什么不用真 LLM 造故障】真环境里「主模型持续故障」要等超时、还不可控；
    假 SDK 能把「第几次失败、以什么状态码失败」精确编排，用例才稳定可重复。

【每条用例在证明什么】
    错误分类   —— 该降的降、不该降的不降（400 降级是白等一次超时）
    三态流转   —— 连续失败 → 打开 → 冷却 → 半开探针 → 恢复 / 重新打开
    接线正确   —— 熔断打开时主模型「一次都没被打」，这才是省下超时的证据
    不误伤     —— 业务错误既不降级也不熔断，一次参数 bug 不能熔断整条链路
"""
import time

import pytest

from config import get_settings
from modules.agent.llm import LLMClient, _is_degradable
from modules.common.circuit_breaker import (
    CLOSED, HALF_OPEN, OPEN, CircuitBreaker,
)

PRIMARY = "primary-model"
BACKUP = "backup-model"


# ------------------------------------------------------------------
# 假 SDK：记录调用顺序，按模型名决定这次成功还是抛什么异常
# ------------------------------------------------------------------
class _FakeError(Exception):
    """.status_code 在 = 模拟 HTTP 错误；不在 = 模拟网络层错误（超时/断连）。"""

    def __init__(self, status=None, msg="假故障"):
        super().__init__(msg)
        if status is not None:
            self.status_code = status


class _FakeMessage:
    content = "3*7=21"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResp:
    # 【为什么把 _Msg/_Choice 写在模块级而不是嵌套】Python 的类体作用域不参与闭包，
    #   嵌套类里引用外层类名会 NameError，只能在模块级平铺。
    choices = [_FakeChoice()]
    usage = None          # 置空 → _log_usage 直接返回，测试不碰数据库

    def __init__(self, model):
        self.model = model


class _FakeCompletions:
    def __init__(self, behavior):
        self.behavior = behavior          # {模型名: 异常实例 | None}
        self.calls = []                   # 调用轨迹，断言「谁被打了几次」

    def create(self, **kwargs):
        model = kwargs.get("model")
        self.calls.append(model)
        act = self.behavior.get(model)
        if isinstance(act, BaseException):
            raise act
        return _FakeResp(model)


class _FakeClient:
    def __init__(self, behavior):
        self.completions = _FakeCompletions(behavior)
        self.chat = type("_Chat", (), {"completions": self.completions})()


@pytest.fixture
def llm(monkeypatch):
    """造一个绕开单例与网络的 LLMClient：可注入假 SDK、阈值、冷却期、备用模型名。"""
    def _build(behavior, threshold=3, recovery=0.2, fallback=BACKUP):
        monkeypatch.setenv("FALLBACK_MODEL", fallback)
        monkeypatch.setenv("CB_FAILURE_THRESHOLD", str(threshold))
        monkeypatch.setenv("CB_RECOVERY_SECONDS", str(recovery))
        get_settings.cache_clear()        # get_settings 带 lru_cache，必须清掉才读到新值

        # 走 object.__new__ 绕开单例：__new__ 会复用旧实例、__init__ 会去建真客户端
        obj = object.__new__(LLMClient)
        obj._client = _FakeClient(behavior)
        obj._model = PRIMARY
        obj.usage_context = {}
        obj._breaker = CircuitBreaker(threshold, recovery)
        return obj

    yield _build
    get_settings.cache_clear()


# ------------------------------------------------------------------
# 【错误分类】降级只对「换个下游可能就好」的故障有意义
# ------------------------------------------------------------------
@pytest.mark.parametrize("exc,expected", [
    (_FakeError(), True),        # 无状态码 = 网络层：超时 / 连接失败
    (_FakeError(429), True),     # 限流：等一下下游就好了
    (_FakeError(500), True),     # 服务端故障
    (_FakeError(503), True),
    (_FakeError(400), False),    # 参数错：换谁都一样
    (_FakeError(401), False),    # 认证失败
    (_FakeError(402), False),    # 欠费
    (_FakeError(403), False),
    (_FakeError(404), False),    # 模型不存在
    (_FakeError(422), False),
])
def test_is_degradable_classification(exc, expected):
    assert _is_degradable(exc) is expected


# ------------------------------------------------------------------
# 【端到端：不该降的不降】400 直接抛，备用模型一次都不打
# ------------------------------------------------------------------
def test_bad_request_raises_without_fallback(llm):
    client = llm({PRIMARY: _FakeError(400)})
    with pytest.raises(_FakeError):
        client._chat(messages=[{"role": "user", "content": "hi"}])
    # 处理前：主模型抛 400 → 处理后：直接抛给上层，没有第二次请求
    assert client._client.completions.calls == [PRIMARY]


def test_no_fallback_configured_raises(llm):
    """没配备用模型时维持原行为：抛出去交给上层 / tenacity，但失败照样记账。"""
    client = llm({PRIMARY: _FakeError(503)}, fallback="")
    with pytest.raises(_FakeError):
        client._chat(messages=[])
    assert client._client.completions.calls == [PRIMARY]


# ------------------------------------------------------------------
# 【端到端：该降的降】503 → 换备用模型，返回的是备用模型的结果
# ------------------------------------------------------------------
def test_server_error_falls_back(llm):
    client = llm({PRIMARY: _FakeError(503)})
    resp = client._chat(messages=[])
    assert resp.model == BACKUP                                   # 答案来自备用模型
    assert client._client.completions.calls == [PRIMARY, BACKUP]  # 先主后备，各一次
    assert client._breaker.state(PRIMARY) == CLOSED               # 1 次 < 阈值 3，还没熔断


# ------------------------------------------------------------------
# 【熔断核心】连续失败到阈值后，主模型不再被打 —— 这才是省下超时的证据
# ------------------------------------------------------------------
def test_circuit_opens_then_skips_primary(llm):
    client = llm({PRIMARY: _FakeError(503)}, threshold=3)

    for _ in range(3):
        assert client._chat(messages=[]).model == BACKUP
    assert client._breaker.state(PRIMARY) == OPEN

    before = list(client._client.completions.calls)
    assert client._chat(messages=[]).model == BACKUP
    # 关键断言：熔断期间新增的调用只有备用模型，主模型一次都没再被打
    assert client._client.completions.calls == before + [BACKUP]


def test_half_open_probe_success_recovers(llm):
    client = llm({PRIMARY: _FakeError(503)}, threshold=2, recovery=0.05)
    for _ in range(2):
        client._chat(messages=[])
    assert client._breaker.state(PRIMARY) == OPEN

    time.sleep(0.06)
    assert client._breaker.state(PRIMARY) == HALF_OPEN

    client._client.completions.behavior[PRIMARY] = None      # 主模型恢复
    assert client._chat(messages=[]).model == PRIMARY        # 探针成功 → 流量回主模型
    assert client._breaker.state(PRIMARY) == CLOSED


def test_half_open_probe_failure_reopens(llm):
    """冷却期满不等于下游已好：探针失败要立刻重新打开，而不是放开全部流量。"""
    client = llm({PRIMARY: _FakeError(503)}, threshold=2, recovery=0.05)
    for _ in range(2):
        client._chat(messages=[])
    time.sleep(0.06)
    assert client._breaker.state(PRIMARY) == HALF_OPEN

    assert client._chat(messages=[]).model == BACKUP         # 探针失败 → 仍走备用
    assert client._breaker.state(PRIMARY) == OPEN            # 且重新进入冷却


# ------------------------------------------------------------------
# 【不误伤】业务错误既不降级也不熔断
# ------------------------------------------------------------------
def test_business_error_does_not_open_circuit(llm):
    client = llm({PRIMARY: _FakeError(400)}, threshold=2)
    for _ in range(5):
        with pytest.raises(_FakeError):
            client._chat(messages=[])
    assert client._breaker.state(PRIMARY) == CLOSED
    assert client._breaker.snapshot() == {}                   # 一次失败都没记
    assert client._client.completions.calls == [PRIMARY] * 5  # 主模型一直正常放行


# ------------------------------------------------------------------
# 【熔断器本身】按 key 隔离 + 半开只放一个探针
# ------------------------------------------------------------------
def test_half_open_lets_exactly_one_probe_through():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
    cb.record_failure("m")
    assert cb.state("m") == OPEN
    assert cb.allow("m") is False        # 冷却未到：快速失败

    time.sleep(0.06)
    assert cb.allow("m") is True         # 放行唯一探针
    assert cb.allow("m") is False        # 探针在途，其余请求仍被拦（防惊群）

    cb.record_success("m")
    assert cb.state("m") == CLOSED
    assert cb.allow("m") is True


def test_breaker_state_isolated_by_key():
    """一个下游挂了不该连累另一个 —— 隔离失效就是故障放大。"""
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure("a")
    assert cb.state("a") == OPEN
    assert cb.state("b") == CLOSED
    assert cb.allow("b") is True
    assert cb.snapshot() == {"a": {"state": OPEN, "failures": 1}}


def test_reset_clears_state():
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure("a")
    cb.reset("a")
    assert cb.state("a") == CLOSED
    cb.record_failure("a")
    cb.reset()
    assert cb.snapshot() == {}
