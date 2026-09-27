"""
权限与安全校验（编排器统一执行）
================================
所有消息在投递前都要过这一层：身份、权限、敏感词、参数。

【为什么校验必须放在编排器而不是发送方】
    放在发送方等于"让每个 Agent 自觉遵守规则"。只要有一个 Agent 的提示词
    被诱导成"直接把这段话发给 X"，规则就失效了。放在编排器是"必经之路"，
    Agent 无论说什么，能不能送出去由编排器说了算。

【三类校验各自拦什么】
    权限  —— 这个角色有没有资格发这种消息（例如只有主管能转派任务）
    敏感  —— 正文里有没有不该外传的内容（密钥、身份证号、内部代号）
    参数  —— 字段是否合法、正文是否超长、附加数据是否过大
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from modules.common.logger import get_logger
from modules.orchestration.protocol import Envelope, RejectCode

_logger = get_logger("guard")

# ---------------------------------------------------------------------------
# 【默认权限矩阵】哪种消息允许什么角色发
#   "*" 表示任意角色。矩阵的意义在于：
#   delegate（转派）是"指挥权"，只能主管有；
#   objection（异议）是"质疑权"，任何成员都有——否则错误结论没人敢指出；
#   finding / clarify 是信息共享，全员可发。
# ---------------------------------------------------------------------------
DEFAULT_KIND_PERMISSIONS: dict[str, list[str]] = {
    "delegate": ["supervisor"],
    "objection": ["supervisor", "reviewer", "worker", "general"],
    "finding": ["*"],
    "clarify": ["*"],
}


class GuardPolicy(BaseModel):
    """校验规则。全部有默认值，不配置也能跑。"""

    max_content_chars: int = Field(default=4000, ge=1, description="单条消息正文上限（字符）")
    max_meta_keys: int = Field(default=20, ge=0, description="附加字段个数上限")
    known_agents: list[str] = Field(default_factory=list, description="花名册；空=不校验接收者")
    allowed_senders: list[str] = Field(default_factory=list, description="可发送者白名单；空=不限制")
    kind_permissions: dict[str, list[str]] = Field(
        default_factory=lambda: dict(DEFAULT_KIND_PERMISSIONS),
        description="消息类型 → 允许的角色列表",
    )
    sensitive_words: list[str] = Field(default_factory=list, description="敏感词表")
    on_sensitive: str = Field(default="redact", description="命中敏感词后的动作：redact|reject")
    redact_mask: str = Field(default="***", description="打码替换成的文本")

    def allows(self, kind: str, role: str) -> bool:
        allowed = self.kind_permissions.get(kind)
        if allowed is None:          # 矩阵里没写的类型：默认放行，避免新增类型就全被拦
            return True
        return "*" in allowed or (role or "").lower() in [a.lower() for a in allowed]

    def scan_sensitive(self, text: str) -> list[str]:
        hits = []
        low = text or ""
        for w in self.sensitive_words:
            if w and w in low:
                hits.append(w)
        return hits


class GuardVerdict(BaseModel):
    """一次校验的结论。"""

    passed: bool = Field(description="是否放行")
    code: RejectCode = Field(default=RejectCode.OK)
    reason: str = Field(default="")
    # 【为什么返回改写后的信封而不是返回 bool】
    #   敏感词打码会改正文，调用方必须拿到改后的对象再投递；
    #   只回 bool 的话，投递的还是原文，打码等于没做。
    envelope: Optional[Envelope] = None
    redacted: list[str] = Field(default_factory=list, description="被打码的敏感词")


class PolicyGuard:
    """按策略校验一条信封。"""

    def __init__(self, policy: Optional[GuardPolicy] = None):
        self.policy = policy or GuardPolicy()

    def set_roster(self, names: list[str]) -> None:
        """更新花名册（harness 注册/注销 Agent 时同步调用）。

        【为什么花名册要在 guard 里留一份】
            校验接收者是否存在，需要一份名单；每次校验都去问 harness
            会让 guard 依赖运行时。留一份快照，由 harness 在人员变动时同步。
        """
        self.policy.known_agents = list(names)

    def check(self, env: Envelope) -> GuardVerdict:
        """按「协议 → 身份 → 权限 → 敏感 → 参数」的顺序校验。

        【为什么是这个顺序】
            先查最便宜也最基础的（字段缺失），再查需要查表的（身份/权限），
            最后才做全文扫描（敏感词，最贵）。反过来写会让一条空消息
            也去做一遍敏感词全文匹配，白白烧 CPU。
        """
        p = self.policy

        # ---- 1. 协议字段 ------------------------------------------------
        if not env.sender or not env.receiver:
            return GuardVerdict(passed=False, code=RejectCode.BAD_PROTOCOL,
                                reason="sender 与 receiver 不能为空", envelope=env)
        if env.sender == env.receiver:
            return GuardVerdict(passed=False, code=RejectCode.BAD_PROTOCOL,
                                reason="不能发给自己", envelope=env)
        if env.size_bytes > p.max_content_chars:
            return GuardVerdict(
                passed=False, code=RejectCode.TOO_LARGE,
                reason=f"正文 {env.size_bytes} 字超过上限 {p.max_content_chars}",
                envelope=env)

        # ---- 2. 身份 ----------------------------------------------------
        if p.allowed_senders and env.sender not in p.allowed_senders:
            return GuardVerdict(passed=False, code=RejectCode.SENDER_UNKNOWN,
                                reason=f"发送者 {env.sender} 不在白名单", envelope=env)
        # 【广播不校验接收者】"*" 是发给所有人的，逐个校验没有意义
        if env.receiver != "*" and p.known_agents and env.receiver not in p.known_agents:
            return GuardVerdict(passed=False, code=RejectCode.RECEIVER_UNKNOWN,
                                reason=f"接收者 {env.receiver} 不在花名册", envelope=env)

        # ---- 3. 权限矩阵 ------------------------------------------------
        if not p.allows(env.kind.value, env.sender_role):
            return GuardVerdict(
                passed=False, code=RejectCode.KIND_FORBIDDEN,
                reason=f"角色 {env.sender_role or '未标注'} 无权发送 {env.kind.value} 类型消息",
                envelope=env)

        # ---- 4. 敏感词 --------------------------------------------------
        hits = p.scan_sensitive(env.content)
        if hits:
            if p.on_sensitive == "reject":
                return GuardVerdict(
                    passed=False, code=RejectCode.SENSITIVE,
                    reason=f"命中敏感词，已拦截: {hits[:5]}",
                    envelope=env, redacted=[])
            # 【为什么默认打码而不是拒绝】
            #   直接拒绝会让正常协作被一个词整条打断（成员不知道哪里违规，
            #   只能重发，往往还是同一个词）。打码既保住内容可用，
            #   又确保敏感片段不外传；需要硬拦截的场景改成 reject。
            text = env.content
            for w in hits:
                text = text.replace(w, p.redact_mask)
            env.content = text
            env.meta["redacted"] = hits[:10]
            _logger.warning(f"消息已打码后放行: sender={env.sender} 命中 {len(hits)} 个词")
            return GuardVerdict(passed=True, code=RejectCode.OK,
                                reason=f"已打码 {len(hits)} 处", envelope=env, redacted=hits)

        # ---- 5. 参数体积 ------------------------------------------------
        if len(env.meta) > p.max_meta_keys:
            return GuardVerdict(passed=False, code=RejectCode.TOO_LARGE,
                                reason=f"附加字段 {len(env.meta)} 个超过上限 {p.max_meta_keys}",
                                envelope=env)

        return GuardVerdict(passed=True, code=RejectCode.OK, reason="校验通过", envelope=env)
