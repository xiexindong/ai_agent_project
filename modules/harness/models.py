"""
工具 Harness 的数据契约
========================
把「工具能干什么、谁批准、怎么隔离、花多少预算」全部变成可声明、可校验的数据。

【为什么要这一层】
    模型只能给出「我想调 tool X，参数是 Y」的意图，它不该自己决定
    「这个操作危不危险、我有没有权限、会不会把外部系统写坏」。
    这些判断必须由 Harness 依据**声明好的能力**来做：
        工具侧声明  →  risk / side_effect / sandbox / permissions / data_scopes
        策略侧裁决  →  allow（直接跑）/ approve（人工批准）/ deny（禁止执行）

【一句话分工】
    models.py      本文件：能力、策略、裁决、请求等数据结构（不含执行逻辑）
    budget.py      预算：调用次数 / token / 墙钟超时
    middleware.py  调用前中间件：参数 / 权限 / 业务范围 / 限流 / 预算
    sandbox.py     沙箱：none / subprocess / docker
    result_filter.py  结果过滤：裁剪 + 脱敏
    harness.py     总装：把上面几件串成 before_call → execute → after_call
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

# 【三个裁决动作】
ACTION_ALLOW = "allow"      # 直接执行
ACTION_APPROVE = "approve"  # 需人工批准（模型只能发起申请）
ACTION_DENY = "deny"        # 禁止执行（配置层面封死）


# 【什么】风险等级：工具副作用的分级标尺
# 【作用】策略按等级决定「直接跑 / 要审批 / 禁止跑」，而不是逐个工具写 if
class RiskLevel(str, Enum):
    """风险等级：从只读到不可逆。等级越高，越不允许模型自行决定。"""

    LOW = "low"            # 只读、无副作用，如 calculator / data_query
    MEDIUM = "medium"      # 有副作用但可回滚，如写草稿、发通知
    HIGH = "high"          # 影响外部系统，如删数据、改配置、下单
    CRITICAL = "critical"  # 不可逆或影响生产，如删库、改生产配置

    # 【什么】等级转序号
    # 【作用】让策略能写「risk >= HIGH 都要审批」这类比较
    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]

    # 【什么】宽松解析：字符串/None/非法值都落到 LOW
    # 【作用】配置里写错等级不会让启动崩掉，退化成最低风险（但不会悄悄放宽高风险工具）
    @classmethod
    def parse(cls, value: Any) -> "RiskLevel":
        if isinstance(value, RiskLevel):
            return value
        text = str(value or "").strip().lower()
        for level in cls:
            if level.value == text:
                return level
        return cls.LOW


# 【什么】单个工具的能力声明
# 【作用】工具注册时填一次，之后所有校验都读它，不再散落在 if 里
@dataclass
class ToolCapability:
    """工具能力声明：风险、副作用、隔离方式、所需权限与业务范围。

    【字段来源优先级】
        工具自己声明（BaseTool.capability） → 配置覆盖（TOOL_RISK_OVERRIDES）
        → 策略兜底（TOOL_DEFAULT_RISK）
    """

    name: str                                    # 工具名
    risk: RiskLevel = RiskLevel.LOW              # 风险等级
    side_effect: bool = False                    # True = 会写外部系统（删/改/发）
    sandbox: str = "none"                        # 隔离方式：none / subprocess / docker
    requires_approval: bool = False              # 显式要求人工批准（不论等级）
    permissions: Tuple[str, ...] = ()            # 需要的权限码，如 ("data:orders:all",)
    data_scopes: Tuple[str, ...] = ()            # 涉及的业务数据类型，如 ("orders",)
    max_result_chars: int = 0                    # 结果裁剪上限；0 = 用全局默认
    timeout_seconds: float = 0.0                 # 单次执行超时；0 = 用全局默认
    description: str = ""                        # 能力说明（审计与 /tools 展示用）

    # 【什么】是否需要人工批准
    # 【作用】高风险或显式声明的工具，模型拿不到"直接执行"的权利
    @property
    def needs_approval(self) -> bool:
        return self.requires_approval or self.risk.rank >= RiskLevel.HIGH.rank

    # 【什么】转成可序列化字典
    # 【作用】/tools 接口、审计 detail、日志展示都用它
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "risk": self.risk.value,
            "side_effect": self.side_effect,
            "sandbox": self.sandbox,
            "requires_approval": self.requires_approval,
            "needs_approval": self.needs_approval,
            "permissions": list(self.permissions),
            "data_scopes": list(self.data_scopes),
            "max_result_chars": self.max_result_chars,
            "timeout_seconds": self.timeout_seconds,
            "description": self.description,
        }


# 【什么】工具裁决结果
# 【作用】策略对「能不能调」给出的唯一答案，带原因（原因会回喂给模型）
@dataclass
class ToolDecision:
    action: str = ACTION_ALLOW     # allow / approve / deny
    reason: str = ""
    risk: str = RiskLevel.LOW.value

    @property
    def denied(self) -> bool:
        return self.action == ACTION_DENY


# 【什么】工具安全策略
# 【作用】把「哪些等级/哪些工具禁止执行、哪些需要审批」从代码里挪到配置
@dataclass
class ToolPolicy:
    """工具安全策略：按风险等级与工具名裁决 allow / approve / deny。

    【裁决顺序】先查名字黑名单/白名单，再查等级，最后默认 allow。
        ——名字优先于等级：可以单独把某个 medium 工具提级为需审批，
          也可以把某个 high 工具加入禁止名单（等级规则会被它盖住）。

    【输入输出对照】
        deny_levels={"critical"}, approve_levels={"high"}, deny_tools={"drop_*"}
          calculator(low)      → allow
          send_notice(high)    → approve（人工批准）
          drop_table(critical) → deny（禁止执行）
          drop_index(medium)   → deny（名字命中 drop_*，盖过等级）
    """

    default_risk: RiskLevel = RiskLevel.LOW                 # 工具未声明时的兜底等级
    overrides: Dict[str, str] = field(default_factory=dict)  # 工具名/前缀 → 风险等级
    deny_levels: Tuple[str, ...] = ("critical",)             # 这些等级一律禁止执行
    approve_levels: Tuple[str, ...] = ("high", "critical")   # 这些等级需人工批准
    deny_tools: Tuple[str, ...] = ()                         # 名字黑名单，支持前缀 *
    approve_tools: Tuple[str, ...] = ()                      # 名字白名单，支持前缀 *
    on_no_gate: str = "deny"        # 需要审批但没有人工闸门时：deny（安全优先）/ allow

    # 【什么】把逗号分隔的配置串切成元组
    # 【作用】配置项写 "drop_*,mysql_exec" 也能直接用
    @staticmethod
    def _split(text: Any) -> Tuple[str, ...]:
        if not text:
            return ()
        if isinstance(text, (list, tuple, set)):
            return tuple(str(x).strip() for x in text if str(x).strip())
        return tuple(x.strip() for x in str(text).split(",") if x.strip())

    # 【什么】按配置构造策略
    # 【作用】config.py 里那组 TOOL_* 配置一次转成一个策略对象
    @classmethod
    def from_settings(cls, s: Any) -> "ToolPolicy":
        overrides: Dict[str, str] = {}
        for item in cls._split(getattr(s, "tool_risk_overrides", "")):
            if ":" in item:                      # 形如 web_search:medium
                name, level = item.split(":", 1)
                overrides[name.strip()] = level.strip()
        return cls(
            default_risk=RiskLevel.parse(getattr(s, "tool_default_risk", "low")),
            overrides=overrides,
            deny_levels=cls._split(getattr(s, "tool_deny_levels", "critical")),
            approve_levels=cls._split(getattr(s, "tool_approve_levels", "high,critical")),
            deny_tools=cls._split(getattr(s, "tool_deny_tools", "")),
            approve_tools=cls._split(getattr(s, "tool_approve_tools", "")),
            on_no_gate=str(getattr(s, "tool_on_no_gate", "deny") or "deny"),
        )

    # 【什么】名字是否命中带 * 的规则
    # 【作用】"drop_*" 能命中 drop_table / drop_index
    @staticmethod
    def _match(name: str, rules: Tuple[str, ...]) -> bool:
        for rule in rules:
            if rule.endswith("*"):
                if name.startswith(rule[:-1]):
                    return True
            elif rule == name:
                return True
        return False

    # 【什么】把工具自己声明的能力与配置覆盖合并
    # 【作用】覆盖只改风险等级，其余字段仍以工具声明为准
    def effective(self, cap: ToolCapability) -> ToolCapability:
        risk = cap.risk
        if cap.risk.rank <= RiskLevel.LOW.rank and self.default_risk.rank > risk.rank:
            risk = self.default_risk                      # 工具没声明 → 用默认等级
        for pattern, level in self.overrides.items():
            # 【作用】按名字查配置有没有给这个工具单独定级（如 web_search:medium）；命中即覆盖工具声明并停止后续比对
            if self._match(cap.name, (pattern,)):
                # 【作用】把配置写的等级字符串（"medium"）转成枚举，替换掉工具自己声明的等级
                risk = RiskLevel.parse(level)             # 配置覆盖优先于工具声明
                break
        if risk is cap.risk:
            return cap
        return ToolCapability(
            name=cap.name, risk=risk, side_effect=cap.side_effect, sandbox=cap.sandbox,
            requires_approval=cap.requires_approval, permissions=cap.permissions,
            data_scopes=cap.data_scopes, max_result_chars=cap.max_result_chars,
            timeout_seconds=cap.timeout_seconds, description=cap.description,
        )

    # 【什么】对一次调用做裁决
    # 【作用】唯一的"能不能调"判定入口；返回的 reason 会原样回喂给模型
    def decide(self, cap: ToolCapability) -> ToolDecision:
        name = cap.name
        if self._match(name, self.deny_tools):
            return ToolDecision(ACTION_DENY, f"工具 {name} 已配置为禁止执行", cap.risk.value)
        if str(cap.risk.value) in self.deny_levels:
            return ToolDecision(ACTION_DENY, f"工具 {name} 风险等级为 {cap.risk.value}，已配置为禁止执行", cap.risk.value)
        if self._match(name, self.approve_tools):
            return ToolDecision(ACTION_APPROVE, f"工具 {name} 已配置为需人工批准", cap.risk.value)
        if str(cap.risk.value) in self.approve_levels or cap.requires_approval:
            return ToolDecision(ACTION_APPROVE, f"工具 {name} 风险等级为 {cap.risk.value}，需人工批准", cap.risk.value)
        return ToolDecision(ACTION_ALLOW, "", cap.risk.value)


# 【什么】一次工具调用请求
# 【作用】中间件拿到的入参全集；中间件只读它、不改它（改参数通过 verdict.arguments 输出）
@dataclass
class ToolCallRequest:
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    user_id: str = ""
    principal: Any = None                 # Principal；None = 未启用权限
    capability: Optional[ToolCapability] = None
    iteration: int = 0                    # 第几轮 ReAct（审计用）
    context: Dict[str, Any] = field(default_factory=dict)   # 服务端注入的私有参数

    @property
    def risk(self) -> str:
        return self.capability.risk.value if self.capability else RiskLevel.LOW.value


# 【什么】调用前中间件的裁决
# 【作用】allowed=False 时 execute 根本不会发生，错误不会下发到真实服务
@dataclass
class ToolVerdict:
    allowed: bool                          # True = 放行
    action: str = ACTION_ALLOW             # allow / approve / deny
    reason: str = ""                       # 被拦截的原因（回喂模型）
    arguments: Dict[str, Any] = field(default_factory=dict)   # 最终参数（人工可改）
    code: str = ""                         # 机器可读的拦截码，如 schema_invalid / rate_limited

    # 【什么】构造一个放行裁决
    @classmethod
    def ok(cls, arguments: dict, action: str = ACTION_ALLOW, note: str = "") -> "ToolVerdict":
        return cls(True, action=action, reason=note, arguments=dict(arguments or {}))

    # 【什么】构造一个拦截裁决
    @classmethod
    def block(cls, code: str, reason: str, action: str = ACTION_DENY) -> "ToolVerdict":
        return cls(False, action=action, reason=reason, code=code)
