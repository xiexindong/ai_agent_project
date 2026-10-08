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
from __future__ import annotations           # 允许类型注解里用未导入的名字（Python 3.7+ 兼容写法）

from dataclasses import dataclass, field     # dataclass 装饰器与字段默认值工厂
from enum import Enum                        # 枚举基类：RiskLevel 用它定义风险等级
from typing import Any, Dict, Optional, Tuple  # 标准库类型：Any 任意 / Dict 字典 / Optional 可空 / Tuple 元组

# 【三个裁决动作】
ACTION_ALLOW = "allow"      # 直接执行
ACTION_APPROVE = "approve"  # 需人工批准（模型只能发起申请）
ACTION_DENY = "deny"        # 禁止执行（配置层面封死）


# 【什么】风险等级：工具副作用的分级标尺
# 【作用】策略按等级决定「直接跑 / 要审批 / 禁止跑」，而不是逐个工具写 if
# 【为什么】等级制让策略可以写"risk >= HIGH 都要审批"这类通用规则，新增工具自动落入管控
class RiskLevel(str, Enum):
    """风险等级：从只读到不可逆。等级越高，越不允许模型自行决定。"""

    LOW = "low"            # 只读、无副作用，如 calculator / data_query
    MEDIUM = "medium"      # 有副作用但可回滚，如写草稿、发通知
    HIGH = "high"          # 影响外部系统，如删数据、改配置、下单
    CRITICAL = "critical"  # 不可逆或影响生产，如删库、改生产配置

    # 【什么】等级转序号
    # 【作用】让策略能写「risk >= HIGH 都要审批」这类比较
    # 【为什么】枚举本身不能直接比大小，映射成整数才能用 >= 表达"某级及以上"
    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]  # 等级字符串 → 序号（0~3）

    # 【什么】宽松解析：字符串/None/非法值都落到 LOW
    # 【作用】配置里写错等级不会让启动崩掉，退化成最低风险（但不会悄悄放宽高风险工具）
    # 【为什么】解析失败兜底成 LOW 是"配置错了也能启动"，生效层会再用覆盖规则把高风险提回来
    @classmethod
    def parse(cls, value: Any) -> "RiskLevel":
        if isinstance(value, RiskLevel):                     # 已经是枚举实例
            return value                                     # 原样返回
        text = str(value or "").strip().lower()              # 归一化：转字符串、去空白、转小写；None 兜底空串
        for level in cls:                                    # 遍历四个等级
            if level.value == text:                          # 值匹配
                return level                                 # 返回对应枚举
        return cls.LOW                                       # 都不匹配（含空串/非法值）→ 兜底 LOW


# 【什么】单个工具的能力声明
# 【作用】工具注册时填一次，之后所有校验都读它，不再散落在 if 里
# 【为什么】能力集中成数据结构后，策略/中间件/审计都读同一份，不会出现各处口径不一
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
    # 【为什么】归成一个属性，策略与 Harness 不用各自重复"等级 >= high 或显式声明"这段判断
    @property
    def needs_approval(self) -> bool:
        return self.requires_approval or self.risk.rank >= RiskLevel.HIGH.rank  # 显式声明 或 等级≥high → 需人工批准

    # 【什么】转成可序列化字典
    # 【作用】/tools 接口、审计 detail、日志展示都用它
    # 【为什么】枚举与元组不能直接 JSON 序列化，统一在这里转成纯 Python 基础类型
    def to_dict(self) -> dict:
        return {
            "name": self.name,                             # 工具名
            "risk": self.risk.value,                       # 风险等级（枚举 → 字符串）
            "side_effect": self.side_effect,               # 是否有副作用
            "sandbox": self.sandbox,                       # 隔离方式
            "requires_approval": self.requires_approval,   # 是否显式要求人工批准
            "needs_approval": self.needs_approval,         # 综合判定后是否需人工批准
            "permissions": list(self.permissions),         # 权限码（元组 → 列表，可 JSON 化）
            "data_scopes": list(self.data_scopes),         # 数据域（元组 → 列表，可 JSON 化）
            "max_result_chars": self.max_result_chars,     # 结果裁剪上限
            "timeout_seconds": self.timeout_seconds,       # 单次执行超时
            "description": self.description,               # 能力说明
        }


# 【什么】工具裁决结果
# 【作用】策略对「能不能调」给出的唯一答案，带原因（原因会回喂给模型）
# 【为什么】裁决独立成对象：action 给机器分支用，reason 给模型理解用，各取所需
@dataclass
class ToolDecision:
    action: str = ACTION_ALLOW     # allow / approve / deny
    reason: str = ""               # 裁决原因（拦截/需审批时回喂模型）
    risk: str = RiskLevel.LOW.value  # 裁决时的风险等级快照（审计用）

    # 【什么】是否被禁止
    # 【作用】before_call 用它快速分支出"配置层面封死"的情况
    @property
    def denied(self) -> bool:
        return self.action == ACTION_DENY                  # action 是 deny → 被禁止


# 【什么】工具安全策略
# 【作用】把「哪些等级/哪些工具禁止执行、哪些需要审批」从代码里挪到配置
# 【为什么】管控口径是运维决策，改配置即生效，不该改代码发版
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
    # 【为什么】配置来源可能是字符串也可能是列表，统一收口成元组，下游不用判断形态
    @staticmethod
    def _split(text: Any) -> Tuple[str, ...]:
        if not text:                                         # 空值（None/空串/空列表）
            return ()                                        # 返回空元组
        if isinstance(text, (list, tuple, set)):             # 已是序列形态
            return tuple(str(x).strip() for x in text if str(x).strip())  # 逐项转字符串、去空白、滤空项
        return tuple(x.strip() for x in str(text).split(",") if x.strip())  # 字符串按逗号切分、去空白、滤空项

    # 【什么】按配置构造策略
    # 【作用】config.py 里那组 TOOL_* 配置一次转成一个策略对象
    # 【为什么】配置读取集中在这一个工厂方法里，配置项改名只动这里
    @classmethod
    def from_settings(cls, s: Any) -> "ToolPolicy":
        overrides: Dict[str, str] = {}                       # 先开一张空的覆盖表
        for item in cls._split(getattr(s, "tool_risk_overrides", "")):  # 逐条解析 TOOL_RISK_OVERRIDES
            if ":" in item:                      # 形如 web_search:medium
                name, level = item.split(":", 1)             # 只按第一个冒号切：名字里不会含冒号，等级里可能含
                overrides[name.strip()] = level.strip()      # 写入覆盖表：工具名/前缀 → 等级字符串
        return cls(
            default_risk=RiskLevel.parse(getattr(s, "tool_default_risk", "low")),  # 兜底等级；非法值落到 LOW
            overrides=overrides,                             # 名字/前缀 → 等级覆盖表
            deny_levels=cls._split(getattr(s, "tool_deny_levels", "critical")),  # 禁止等级名单
            approve_levels=cls._split(getattr(s, "tool_approve_levels", "high,critical")),  # 审批等级名单
            deny_tools=cls._split(getattr(s, "tool_deny_tools", "")),   # 禁止工具名单
            approve_tools=cls._split(getattr(s, "tool_approve_tools", "")),  # 审批工具名单
            on_no_gate=str(getattr(s, "tool_on_no_gate", "deny") or "deny"),  # 无闸门时策略；空值兜底 deny（安全优先）
        )

    # 【什么】名字是否命中带 * 的规则
    # 【作用】"drop_*" 能命中 drop_table / drop_index
    # 【为什么】前缀通配让黑名单按"一族"封禁，不用枚举每个危险工具名
    @staticmethod
    def _match(name: str, rules: Tuple[str, ...]) -> bool:
        for rule in rules:                                   # 逐条规则比对
            if rule.endswith("*"):                 # 通配规则："drop_*" 去掉 * 后是前缀 "drop_"
                if name.startswith(rule[:-1]):     # name 以该前缀开头即命中：drop_table / drop_index 都中
                    return True                              # 命中：通配匹配成功
            elif rule == name:                     # 精确规则：名字一字不差才命中
                return True                                  # 命中：精确匹配成功
        return False                               # 所有规则都没中：返回 False（由调用方决定放行还是拦）

    # 【什么】把工具自己声明的能力与配置覆盖合并
    # 【作用】覆盖只改风险等级，其余字段仍以工具声明为准
    # 【为什么】分级是运维口径，副作用/权限/沙箱是工具事实，两类信息各归各管
    def effective(self, cap: ToolCapability) -> ToolCapability:
        risk = cap.risk                                      # 先取工具自己声明的等级
        if cap.risk.rank <= RiskLevel.LOW.rank and self.default_risk.rank > risk.rank:  # 工具按 LOW 兜底且默认等级更高
            risk = self.default_risk                      # 工具没声明 → 用默认等级
        for pattern, level in self.overrides.items():        # 遍历配置覆盖表
            # 【作用】按名字查配置有没有给这个工具单独定级（如 web_search:medium）；命中即覆盖工具声明并停止后续比对
            if self._match(cap.name, (pattern,)):            # 名字命中这条覆盖规则
                # 【作用】把配置写的等级字符串（"medium"）转成枚举，替换掉工具自己声明的等级
                risk = RiskLevel.parse(level)             # 配置覆盖优先于工具声明
                break                                        # 命中即停：后面的规则不再比对
        if risk is cap.risk:                                 # 等级没被任何规则改动
            return cap                                       # 原对象直接返回，不新建
        return ToolCapability(
            name=cap.name, risk=risk, side_effect=cap.side_effect, sandbox=cap.sandbox,  # 名字/新等级/副作用/沙箱
            requires_approval=cap.requires_approval, permissions=cap.permissions,        # 审批要求/权限码
            data_scopes=cap.data_scopes, max_result_chars=cap.max_result_chars,          # 数据域/裁剪上限
            timeout_seconds=cap.timeout_seconds, description=cap.description,            # 超时/说明
        )

    # 【什么】对一次调用做裁决
    # 【作用】唯一的"能不能调"判定入口；返回的 reason 会原样回喂给模型
    # 【为什么】五条互斥规则命中即返回，deny 永远排在 approve 前，不给封死工具留审批口子
    def decide(self, cap: ToolCapability) -> ToolDecision:
        """按五条互斥规则裁决，命中即返回：deny名 → deny级 → approve名 → approve级 → allow。

        【为什么 deny 排在 approve 前】一个工具既在名字黑名单、风险又是
            high 时，禁止优先于审批 —— 不能给一个本该封死的工具留"批准后放行"的口子。

        【输入输出对照】deny_levels=("critical",), approve_levels=("high",)：
            drop_table 命中 deny_tools "drop_*" → deny（名，盖过等级）
            mysql_exec(critical)                → deny（级）
            write_draft(medium) 命中 approve_tools → approve（名，把中风险单独提级）
            update_config(high)                 → approve（级）
            calculator(low)                    → allow（兜底）
        """
        name = cap.name                                      # 取工具名，下面各条规则反复用
        # 【规则①：名字黑名单】精确名或前缀通配（drop_*）命中 → 直接禁止
        if self._match(name, self.deny_tools):               # 名字命中禁止名单
            return ToolDecision(ACTION_DENY, f"工具 {name} 已配置为禁止执行", cap.risk.value)  # 裁决：禁止
        # 【规则②：风险等级黑名单】等级在 deny_levels 里（默认 critical）→ 禁止
        #   str() 转换：cap.risk 虽是 str 枚举，这里仍显式转，集合判断更稳
        if str(cap.risk.value) in self.deny_levels:          # 等级命中禁止名单
            return ToolDecision(ACTION_DENY, f"工具 {name} 风险等级为 {cap.risk.value}，已配置为禁止执行", cap.risk.value)  # 裁决：禁止
        # 【规则③：名字审批名单】可以把某个等级没到 high 的工具单独提级为需人工批准
        if self._match(name, self.approve_tools):            # 名字命中审批名单
            return ToolDecision(ACTION_APPROVE, f"工具 {name} 已配置为需人工批准", cap.risk.value)  # 裁决：需审批
        # 【规则④：风险等级审批名单 或 工具显式声明】
        #   等级在 approve_levels（默认 high/critical），或 cap.requires_approval=True
        #   （工具自己声明"我必须人批"，即使等级被配置改低了也拦住）
        if str(cap.risk.value) in self.approve_levels or cap.requires_approval:  # 等级需审批 或 工具显式要求审批
            return ToolDecision(ACTION_APPROVE, f"工具 {name} 风险等级为 {cap.risk.value}，需人工批准", cap.risk.value)  # 裁决：需审批
        # 【规则⑤：兜底放行】四条拦截都没中（典型 low 级只读工具）→ 直接执行
        #   reason 留空：放行不需要向模型解释
        return ToolDecision(ACTION_ALLOW, "", cap.risk.value)  # 裁决：放行


# 【什么】一次工具调用请求
# 【作用】中间件拿到的入参全集；中间件只读它、不改它（改参数通过 verdict.arguments 输出）
# 【为什么】请求对象化后，中间件契约稳定，新增字段（如 iteration）不用改每个中间件签名
@dataclass
class ToolCallRequest:
    tool_name: str                             # 工具名
    arguments: Dict[str, Any] = field(default_factory=dict)  # 模型传入的实参
    session_id: str = ""                       # 会话标识（预算/限流/审计的分组键）
    user_id: str = ""                          # 用户标识（限流兜底键之一）
    principal: Any = None                 # Principal；None = 未启用权限
    capability: Optional[ToolCapability] = None  # 生效后的能力声明（注册期挂上来）
    iteration: int = 0                    # 第几轮 ReAct（审计用）
    context: Dict[str, Any] = field(default_factory=dict)   # 服务端注入的私有参数

    # 【什么】请求的风险等级（便捷属性）
    # 【作用】没挂能力声明时按 LOW 兜底，调用方不用判空
    @property
    def risk(self) -> str:
        return self.capability.risk.value if self.capability else RiskLevel.LOW.value  # 有能力声明取其等级，否则 LOW


# 【什么】调用前中间件的裁决
# 【作用】allowed=False 时 execute 根本不会发生，错误不会下发到真实服务
# 【为什么】裁决与执行分离：拦截在这里定型，执行层只认 allowed=True 的请求
@dataclass
class ToolVerdict:
    allowed: bool                          # True = 放行
    action: str = ACTION_ALLOW             # allow / approve / deny
    reason: str = ""                       # 被拦截的原因（回喂模型）
    arguments: Dict[str, Any] = field(default_factory=dict)   # 最终参数（人工可改）
    code: str = ""                         # 机器可读的拦截码，如 schema_invalid / rate_limited

    # 【什么】构造一个放行裁决
    # 【作用】allowed=True；note 可带"人工已批准"等说明；参数拷贝防外部篡改
    @classmethod
    def ok(cls, arguments: dict, action: str = ACTION_ALLOW, note: str = "") -> "ToolVerdict":
        return cls(True, action=action, reason=note, arguments=dict(arguments or {}))  # 放行裁决；None 参数兜底成空 dict

    # 【什么】构造一个拦截裁决
    # 【作用】allowed=False；code 给机器分支用，reason 回喂模型
    @classmethod
    def block(cls, code: str, reason: str, action: str = ACTION_DENY) -> "ToolVerdict":
        return cls(False, action=action, reason=reason, code=code)  # 拦截裁决；不带 arguments（不会被执行）
