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
from __future__ import annotations  # 注解延迟求值：允许类型标注前置引用尚未定义的名字

from typing import Optional  # 可选类型标注

from pydantic import BaseModel, Field  # 数据模型基类与字段声明：策略与结论都是 pydantic 模型

from modules.common.logger import get_logger  # 统一日志入口
from modules.orchestration.protocol import Envelope, RejectCode  # 信封与拒绝原因码

_logger = get_logger("guard")  # 本模块日志器：名字即模块名，日志可溯源

# ---------------------------------------------------------------------------
# 【默认权限矩阵】哪种消息允许什么角色发
#   "*" 表示任意角色。矩阵的意义在于：
#   delegate（转派）是"指挥权"，只能主管有；
#   objection（异议）是"质疑权"，任何成员都有——否则错误结论没人敢指出；
#   finding / clarify 是信息共享，全员可发。
# ---------------------------------------------------------------------------
DEFAULT_KIND_PERMISSIONS: dict[str, list[str]] = {  # {消息类型: 允许的角色列表}
    "delegate": ["supervisor"],  # 转派=指挥权：只有主管能发
    "objection": ["supervisor", "reviewer", "worker", "general"],  # 异议=质疑权：四类角色都能发
    "finding": ["*"],  # 发现=信息共享：全员可发
    "clarify": ["*"],  # 澄清=信息共享：全员可发
}


# ---------------------------------------------------------------------------
# 【什么】校验规则（pydantic 模型）
# 【作用】把"正文上限/附加字段上限/花名册/白名单/权限矩阵/敏感词表/命中动作/打码文本"
#        八条规则集中成一份可校验配置；全部有默认值，不配置也能跑
# 【为什么用 pydantic】配错参数（比如上限传负数）在启动期就报错
# ---------------------------------------------------------------------------
class GuardPolicy(BaseModel):
    """校验规则。全部有默认值，不配置也能跑。"""

    max_content_chars: int = Field(default=4000, ge=1, description="单条消息正文上限（字符）")  # 正文上限：防一条消息撑爆上下文
    max_meta_keys: int = Field(default=20, ge=0, description="附加字段个数上限")  # meta 上限：防附加数据无限膨胀
    known_agents: list[str] = Field(default_factory=list, description="花名册；空=不校验接收者")  # 花名册：由 harness 同步
    allowed_senders: list[str] = Field(default_factory=list, description="可发送者白名单；空=不限制")  # 发送白名单
    kind_permissions: dict[str, list[str]] = Field(
        default_factory=lambda: dict(DEFAULT_KIND_PERMISSIONS),  # 拷贝一份默认矩阵：防多个 policy 实例改到同一张表
        description="消息类型 → 允许的角色列表",
    )
    sensitive_words: list[str] = Field(default_factory=list, description="敏感词表")  # 敏感词：空表=不扫
    on_sensitive: str = Field(default="redact", description="命中敏感词后的动作：redact|reject")  # 命中动作：默认打码
    redact_mask: str = Field(default="***", description="打码替换成的文本")  # 打码文本

    # 【什么】查权限矩阵：这个角色能不能发这种消息
    # 【作用】矩阵里没写的类型默认放行（新增类型不至于全被拦）；
    #        "*" 通配任意角色；角色比较不分大小写
    def allows(self, kind: str, role: str) -> bool:  # kind=消息类型；role=发送者角色
        allowed = self.kind_permissions.get(kind)  # 查矩阵：这个类型允许哪些角色
        if allowed is None:          # 矩阵里没写的类型：默认放行，避免新增类型就全被拦
            return True
        return "*" in allowed or (role or "").lower() in [a.lower() for a in allowed]  # 通配符命中 或 角色（小写对齐后）在名单里

    # 【什么】扫敏感词：正文里命中了哪些词
    # 【作用】返回命中词列表；空列表=没命中
    # 【为什么逐个 in 而不是正则】词表是运维可配的，正则写错一个就全崩；in 朴素但稳
    def scan_sensitive(self, text: str) -> list[str]:  # 扫一遍正文
        hits = []  # 命中收集器
        low = text or ""  # None 兜底成空串
        for w in self.sensitive_words:  # 逐个词过
            if w and w in low:  # 空词跳过；命中就记下
                hits.append(w)  # 记下命中词：打码与留痕都要它
        return hits  # 返回全部命中


# ---------------------------------------------------------------------------
# 【什么】一次校验的结论
# 【作用】check 的唯一返回类型：过没过 + 原因码 + 人读理由 + 信封 + 打码清单
# 【为什么带信封】见字段注释：打码会改正文，调用方必须拿到改后的对象
# ---------------------------------------------------------------------------
class GuardVerdict(BaseModel):
    """一次校验的结论。"""

    passed: bool = Field(description="是否放行")  # 总开关：False = 这条消息到此为止
    code: RejectCode = Field(default=RejectCode.OK)  # 机器读的原因码：监控按它聚合
    reason: str = Field(default="")  # 人读的理由
    # 【为什么返回改写后的信封而不是返回 bool】
    #   敏感词打码会改正文，调用方必须拿到改后的对象再投递；
    #   只回 bool 的话，投递的还是原文，打码等于没做。
    envelope: Optional[Envelope] = None  # 校验后的信封（可能已被打码改写）
    redacted: list[str] = Field(default_factory=list, description="被打码的敏感词")  # 打码清单：审计用


# ---------------------------------------------------------------------------
# 【什么】按策略校验一条信封的执行器
# 【作用】编排器的第 1 关：协议→身份→权限→敏感→参数，五连按"先便宜后贵"的顺序过
# 【为什么花名册留在 guard 里一份快照】见 set_roster 的 docstring
# ---------------------------------------------------------------------------
class PolicyGuard:
    """按策略校验一条信封。"""

    # 【什么】构造：没传策略就用全默认策略
    # 【作用】零配置可用；要定制时再注入 GuardPolicy
    def __init__(self, policy: Optional[GuardPolicy] = None):  # None = 全默认
        self.policy = policy or GuardPolicy()  # 策略兜底：没传就建全默认的

    # 【什么】更新花名册
    # 【作用】harness 注册/注销 Agent 时同步调用；校验接收者是否存在就靠这份名单
    def set_roster(self, names: list[str]) -> None:
        """更新花名册（harness 注册/注销 Agent 时同步调用）。

        【为什么花名册要在 guard 里留一份】
            校验接收者是否存在，需要一份名单；每次校验都去问 harness
            会让 guard 依赖运行时。留一份快照，由 harness 在人员变动时同步。
        """
        self.policy.known_agents = list(names)  # 拷贝一份存进策略：防外部改到内部名单

    # 【什么】五连校验：协议→身份→权限→敏感→参数
    # 【作用】任一不过立即返回 GuardVerdict(passed=False,...)；全过返回 passed=True
    def check(self, env: Envelope) -> GuardVerdict:
        """按「协议 → 身份 → 权限 → 敏感 → 参数」的顺序校验。

        【为什么是这个顺序】
            先查最便宜也最基础的（字段缺失），再查需要查表的（身份/权限），
            最后才做全文扫描（敏感词，最贵）。反过来写会让一条空消息
            也去做一遍敏感词全文匹配，白白烧 CPU。
        """
        p = self.policy  # 策略取个短名：下面要用很多次

        # ---- 1. 协议字段 ------------------------------------------------
        if not env.sender or not env.receiver:  # 发送者/接收者任一为空：连"谁发给谁"都说不清
            return GuardVerdict(passed=False, code=RejectCode.BAD_PROTOCOL,
                                reason="sender 与 receiver 不能为空", envelope=env)
        if env.sender == env.receiver:  # 自己发给自己：协议上没意义（自言自语不该走总线）
            return GuardVerdict(passed=False, code=RejectCode.BAD_PROTOCOL,
                                reason="不能发给自己", envelope=env)
        if env.size_bytes > p.max_content_chars:  # 正文超长：一条消息撑爆上下文的口子在这堵
            return GuardVerdict(
                passed=False, code=RejectCode.TOO_LARGE,
                reason=f"正文 {env.size_bytes} 字超过上限 {p.max_content_chars}",
                envelope=env)

        # ---- 2. 身份 ----------------------------------------------------
        if p.allowed_senders and env.sender not in p.allowed_senders:  # 配了白名单且发送者不在里面：拒
            return GuardVerdict(passed=False, code=RejectCode.SENDER_UNKNOWN,
                                reason=f"发送者 {env.sender} 不在白名单", envelope=env)
        # 【广播不校验接收者】"*" 是发给所有人的，逐个校验没有意义
        if env.receiver != "*" and p.known_agents and env.receiver not in p.known_agents:  # 非广播 + 配了花名册 + 接收者不在里面：拒
            return GuardVerdict(passed=False, code=RejectCode.RECEIVER_UNKNOWN,
                                reason=f"接收者 {env.receiver} 不在花名册", envelope=env)

        # ---- 3. 权限矩阵 ------------------------------------------------
        if not p.allows(env.kind.value, env.sender_role):  # 角色无权发这种类型：拒
            return GuardVerdict(
                passed=False, code=RejectCode.KIND_FORBIDDEN,
                reason=f"角色 {env.sender_role or '未标注'} 无权发送 {env.kind.value} 类型消息",
                envelope=env)

        # ---- 4. 敏感词 --------------------------------------------------
        hits = p.scan_sensitive(env.content)  # 全文扫一遍：命中最贵所以放最后
        if hits:  # 有命中
            if p.on_sensitive == "reject":  # 策略是硬拦截：整条拒掉
                return GuardVerdict(
                    passed=False, code=RejectCode.SENSITIVE,
                    reason=f"命中敏感词，已拦截: {hits[:5]}",  # 理由只带前 5 个词：防理由本身爆长
                    envelope=env, redacted=[])
            # 【为什么默认打码而不是拒绝】
            #   直接拒绝会让正常协作被一个词整条打断（成员不知道哪里违规，
            #   只能重发，往往还是同一个词）。打码既保住内容可用，
            #   又确保敏感片段不外传；需要硬拦截的场景改成 reject。
            text = env.content  # 取出原文：接下来逐个词替换
            for w in hits:  # 逐个命中词
                text = text.replace(w, p.redact_mask)  # 全部替换成打码文本（默认 ***）
            env.content = text  # 改写后的正文写回信封：之后投递的就是打码版
            env.meta["redacted"] = hits[:10]  # meta 里留打码清单（最多 10 个）：审计可查
            _logger.warning(f"消息已打码后放行: sender={env.sender} 命中 {len(hits)} 个词")  # 告警日志：打码是要被看见的事件
            return GuardVerdict(passed=True, code=RejectCode.OK,
                                reason=f"已打码 {len(hits)} 处", envelope=env, redacted=hits)  # 放行但带打码清单

        # ---- 5. 参数体积 ------------------------------------------------
        if len(env.meta) > p.max_meta_keys:  # 附加字段个数超上限：防 meta 无限膨胀
            return GuardVerdict(passed=False, code=RejectCode.TOO_LARGE,
                                reason=f"附加字段 {len(env.meta)} 个超过上限 {p.max_meta_keys}",
                                envelope=env)

        return GuardVerdict(passed=True, code=RejectCode.OK, reason="校验通过", envelope=env)  # 五连全过：放行
