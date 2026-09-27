"""
HITL 触发策略模块
=================
决定「什么时候该停下来等人」——把业务判断从链路代码里抽出来。

【为什么策略要独立成一个对象】
    中断判断散落在链路里会变成一堆 if：
        if tool in ("delete",) and level == "admin": ...
        if review_score < 0.6: ...
    改一条规则要翻代码，且没法按环境给不同配置
    （本地 demo 想全部自动放行，生产想高危才拦）。
    抽成 HITLPolicy 后，规则集中在一处、可按 .env 覆盖、可单测。

【默认动作为什么按中断点分别配】
    同样是"没人来处理"，不同环节的代价完全不同：
        调敏感工具 → 默认 reject（宁可不做，也不能做错，错了不可逆）
        出最终答案 → 默认 approve（答案已经过汇总与终审，卡住比放行更糟）
    一刀切给同一个默认值，等于在最危险的地方和最安全的地方用同一个策略。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from modules.hitl.models import HumanAction, InterruptKind


class HITLPolicy(BaseModel):
    """什么时候打断、等太久怎么办。

    Attributes:
        enabled:            总开关。关掉时所有 ask() 立刻返回 approve，零开销
        kinds:              启用的中断点；留空表示全部启用
        sensitive_tools:    调用这些工具前必须人工批准
        timeout_seconds:    单次等待上限
        tool_action:        BEFORE_TOOL 超时兜底（默认 reject）
        final_action:       BEFORE_FINAL 超时兜底（默认 approve）
        plan_action:        BEFORE_PLAN 超时兜底（默认 approve）
        dispute_action:     ON_DISPUTE 超时兜底（默认 approve，按仲裁结果走）
        review_min_score:   终审分低于此值触发 ON_REVIEW_FAIL
        dispute_max_overlap: 产出相似度低于此值判定为"结论冲突"
        budget_ratio:       预算（步数或时间）用掉这个比例时触发 ON_BUDGET
    """

    enabled: bool = Field(default=False, description="HITL 总开关")
    kinds: list[str] = Field(default_factory=list, description="启用的中断点，留空=全部")
    sensitive_tools: list[str] = Field(
        default_factory=list,
        description="需要人工批准的工具名；留空表示不按工具名拦截",
    )
    timeout_seconds: float = Field(default=60.0, gt=0)

    tool_action: HumanAction = Field(default=HumanAction.REJECT)
    final_action: HumanAction = Field(default=HumanAction.APPROVE)
    plan_action: HumanAction = Field(default=HumanAction.APPROVE)
    dispute_action: HumanAction = Field(default=HumanAction.APPROVE)

    review_min_score: float = Field(default=0.6, ge=0.0, le=1.0)
    dispute_max_overlap: float = Field(default=0.35, ge=0.0, le=1.0)
    budget_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    # ------------------------------------------------------------------
    # 【是否启用某个中断点】
    #   kinds 留空 = 全启用（方便"开了就想全看一遍"的调试场景）；
    #   填了就只认填的那些，避免本地调试时被敏感工具的审批弹窗反复打断。
    # ------------------------------------------------------------------
    def kind_enabled(self, kind: InterruptKind) -> bool:
        if not self.enabled:
            return False
        if not self.kinds:
            return True
        return kind.value in self.kinds

    # ------------------------------------------------------------------
    # 【超时兜底动作】按中断点分别取
    #   默认走 approve（放行），只有 BEFORE_TOOL 在下面单独判成 reject
    # ------------------------------------------------------------------
    def default_action_for(self, kind: InterruptKind) -> HumanAction:
        if kind == InterruptKind.BEFORE_TOOL:
            return self.tool_action
        if kind == InterruptKind.BEFORE_FINAL:
            return self.final_action
        if kind == InterruptKind.BEFORE_PLAN or kind == InterruptKind.BEFORE_TASK:
            return self.plan_action
        if kind == InterruptKind.ON_DISPUTE:
            return self.dispute_action
        return HumanAction.APPROVE

    # ------------------------------------------------------------------
    # 【工具是否敏感】返回命中原因，没命中返回 None
    #   匹配规则：先精确匹配全名，再匹配前缀（"mysql_" 一类批量放行/拦截）
    # ------------------------------------------------------------------
    def check_tool(self, tool_name: str) -> Optional[str]:
        if not tool_name or not self.sensitive_tools:
            return None
        name = tool_name.strip()
        for rule in self.sensitive_tools:
            r = rule.strip()
            if not r:
                continue
            if r.endswith("*"):
                if name.startswith(r[:-1]):
                    return f"工具名命中敏感前缀 {r}"
            elif name == r:
                return f"工具 {name} 在人工批准白名单内"
        return None
