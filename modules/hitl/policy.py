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
from __future__ import annotations  # 启用延迟注解求值，允许类型提示前向引用

from typing import Optional  # 可选类型标注

from pydantic import BaseModel, Field  # 配置模型基类与字段定义

from modules.hitl.models import HumanAction, InterruptKind  # 决策动作与中断点枚举


# 【什么】HITL 触发策略
# 【作用】集中定义"什么时候该停下来等人"与"等太久怎么办"
# 【为什么】规则抽成对象后集中一处、可按 .env 覆盖、可单测
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

    enabled: bool = Field(default=False, description="HITL 总开关")  # 总开关，默认关（零开销）
    kinds: list[str] = Field(default_factory=list, description="启用的中断点，留空=全部")  # 启用的中断点白名单
    sensitive_tools: list[str] = Field(  # 敏感工具名单
        default_factory=list,  # 默认空 = 不按工具名拦截
        description="需要人工批准的工具名；留空表示不按工具名拦截",  # 字段说明
    )
    timeout_seconds: float = Field(default=60.0, gt=0)  # 单次等待上限，默认 60 秒

    tool_action: HumanAction = Field(default=HumanAction.REJECT)  # 工具调用超时默认拒绝（宁可不做，也不能做错）
    final_action: HumanAction = Field(default=HumanAction.APPROVE)  # 出答案超时默认放行（卡住比放行更糟）
    plan_action: HumanAction = Field(default=HumanAction.APPROVE)  # 派活 / 任务超时默认放行
    dispute_action: HumanAction = Field(default=HumanAction.APPROVE)  # 冲突裁决超时默认按仲裁结果走

    review_min_score: float = Field(default=0.6, ge=0.0, le=1.0)  # 终审分阈值，低于则触发人工
    dispute_max_overlap: float = Field(default=0.35, ge=0.0, le=1.0)  # 产出相似度阈值，低于判定为结论冲突
    budget_ratio: float = Field(default=0.8, gt=0.0, le=1.0)  # 预算消耗比例阈值，达到则触发人工确认

    # ------------------------------------------------------------------
    # 【是否启用某个中断点】
    #   kinds 留空 = 全启用（方便"开了就想全看一遍"的调试场景）；
    #   填了就只认填的那些，避免本地调试时被敏感工具的审批弹窗反复打断。
    # ------------------------------------------------------------------
    # 【什么】判断某个中断点是否启用
    # 【作用】总开关 + 白名单两级判断
    # 【为什么】kinds 留空 = 全启用方便调试；填了就只认填的，避免本地被反复打断
    def kind_enabled(self, kind: InterruptKind) -> bool:  # 返回该中断点是否启用
        if not self.enabled:  # 总开关关着
            return False  # 一律不启用
        if not self.kinds:  # 白名单为空
            return True  # 全部启用
        return kind.value in self.kinds  # 只认白名单里的

    # ------------------------------------------------------------------
    # 【超时兜底动作】按中断点分别取
    #   默认走 approve（放行），只有 BEFORE_TOOL 在下面单独判成 reject
    # ------------------------------------------------------------------
    # 【什么】按中断点取超时兜底动作
    # 【作用】不同环节的超时代价不同，兜底动作分别配置
    # 【为什么】一刀切等于在最危险和最安全的地方用同一策略
    def default_action_for(self, kind: InterruptKind) -> HumanAction:  # 返回兜底动作
        if kind == InterruptKind.BEFORE_TOOL:  # 工具调用
            return self.tool_action  # 用工具兜底动作（默认 reject）
        if kind == InterruptKind.BEFORE_FINAL:  # 出最终答案
            return self.final_action  # 用答案兜底动作（默认 approve）
        if kind == InterruptKind.BEFORE_PLAN or kind == InterruptKind.BEFORE_TASK:  # 派活前 / 任务前
            return self.plan_action  # 用计划兜底动作（默认 approve）
        if kind == InterruptKind.ON_DISPUTE:  # 结论冲突
            return self.dispute_action  # 用冲突兜底动作（默认 approve）
        return HumanAction.APPROVE  # 其余中断点默认放行

    # ------------------------------------------------------------------
    # 【工具是否敏感】返回命中原因，没命中返回 None
    #   匹配规则：先精确匹配全名，再匹配前缀（"mysql_" 一类批量放行/拦截）
    # ------------------------------------------------------------------
    # 【什么】检查工具是否敏感
    # 【作用】命中返回原因文案，未命中返回 None
    # 【为什么】先精确匹配再前缀匹配，支持 "mysql_*" 一类批量拦截
    def check_tool(self, tool_name: str) -> Optional[str]:  # 返回命中原因或 None
        if not tool_name or not self.sensitive_tools:  # 工具名为空或未配名单
            return None  # 不拦截
        name = tool_name.strip()  # 工具名去空白
        for rule in self.sensitive_tools:  # 逐条匹配敏感规则
            r = rule.strip()  # 规则去空白
            if not r:  # 空规则
                continue  # 跳过
            if r.endswith("*"):  # 前缀规则（如 "mysql_*"）
                if name.startswith(r[:-1]):  # 工具名命中前缀
                    return f"工具名命中敏感前缀 {r}"  # 返回命中原因
            elif name == r:  # 精确匹配全名
                return f"工具 {name} 在人工批准白名单内"  # 返回命中原因
        return None  # 全部规则未命中
