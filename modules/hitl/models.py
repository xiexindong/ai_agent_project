"""
Human-in-the-Loop 数据契约模块
================================
定义「Agent 跑到一半停下来等人」这件事的全部数据形状。

【HITL 到底解决什么】
    Agent 自己能跑，但有四类事它不该自己拍板：
        1. 高危动作   —— 删库、转账、对外发消息，错了不可逆
        2. 依据不足   —— 检索为空、置信度低，硬答就是编
        3. 结论打架   —— 两个 Agent 给出相反数字，谁对得人来判断
        4. 昂贵动作   —— 超预算的调用链，烧钱前该有人点个头
    这四类统一抽象成：Agent 抛出一个 InterruptRequest，
    人回一个 HumanReply，Agent 按回复继续。

【为什么必须先定死数据契约，和 multiagent 一样】
    中断点散落在链路各处（派活前、调工具前、出答案前、终审后），
    人这一侧（API / 前端审批页 / 命令行）又是另一个进程。
    两边靠 JSON 传话，没有 BaseModel 就会变成各自拼字典，
    字段一漂移就是"人点了批准，Agent 没收到"这种最难查的故障。

【四种人工动作，为什么只有四种】
    approve 照原样放行
    modify  带着修改内容放行（人改了指令/参数，Agent 用改后的继续）
    reject  这一步不做，但流程继续（跳过这个子任务、换个方案）
    abort   整次运行终止
    这四种覆盖了"继续 / 改着继续 / 局部放弃 / 全部放弃"四种意图，
    再多是人为复杂化，实际审批页上也就是这几个按钮。
"""
from __future__ import annotations  # 启用延迟注解求值，允许类型提示前向引用

import time  # 取当前时间戳
import uuid  # 生成请求 ID
from enum import Enum  # 枚举基类
from typing import Any, Optional  # 类型标注

from pydantic import BaseModel, Field, field_validator  # 数据契约基类、字段与校验器


# 【什么】生成中断请求短 ID
# 【作用】产出 hitl_a1b2c3d4 形式的标识
# 【为什么】uuid4 前 8 位在日志里好认又不撞车
def new_interrupt_id() -> str:  # 返回短 ID 字符串
    """生成短 ID：hitl_a1b2c3d4。用 uuid4 前 8 位，日志里好认又不撞车。"""
    return f"hitl_{uuid.uuid4().hex[:8]}"  # 取 uuid4 十六进制前 8 位加前缀


# 【什么】当前毫秒时间戳
# 【作用】给所有时间字段统一供时
# 【为什么】统一毫秒避免混用秒和毫秒
def now_ms() -> int:  # 返回毫秒时间戳
    """当前毫秒时间戳。所有时间字段统一用它，避免混用秒和毫秒。"""
    return int(time.time() * 1000)  # 秒转毫秒取整


# 【什么】中断点类型枚举
# 【作用】标识 Agent 在哪个环节停下来等人
# 【为什么】按"环节"分类（有限七个）而非按"原因"（可后天加），分类稳定处理才好写
class InterruptKind(str, Enum):
    """中断点类型：Agent 在哪个环节停下来等人。

    【为什么按"环节"分类而不是按"原因"】
        原因（敏感工具/依据不足/预算超了）是policy 的事，可以后天加；
        环节是有限的六个，且决定了"人在什么时候被打断、能看到什么上下文"。
        分类稳定，处理才好写。
    """

    BEFORE_PLAN = "before_plan"          # 派活前：让人确认任务分解是否合理
    BEFORE_TASK = "before_task"          # 单个子任务执行前
    BEFORE_TOOL = "before_tool"          # 调用敏感工具前：最典型的一处
    BEFORE_FINAL = "before_final"        # 出最终答案前：最后一道人工关卡
    ON_REVIEW_FAIL = "on_review_fail"    # 终审不通过：让真人接手
    ON_DISPUTE = "on_dispute"            # 多 Agent 结论冲突：让人裁决
    ON_BUDGET = "on_budget"              # 预算/步数超阈值：继续烧钱前确认


# 【什么】人工决策动作枚举
# 【作用】覆盖"继续 / 改着继续 / 局部放弃 / 全部放弃"四种意图
# 【为什么】四种足够覆盖审批页上的全部按钮，再多是人为复杂化
class HumanAction(str, Enum):
    """人工决策动作。"""

    APPROVE = "approve"      # 照原样放行
    MODIFY = "modify"        # 带修改内容放行（看 HumanReply.edited）
    REJECT = "reject"        # 这一步跳过，流程继续
    ABORT = "abort"          # 整次运行终止

    # 【什么】是否继续执行
    # 【作用】approve 与 modify 都算"继续"
    # 【为什么】调用方只关心"过不过"，内容差异另外取
    @property  # 只读属性
    def is_go(self) -> bool:  # 返回是否继续
        """是否继续执行（approve 与 modify 都是"继续"，只是内容可能不同）。"""
        return self in (HumanAction.APPROVE, HumanAction.MODIFY)  # 放行或修改放行都算继续


# 【什么】中断请求生命周期状态枚举
# 【作用】标识一次请求走到哪一步
# 【为什么】状态机清晰才能挡住重复提交、识别超时与作废
class InterruptStatus(str, Enum):
    """一次中断请求的生命周期状态。"""

    WAITING = "waiting"        # 已抛出，等人
    RESOLVED = "resolved"      # 人已回复
    EXPIRED = "expired"        # 超时，已按 default_action 兜底
    CANCELLED = "cancelled"    # 流程提前结束，请求作废


# 【什么】一次「请人拍板」的请求
# 【作用】承载中断的全部信息：谁抛的、为什么、给人看什么、超时怎么办
# 【为什么】default_action 必填：人可能永远不来，没兜底动作 Agent 就永远挂住
class InterruptRequest(BaseModel):
    """一次「请人拍板」的请求。

    【default_action 为什么必填】
        人可能永远不来（下班了、审批页没开着）。
        没有默认动作，Agent 就永远卡在这一步 —— 比答错更糟，
        因为它是"挂住"而不是"错"，监控上看不到失败率异常，
        只有用户投诉"系统没反应"才会被发现。
        所以每个请求必须自带兜底动作和超时时间。

    【输入 → 输出对照】
        输入: InterruptRequest(kind=BEFORE_TOOL, title="调用 data_query",
                              context={"tool": "data_query", "args": {"id": "1"}},
                              default_action="reject", timeout_seconds=30)
        输出: id="hitl_3f9a2c11" status=WAITING
              人未响应 30 秒后 → status=EXPIRED，按 reject 处理
    """

    id: str = Field(default_factory=new_interrupt_id)  # 请求 ID，自动生成短 ID
    run_id: str = Field(default="", description="所属运行 ID，便于与 RunResult 关联")  # 运行标识
    session_id: str = Field(default="", description="会话 ID，前端按会话展示审批卡片")  # 会话标识
    agent: str = Field(default="", description="抛出中断的 Agent 名")  # 来源 Agent
    kind: InterruptKind = Field(description="中断点类型")  # 在哪个环节停下
    title: str = Field(description="一句话说明要人拍板什么，直接显示在审批页")  # 标题（必填，校验非空）
    reason: str = Field(default="", description="为什么需要人：命中了哪条策略")  # 触发原因
    context: dict = Field(default_factory=dict, description="给人看的上下文（工具名/参数/草稿等）")  # 上下文
    options: list[str] = Field(default_factory=list, description="可选动作；为空表示四种全开放")  # 可选动作
    default_action: HumanAction = Field(default=HumanAction.APPROVE, description="超时或无人处理时的兜底动作")  # 兜底动作
    timeout_seconds: float = Field(default=60.0, gt=0, description="最长等待秒数")  # 等待上限
    created_at: int = Field(default_factory=now_ms)  # 创建时间（毫秒）
    status: InterruptStatus = Field(default=InterruptStatus.WAITING)  # 生命周期状态，初始等待中

    # 【什么】标题非空校验
    # 【作用】拒绝空白标题
    # 【为什么】审批页上空白标题等于没法看
    @field_validator("title")  # 声明对 title 字段的校验器
    @classmethod  # 类方法校验器
    def _title_not_blank(cls, v: str) -> str:  # 返回清洗后的标题
        if not v.strip():  # 去掉空白后为空
            raise ValueError("中断请求必须有标题，否则审批页上是一片空白")  # 抛校验错误
        return v.strip()  # 返回去空白后的标题

    # 【什么】请求是否已终结
    # 【作用】非 WAITING 即终结
    # 【为什么】终结判断收敛一处，避免各处自己比状态
    @property  # 只读属性
    def is_terminal(self) -> bool:  # 返回是否已结束
        return self.status != InterruptStatus.WAITING  # 不在等待即终结

    # 【什么】到期时间戳
    # 【作用】统一超时判定的计算口径
    # 【为什么】避免各处自己算 created_at + timeout 算出不一致
    @property  # 只读属性
    def deadline_ms(self) -> int:  # 返回到期毫秒时间戳
        """到期时间戳。超时判定统一用它，避免各处自己算 created_at + timeout。"""
        return self.created_at + int(self.timeout_seconds * 1000)  # 创建时间 + 等待毫秒数


# 【什么】人对一次中断的回复
# 【作用】承载决策动作与修改内容
# 【为什么】modify 单独带 edited：审批页最常见的操作是"改一下再放行"，改后内容直接取用
class HumanReply(BaseModel):
    """人对一次中断的回复。

    【modify 为什么单独带 edited 字段】
        审批页上最常见的操作不是"批准/拒绝"，而是"改一下再放行"
        （改 SQL 条件、改金额上限、改收件人）。
        把修改后的内容放进 edited，Agent 侧直接取用，
        不用再去人写的话里正则抠出参数。
    """

    request_id: str = Field(description="对应的 InterruptRequest.id")  # 关联的请求 ID
    action: HumanAction = Field(description="决策动作")  # 决策动作
    edited: str = Field(default="", description="modify 时的修改后内容（指令/参数/草稿）")  # 修改内容
    operator: str = Field(default="", description="操作人标识，进审计")  # 操作人
    note: str = Field(default="", description="备注理由")  # 备注
    replied_at: int = Field(default_factory=now_ms)  # 回复时间（毫秒）

    # 【什么】edited 字段的软校验
    # 【作用】只在 modify 时提示，不强制要求填
    # 【为什么】有些审批页的"修改"是改表单字段，编辑框本身可以留空；软提示不阻断
    @field_validator("edited")  # 声明对 edited 字段的校验器
    @classmethod  # 类方法校验器
    def _edited_only_with_modify(cls, v: str, info) -> str:  # 返回原值（当前不做硬校验）
        # 【为什么只在 modify 时校验而不是强制要求填】
        #   有些审批页的"修改"是改表单里的字段，编辑框本身可以留空
        #   （比如只勾掉了某个选项）。这里只做软提示，不阻断。
        return v  # 原样返回

    # 【什么】是否继续执行
    # 【作用】透传动作的 is_go
    # 【为什么】调用方直接问回复"过不过"，不用先取动作再判断
    @property  # 只读属性
    def is_go(self) -> bool:  # 返回是否继续
        return self.action.is_go  # 由动作决定

    # 【什么】取最终要用的内容
    # 【作用】modify 且有编辑内容 → 用编辑后的；否则用原文
    # 【为什么】调用方一处取内容，不用自己判断 modify 分支
    def content_or(self, original: str) -> str:  # original: 原始内容；返回最终内容
        """取"最终要用的内容"：modify 且有编辑内容 → 用编辑后的；否则用原文。

        【输入 → 输出对照】
            输入: action=modify, edited="查询 2024 年营收", original="查营收"
            输出: "查询 2024 年营收"
            输入: action=approve, edited="", original="查营收"
            输出: "查营收"
        """
        if self.action == HumanAction.MODIFY and self.edited.strip():  # 修改放行且编辑内容非空
            return self.edited.strip()  # 用改后内容
        return original  # 否则用原文


# 【什么】请求 + 回复的成对归档
# 【作用】一次人工介入的完整留痕
# 【为什么】审计需要"谁在什么时候、为什么被打断、人怎么回的"成对可查
class InterruptRecord(BaseModel):
    """请求 + 回复 的成对归档，一次人工介入的完整留痕。"""

    request: InterruptRequest  # 原始请求
    reply: Optional[HumanReply] = None  # 人工回复；超时兜底前可能为空
    resolved_at: int = 0  # 解决时间（毫秒）；0 表示未解决

    # 【什么】最终生效的动作
    # 【作用】有回复用回复的动作，超时用请求的兜底动作
    # 【为什么】审计看"最终按什么执行了"，一处取数不用自己判空
    @property  # 只读属性
    def final_action(self) -> HumanAction:  # 返回最终动作
        """最终生效的动作：有回复用回复，超时用兜底。"""
        return self.reply.action if self.reply else self.request.default_action  # 回复优先，兜底垫后
