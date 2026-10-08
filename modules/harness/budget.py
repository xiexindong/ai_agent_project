"""
执行预算与循环控制
==================
管住三件事：调用次数、回灌上下文的 token、墙钟时间。

【为什么必须单独管】
    ReAct 循环里最大的风险不是"调错工具"，而是"停不下来"：
        - 工具每次返回一大段 → 全部塞进下一轮 prompt（上下文膨胀）
        - 模型没拿到想要的字段 → 换参数再调一次同一个工具
        - 某个工具卡住 → 整条会话挂在那里
    这三样都不靠"模型自觉"，只能由 Harness 用硬配额卡死。

【三个配额各自管什么】
    max_calls_per_run   单次会话最多调几次工具（防整体失控）
    max_calls_per_tool  单个工具最多调几次（防死循环，取代原先硬编码的 5）
    token_budget        回灌上下文的累计 token（防上下文膨胀）
    timeout_seconds     墙钟超时（防卡死）
"""
from __future__ import annotations           # 允许类型注解里用未导入的名字（Python 3.7+ 兼容写法）

import threading                             # 标准库线程锁：多会话并发建账/销账的互斥保护
import time                                  # 标准库时间：墙钟计时（建账时刻/到期判定）
from dataclasses import dataclass, field     # dataclass 装饰器与字段默认值工厂
from typing import Dict, Optional            # 标准库类型：Dict 字典 / Optional 可空

from modules.common.logger import get_logger # 统一日志工厂

_logger = get_logger("harness.budget")       # 模块级 logger，名字 = "harness.budget"


# 【什么】一次运行的预算上限
# 【作用】全部来自配置；不传则用下面的默认值，保证离线可跑
# 【为什么】配额是运维口径，做成 dataclass 后配置项到字段一一对应，改配置不改逻辑
@dataclass
class BudgetSpec:
    max_calls_per_run: int = 8             # 单次会话工具调用总次数
    max_calls_per_tool: int = 5            # 单工具累计次数（防死循环）
    token_budget: int = 6000               # 回灌上下文的累计 token 预算
    timeout_seconds: float = 120.0         # 墙钟超时（秒）
    max_result_chars: int = 2000           # 工具返回值裁剪上限（交给 ResultFilter 执行）


# 【什么】一次运行的预算账本
# 【作用】边跑边记账，任一配额打满就拦截后续调用
# 【为什么】账本按会话隔离且不共享：一个会话的配额消耗绝不影响另一个会话
class RunBudget:
    """单次运行的预算账本（非线程共享，按会话隔离）。

    【输入输出对照】
        spec = BudgetSpec(max_calls_per_run=2, max_calls_per_tool=1)
        b = RunBudget("s1", spec)
        b.check("calculator")       → (True, "")
        b.consume("calculator", 30)
        b.check("calculator")       → (False, "工具 calculator 单次运行最多调用 1 次")
        b.check("data_query")       → (False, "本次运行工具调用已达上限 2 次")
    """

    # 【什么】构造函数
    # 【作用】建账：记录归属会话、配额表、起止时刻，清零各计数器
    def __init__(self, session_id: str, spec: Optional[BudgetSpec] = None):
        self.session_id = session_id                   # 账本归属的会话；日志与快照用它定位是哪次对话
        self.spec = spec or BudgetSpec()               # 配额表；不传就用默认（8 次/5 次/6000 token/120 秒）
        self.started = time.time()                     # 建账时刻；deadline 与 remaining_seconds 都以它为起点
        self.deadline = self.started + max(0.0, float(self.spec.timeout_seconds))   # 墙钟到期时刻；max 防负数配置导致"一建账就超时"
        self.total_calls = 0                       # 已调用次数（含被拦截的不计）
        self.per_tool: Dict[str, int] = {}         # 工具名 → 已调用次数
        self.tokens = 0                            # 已回灌上下文的 token 累计
        self.blocked = 0                           # 被预算拦截的次数（观测用）

    # 【什么】距超时还剩几秒
    # 【作用】0 表示已经超时；调用方据此决定"还要不要继续"
    # 【为什么】max(0.0, ...) 保证已超时也返回 0 而不是负数，调用方判断口径统一
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.time())     # 到期时刻减当前时刻，负数钳到 0

    # 【什么】是否已超时
    # 【作用】check() 的第一道闸门；做成属性便于 if budget.expired 直读
    @property
    def expired(self) -> bool:
        return self.remaining_seconds() <= 0.0           # 剩余 ≤ 0 → 已超时

    # 【什么】调用前检查
    # 【作用】超时 / 总次数 / 单工具次数 / token 四道闸门，任一打满返回 False + 原因
    # 【为什么】只"问"不扣：配额要在调用成功后才记，被拦的调用不该消耗额度
    def check(self, tool_name: str) -> tuple[bool, str]:
        if self.expired:                                     # 闸门①：墙钟已超时
            return False, f"本次运行已超时（{self.spec.timeout_seconds:.0f}s），停止调用工具"  # 拦截：超时
        if self.total_calls >= self.spec.max_calls_per_run:  # 闸门②：总调用次数打满
            return False, f"本次运行工具调用已达上限 {self.spec.max_calls_per_run} 次"          # 拦截：总次数
        used = self.per_tool.get(tool_name, 0)               # 取这个工具已调用的次数（没调过记 0）
        if used >= self.spec.max_calls_per_tool:             # 闸门③：单工具次数打满
            return False, f"工具 {tool_name} 单次运行最多调用 {self.spec.max_calls_per_tool} 次，疑似死循环"  # 拦截：单工具死循环
        if self.tokens >= self.spec.token_budget:            # 闸门④：token 预算打满
            return False, f"上下文 token 预算已用尽（{self.spec.token_budget}），停止调用工具"  # 拦截：token 用尽
        return True, ""                                      # 四道全过 → 放行（原因留空）

    # 【什么】调用成功后记账
    # 【作用】只记真正发生的调用；被拦截的不消耗配额
    # 【为什么】记账放在 after_call：先拦后扣的顺序保证"被拒的尝试"不会把配额刷光
    def consume(self, tool_name: str, tokens: int = 0) -> None:
        self.total_calls += 1                                # 总次数 +1
        self.per_tool[tool_name] = self.per_tool.get(tool_name, 0) + 1  # 该工具次数 +1
        self.tokens += max(0, int(tokens or 0))              # token 累加；负数/None 钳到 0

    # 【什么】记录一次被预算拦截
    # 【作用】观测用：blocked 高说明配额偏紧或模型在打转
    def record_block(self) -> None:
        self.blocked += 1                                    # 拦截计数 +1

    # 【什么】外部补记 token（如 LLM 用量）
    # 【作用】工具结果只是上下文的一部分，LLM 侧也能往同一本账上记
    # 【为什么】token 预算管的是"回灌上下文的总量"，LLM 与工具结果都要算进同一本账
    def add_tokens(self, tokens: int) -> None:
        self.tokens += max(0, int(tokens or 0))              # token 累加；负数/None 钳到 0

    # 【什么】预算快照
    # 【作用】日志与 /health 观测用，看还剩多少
    # 【为什么】返回纯 dict：可直接 JSON 序列化进审计 detail 与监控接口
    def snapshot(self) -> dict:
        return {
            "session_id": self.session_id,                   # 账本归属会话
            "total_calls": self.total_calls,                 # 已调用总次数
            "max_calls_per_run": self.spec.max_calls_per_run,  # 总次数上限
            "per_tool": dict(self.per_tool),                 # 各工具已调用次数（拷贝防外部篡改）
            "tokens": self.tokens,                           # 已耗 token
            "token_budget": self.spec.token_budget,          # token 上限
            "remaining_seconds": round(self.remaining_seconds(), 2),  # 剩余秒数（保留两位小数）
            "blocked": self.blocked,                         # 被拦截次数
        }


# 【什么】按会话管理预算账本
# 【作用】会话开始时建账、结束时销账；并发安全
# 【为什么】多会话并发时建账/销账要加锁，否则两个线程同时 get 会同建两本账丢一本
class BudgetManager:
    """会话级预算账本管理器。

    【为什么不放在 AgentState 里】
        预算要跨"工具调用 / LLM 调用 / 循环判定"多处共享，
        放 state 里得改 TypedDict + dataclass + to_dict/from_dict + 图节点回传四处，
        漏一处就静默丢失；挂在 manager 上按 session_id 取，改动面最小。
    """

    # 【什么】构造函数
    # 【作用】持有默认配额表 + 会话账本字典 + 一把互斥锁
    def __init__(self, spec: Optional[BudgetSpec] = None):
        self.spec = spec or BudgetSpec()                     # 默认配额表；begin/get 没传 spec 时用它
        self._budgets: Dict[str, RunBudget] = {}             # 会话 id → 账本
        self._lock = threading.Lock()                        # 互斥锁：保护 _budgets 的读写

    # 【什么】开一次运行的账
    # 【作用】同 session 重复 begin 会重置（新一轮提问 = 新预算）
    # 【为什么】覆盖旧账是刻意的：新一轮提问不该继承上一轮的已用配额
    def begin(self, session_id: str, spec: Optional[BudgetSpec] = None) -> RunBudget:
        with self._lock:                                     # 加锁：防并发 begin/get 交错
            budget = RunBudget(session_id, spec or self.spec)  # 建账；没传 spec 用默认
            self._budgets[session_id] = budget               # 覆盖写入（重复 begin = 重置）
            return budget                                    # 返回新账本

    # 【什么】取账本；没有就按默认 spec 开一本
    # 【作用】调用方不必关心 begin 是否被调用过
    # 【为什么】懒建账保证"忘了 begin"的会话也能正常被预算约束，而不是抛 KeyError
    def get(self, session_id: str) -> RunBudget:
        with self._lock:                                     # 加锁：防两个线程同时给同一会话建两本账
            budget = self._budgets.get(session_id)           # 先查有没有
            if budget is None:                               # 没 begin 过
                budget = RunBudget(session_id, self.spec)    # 按默认 spec 现开一本
                self._budgets[session_id] = budget           # 存入字典
            return budget                                    # 返回账本

    # 【什么】销账
    # 【作用】会话结束释放内存；不销也不会出错，只是账本会堆积
    # 【为什么】pop 带默认值：重复销账/销不存在的账都不报错
    def end(self, session_id: str) -> None:
        with self._lock:                                     # 加锁：与 begin/get 互斥
            self._budgets.pop(session_id, None)              # 移除账本；不存在则无事发生

    # 【什么】全部账本快照
    # 【作用】观测接口用；并发会话多时可看谁的配额快打满
    # 【为什么】在锁内一次性快照，避免遍历过程中账本被 begin/end 改动
    def stats(self) -> dict:
        with self._lock:                                     # 加锁：遍历时防账本表被并发改动
            return {sid: b.snapshot() for sid, b in self._budgets.items()}  # 会话 id → 该账本的快照
