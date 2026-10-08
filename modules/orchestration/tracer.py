"""
统一观测链路（Tracer）
======================
自建的 span 树追踪：每一次运行一条 trace，每个环节一个 span，
span 之间靠 parent_id 串成树，记录耗时、输入输出、状态与错误。

【为什么要有自建 span 树】
    LangSmith 那一类是外部平台，特点是要联网、要 Key、按量计费；
    它解决的是"事后看大盘"，解决不了三件本地刚需：
        1. 离线/内网环境也要能追溯（CI 里跑完立刻断言 span 结构）
        2. 多 Agent 的"谁在等谁"是自定义结构，平台不认识这些节点
        3. 编排器要按 span 结果做决策（例如某 span 超时就降级）
    因此自建 span 树，同时保留导出成 JSON 的能力，需要时再上报外部平台。

【一条 trace 长什么样】
    run.multiagent (root)
    ├─ plan              主管拆活
    ├─ dispatch.batch_1
    │  ├─ agent.researcher_1   耗时 480ms  status=ok
    │  └─ agent.analyst_1      耗时 502ms  status=ok
    ├─ consensus         相似度 0.21 → 冲突
    ├─ arbitration
    └─ synthesize
    线上出问题，按 trace_id 把整棵树拉出来，哪一环慢、哪一环错，一眼可见。
"""
from __future__ import annotations  # 注解延迟求值：允许类型标注前置引用尚未定义的名字（如返回 "Span"）

import json  # 导出 JSON：to_json 用
import threading  # 锁：span 树是多线程共享的
import time  # 计时：span 上下文管理器算墙钟用
from contextlib import contextmanager  # 上下文管理器装饰器：span 的 with 写法靠它
from enum import Enum  # 枚举基类：SpanStatus 用
from typing import Any, Iterator, Optional  # 类型标注：任意值/迭代器/可选

from pydantic import BaseModel, Field  # 数据模型基类与字段声明：Span/Trace 都是 pydantic 模型

from modules.common.logger import get_logger  # 统一日志入口
from modules.multiagent.models import new_id, now_ms  # id 生成器与毫秒时钟

_logger = get_logger("tracer")  # 本模块日志器：名字即模块名，日志可溯源


# ---------------------------------------------------------------------------
# 【什么】span 状态枚举
# 【作用】一个环节当前处于什么阶段：还在跑 / 正常完成 / 出错 / 超时
# 【为什么继承 str】序列化成 JSON 时直接变成字符串，不用额外转换
# ---------------------------------------------------------------------------
class SpanStatus(str, Enum):  # str  mixin：span.status == "ok" 也能判
    RUNNING = "running"  # 还在跑：end_ms 为 0，duration 未结算
    OK = "ok"  # 正常完成
    ERROR = "error"  # 出错：error 字段必有值
    TIMEOUT = "timeout"  # 超时或被负载控制拦下（编排器把限流/配额也记成这个）


# ---------------------------------------------------------------------------
# 【什么】一个观测单元（span）
# 【作用】链路里的一个环节：一次 LLM 调用、一条消息收发、一轮共识，都是一个 span
# 【为什么用 pydantic】字段类型/默认值声明式定义，model_dump 直接序列化
# ---------------------------------------------------------------------------
class Span(BaseModel):  # 一个观测单元（对应链路里的一个环节）
    """一个观测单元（对应链路里的一个环节）。"""

    span_id: str = Field(default_factory=lambda: new_id("span"))  # 本 span 的唯一 id：默认自动生成
    parent_id: str = Field(default="", description="父 span id；root 为空")  # 树边：靠它把 span 串成树
    trace_id: str = Field(default="")  # 所属链路 id：同一条 trace 下所有 span 同源
    name: str = Field(description="环节名，如 plan / agent.researcher_1")  # 人读的名字：树形摘要直接显示它
    kind: str = Field(default="", description="环节类型，便于按类聚合")  # 机器读的类别：监控按它聚合
    start_ms: int = Field(default_factory=now_ms)  # 开始时刻（毫秒）：默认取创建时刻
    end_ms: int = Field(default=0, description="结束时刻；RUNNING 时为 0")  # 结束时刻：0 = 还没结束
    duration_ms: int = Field(default=0)  # 耗时：end_span 时结算
    status: SpanStatus = Field(default=SpanStatus.RUNNING)  # 状态：出生即 RUNNING
    # 【为什么输入输出各限 2000 字】
    #   span 是要进结果体的，正文全量塞进去会让 RunResult 膨胀到几 MB，
    #   API 响应直接超时。排查时先看摘要，需要全文再按 run_id 拉原始消息。
    inputs: dict[str, Any] = Field(default_factory=dict)  # 输入摘要：调用参数/消息摘要
    outputs: dict[str, Any] = Field(default_factory=dict)  # 输出摘要：返回结果/裁决结果
    error: str = Field(default="")  # 错误信息：ERROR 状态时必有值
    attributes: dict[str, Any] = Field(default_factory=dict)  # 自由属性：调用方想额外挂什么就挂这里
    children: list[str] = Field(default_factory=list, description="子 span id 列表，按开始顺序")  # 树边（反向）：靠它从父找到子

    # 【什么】只读属性：这个 span 是否已结束
    # 【作用】finish 收口时靠它找"还在 RUNNING 的 span"
    @property
    def done(self) -> bool:  # True = 已结束（OK/ERROR/TIMEOUT 任一）
        return self.status != SpanStatus.RUNNING  # 只要不是 RUNNING 就算结束

    # 【什么】补记输入输出
    # 【作用】span 跑着跑着才拿到的数据，用这个方法补进去
    # 【为什么返回 self】支持链式：span.set_io(...).set_io(...)
    def set_io(self, inputs: Optional[dict] = None, outputs: Optional[dict] = None) -> "Span":
        """补记输入输出（在 span 结束前调用，结束后补记不会重算耗时）。"""
        if inputs:  # 传了才合并：None 与空 dict 都不动
            self.inputs.update(inputs)  # 合并而不是覆盖：多次补记累积
        if outputs:  # 同上
            self.outputs.update(outputs)  # 合并而不是覆盖
        return self  # 返回自身：支持链式调用


# ---------------------------------------------------------------------------
# 【什么】一次运行的完整链路（trace）
# 【作用】一条 trace 装一次运行的全部 span：spans 是 {span_id: Span} 的平铺表，
#        root_id 指向树根，children 负责把平铺表还原成树
# 【为什么平铺存而不是嵌套存】span 是并发写的，平铺 dict 好加锁、好按 id 直查
# ---------------------------------------------------------------------------
class Trace(BaseModel):  # 一次运行的完整链路
    """一次运行的完整链路。"""

    trace_id: str = Field(default_factory=lambda: new_id("trace"))  # 本链路的唯一 id：默认自动生成
    name: str = Field(default="")  # 链路名：如 run.multiagent
    run_id: str = Field(default="", description="关联的 RunResult.run_id")  # 关联运行 id：trace 与结果体对上号
    started_ms: int = Field(default_factory=now_ms)  # 链路开始时刻（毫秒）
    ended_ms: int = Field(default=0)  # 链路结束时刻：0 = 还没结束
    root_id: str = Field(default="", description="根 span id")  # 树根：to_tree/summary 从它开始遍历
    spans: dict[str, Span] = Field(default_factory=dict, description="{span_id: Span}")  # 平铺表：所有 span 都在这

    # 【什么】只读属性：链路总耗时
    # 【作用】没结束的 trace 用"现在 - 开始"算实时耗时；结束了用"结束 - 开始"
    @property
    def duration_ms(self) -> int:  # 链路总耗时（毫秒）
        return max(0, (self.ended_ms or now_ms()) - self.started_ms)  # max(0,·)：防时钟回拨出负数

    # 【什么】只读属性：根 span
    # 【作用】给外部直接拿树根，不用自己写 spans.get(root_id)
    @property
    def root(self) -> Optional[Span]:  # 根 span；还没开任何 span 时为 None
        return self.spans.get(self.root_id)  # 平铺表里按 root_id 直查

    # 【什么】导出嵌套 JSON 树
    # 【作用】前端画火焰图/树图需要嵌套结构；平铺表在这里递归拼成树
    def to_tree(self, span_id: str = "") -> dict:  # 从指定节点（默认根）开始递归导出
        """导出成嵌套 JSON 结构（便于前端画火焰图）。"""
        node = self.spans.get(span_id or self.root_id)  # 取起点：没传 span_id 就从根开始
        if node is None:  # 起点不存在（比如还没开任何 span）：返回空树
            return {}
        out = node.model_dump()  # 先把本节点压成字典
        out["children"] = [self.to_tree(cid) for cid in node.children]  # 再递归把每个子节点也压成嵌套字典
        return out  # 返回嵌套结构

    # 【什么】导出整条链路的摘要字典
    # 【作用】to_json / 上报外部平台用：带统计信息 + 嵌套树
    def to_dict(self) -> dict:  # 链路摘要 + 嵌套树
        return {
            "trace_id": self.trace_id,  # 链路 id
            "run_id": self.run_id,  # 关联运行 id
            "name": self.name,  # 链路名
            "started_ms": self.started_ms,  # 开始时刻
            "ended_ms": self.ended_ms,  # 结束时刻（0=未结束）
            "duration_ms": self.duration_ms,  # 总耗时
            "span_count": len(self.spans),  # span 总数：一眼看出这次运行干了多少活
            "tree": self.to_tree(),  # 嵌套树：从根开始的完整结构
        }

    # 【什么】导出 JSON 字符串
    # 【作用】落盘/打印/HTTP 响应直接用；default=str 兜住 datetime 等不可序列化类型
    def to_json(self, indent: int = 2) -> str:  # indent=2：人读友好；要紧凑传 None
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent,
                          default=str)  # ensure_ascii=False：中文不转义；default=str：不可序列化对象转成字符串

    # ------------------------------------------------------------------
    # 【为什么统计要遍历 span 而不是在写入时累加】
    #   并发写累加要加锁且容易漏；遍历是幂等的，
    #   跑几次结果都一样，适合进 CI 断言。
    # ------------------------------------------------------------------
    # 【什么】按状态与类型聚合的统计
    # 【作用】给监控看大盘：多少 ok / 多少 error、哪类环节最耗时
    def stats(self) -> dict[str, Any]:  # 聚合统计
        """按状态与类型聚合，给监控看大盘。"""
        by_status: dict[str, int] = {}  # {状态: 数量}
        by_kind: dict[str, dict[str, Any]] = {}  # {类别: {count, total_ms, max_ms}}
        for sp in self.spans.values():  # 遍历所有 span
            by_status[sp.status.value] = by_status.get(sp.status.value, 0) + 1  # 状态计数 +1
            slot = by_kind.setdefault(sp.kind or "unknown", {"count": 0, "total_ms": 0, "max_ms": 0})  # 取（或建）该类别的聚合槽
            slot["count"] += 1  # 该类别数量 +1
            slot["total_ms"] += sp.duration_ms  # 该类别总耗时累加
            slot["max_ms"] = max(slot["max_ms"], sp.duration_ms)  # 该类别最大耗时刷新
        return {"by_status": by_status, "by_kind": by_kind,  # 两维聚合结果
                "span_count": len(self.spans), "duration_ms": self.duration_ms}  # 附总量与总耗时

    # 【什么】树形文本摘要
    # 【作用】终端排查时直接 print 这个：一眼看出哪一环慢、哪一环错
    def summary(self, limit: int = 400) -> str:  # limit 控制输出字段的截断长度
        """树形文本摘要：终端排查时直接 print 这个。"""
        lines = [f"trace {self.trace_id} [{self.name}] {self.duration_ms}ms "
                 f"spans={len(self.spans)}"]  # 首行：链路 id + 名字 + 总耗时 + span 总数

        # 【什么】递归打印一个节点及其子树（内部函数：闭包共享 lines）
        # 【作用】prefix/is_last 控制树枝字符（├─/└─/│ ），打出树形缩进
        def walk(sid: str, prefix: str = "", is_last: bool = True) -> None:  # sid=当前节点 id
            sp = self.spans.get(sid)  # 取当前节点
            if sp is None:  # 节点不存在（比如 children 里记了已被清的 id）：跳过
                return
            branch = ("└─ " if is_last else "├─ ") if prefix else ""  # 树枝：是最后一个子节点用 └─，否则 ├─；根节点不画树枝
            tail = ""  # 行尾附加信息：错误或输出摘要
            if sp.error:  # 有错误优先显示错误
                tail = f" error={sp.error[:60]}"  # 错误截 60 字：防一行爆长
            elif sp.outputs:  # 没错误但有输出：显示前 3 个输出字段
                shown = ", ".join(f"{k}={str(v)[:limit // 8]}" for k, v in list(sp.outputs.items())[:3])  # 每个值再截短
                tail = f" {shown}"
            lines.append(f"{prefix}{branch}{sp.name} [{sp.status.value}] {sp.duration_ms}ms{tail}")  # 本行：树枝+名字+状态+耗时+附加
            child_prefix = prefix + ("   " if is_last else "│  ")  # 子节点缩进：父是最后一个就空格，否则画竖线
            kids = sp.children  # 子节点 id 列表
            for i, cid in enumerate(kids):  # 逐个递归
                walk(cid, child_prefix, i == len(kids) - 1)  # 最后一个子节点标记 is_last

        if self.root_id:  # 有根才开始遍历；没根（还没开 span）只有首行
            walk(self.root_id)  # 从根开始递归
        return "\n".join(lines)  # 拼成多行文本


# ---------------------------------------------------------------------------
# 【什么】span 树的写入器
# 【作用】开 trace、开/关 span、上下文管理器写法、运行结束收口
# 【为什么线程安全】批次是线程池并行跑的，多个 Agent 同时开 span、写 children
# ---------------------------------------------------------------------------
class Tracer:  # span 树的写入器
    """span 树的写入器。线程安全：可安全用于并行执行的批次。"""

    # 【什么】Tracer 构造：一把可重入锁 + 一张 trace 登记表
    # 【作用】enabled=False 时所有写法变成"写了但不进树"，关追踪不需要改调用点
    def __init__(self, enabled: bool = True):  # 默认开追踪
        self.enabled = enabled  # 总开关：关掉后 span() 返回游离 span，不进树
        # 【为什么需要锁】
        #   批次是线程池并行跑的，多个 Agent 同时开 span、写 children。
        #   不加锁会出现字典并发写与"父 span 的 children 少记一个"，
        #   表现为线上偶尔少一条 span，且无法复现。
        self._lock = threading.RLock()  # 可重入锁：start_span 里调别的加锁方法不会死锁
        self.traces: dict[str, Trace] = {}  # {trace_id: Trace}：一次运行一条

    # 【什么】开一条新链路
    # 【作用】一次运行开始时调；返回的 Trace 之后所有 span 都挂它下面
    def new_trace(self, name: str, run_id: str = "") -> Trace:  # name=链路名；run_id=关联运行 id
        trace = Trace(name=name, run_id=run_id)  # 建链路对象：trace_id 自动生成
        with self._lock:  # 拿锁：登记表是共享 dict
            self.traces[trace.trace_id] = trace  # 登记：之后可按 trace_id 取回
        return trace  # 返回给调用方：之后 start_span 都要带它

    # 【什么】按 id 取回链路
    # 【作用】事后查链路：API 层按 trace_id 拉树用
    def get(self, trace_id: str) -> Optional[Trace]:  # 查不到返回 None
        return self.traces.get(trace_id)  # 直查登记表

    # 【什么】开一个 span
    # 【作用】登记进平铺表；parent_id 非空则挂到父的 children，空则（若是第一个）当根
    def start_span(self, trace: Trace, name: str, kind: str = "", parent_id: str = "",
                   **attributes) -> Span:  # attributes 进 span.attributes：自由属性
        """开一个 span；parent_id 非空时自动挂到父节点的 children 上。"""
        span = Span(trace_id=trace.trace_id, name=name, kind=kind,
                    parent_id=parent_id, attributes=attributes)  # 建 span：开始时刻默认取现在
        with self._lock:  # 拿锁：平铺表与父的 children 都要原子写
            trace.spans[span.span_id] = span  # 登记进平铺表
            if parent_id and parent_id in trace.spans:  # 有父且父存在：挂到父的 children
                trace.spans[parent_id].children.append(span.span_id)  # 父节点记一条"我有这个子"
            elif not parent_id and not trace.root_id:  # 没父且还没根：这个 span 就是根
                trace.root_id = span.span_id  # 登记根
        return span  # 返回给调用方：之后 set_io / end_span 都用它

    # 【什么】结束一个 span
    # 【作用】结算耗时、定状态、补输出与错误
    def end_span(self, span: Span, status: SpanStatus = SpanStatus.OK,
                 outputs: Optional[dict] = None, error: str = "") -> Span:  # 默认正常结束
        span.end_ms = now_ms()  # 结束时刻：现在
        span.duration_ms = max(0, span.end_ms - span.start_ms)  # 结算耗时：max(0,·) 防时钟回拨出负数
        span.status = status  # 定状态：OK/ERROR/TIMEOUT
        if outputs:  # 传了输出才合并
            span.outputs.update(outputs)  # 合并而不是覆盖
        if error:  # 传了错误才写
            span.error = error  # 错误信息
        return span  # 返回自身：方便链式或断言

    # 【什么】span 的上下文管理器写法
    # 【作用】with tracer.span(...) as sp: 正常退出记 OK，抛异常记 ERROR 并原样向上抛
    @contextmanager  # 生成器即上下文管理器：yield 前是 __enter__，yield 后是 __exit__
    def span(self, trace: Trace, name: str, kind: str = "", parent_id: str = "",
             **attributes) -> Iterator[Span]:  # 用法与 start_span 相同，但自动收尾
        """上下文管理器写法：正常退出记 OK，抛异常记 ERROR 并原样向上抛。

        【为什么异常要原样抛出而不是吞掉】
            观测层的职责是"记录"，不是"兜底"。吞掉异常会让上层的降级逻辑
            永远不触发，表现为"trace 里一片红，但流程一路走到成功"。
        """
        if not self.enabled:
            # 关闭时返回一个游离 span：调用方照常写 inputs/outputs，不报错，
            # 只是不进树。这样"关追踪"不需要改任何调用点。
            yield Span(trace_id=trace.trace_id, name=name, kind=kind)  # 游离 span：不进平铺表、不挂树
            return

        sp = self.start_span(trace, name, kind=kind, parent_id=parent_id, **attributes)  # 正常开 span
        started = time.time()  # 墙钟起点：目前仅用于调试时对照（耗时以 end_span 的 now_ms 为准）
        try:
            yield sp  # 把 span 交给 with 块：块内正常跑
        except Exception as e:  # 块内抛异常：记 ERROR 后原样向上抛
            self.end_span(sp, SpanStatus.ERROR, error=f"{type(e).__name__}: {e}")  # 错误带异常类型名：排查时先看类型再看消息
            raise  # 原样向上抛：观测层不兜底
        else:  # 块内正常退出
            self.end_span(sp, SpanStatus.OK)  # 记 OK

    # 【什么】运行结束收口
    # 【作用】给还在 RUNNING 的 span 强制收尾为 TIMEOUT：区分"卡住了"和"忘了关"
    def finish(self, trace: Trace) -> Trace:  # 返回同一条 trace：方便链式
        trace.ended_ms = now_ms()  # 记链路结束时刻
        # 【退出时把还在 RUNNING 的 span 收口】
        #   运行被打断（超时/abort）时有些 span 没走 end_span，
        #   不收口的话导出的树里永远挂着 running，看不出是"卡住了"还是"忘了关"。
        for sp in trace.spans.values():  # 遍历所有 span
            if not sp.done:  # 还没结束的
                self.end_span(sp, SpanStatus.TIMEOUT, error="trace 结束仍未关闭")  # 强制收尾为 TIMEOUT
        return trace  # 返回链路：之后 to_dict/to_json/summary 用
