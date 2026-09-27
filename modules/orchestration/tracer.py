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
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Iterator, Optional

from pydantic import BaseModel, Field

from modules.common.logger import get_logger
from modules.multiagent.models import new_id, now_ms

_logger = get_logger("tracer")


class SpanStatus(str, Enum):
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


class Span(BaseModel):
    """一个观测单元（对应链路里的一个环节）。"""

    span_id: str = Field(default_factory=lambda: new_id("span"))
    parent_id: str = Field(default="", description="父 span id；root 为空")
    trace_id: str = Field(default="")
    name: str = Field(description="环节名，如 plan / agent.researcher_1")
    kind: str = Field(default="", description="环节类型，便于按类聚合")
    start_ms: int = Field(default_factory=now_ms)
    end_ms: int = Field(default=0, description="结束时刻；RUNNING 时为 0")
    duration_ms: int = Field(default=0)
    status: SpanStatus = Field(default=SpanStatus.RUNNING)
    # 【为什么输入输出各限 2000 字】
    #   span 是要进结果体的，正文全量塞进去会让 RunResult 膨胀到几 MB，
    #   API 响应直接超时。排查时先看摘要，需要全文再按 run_id 拉原始消息。
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str = Field(default="")
    attributes: dict[str, Any] = Field(default_factory=dict)
    children: list[str] = Field(default_factory=list, description="子 span id 列表，按开始顺序")

    @property
    def done(self) -> bool:
        return self.status != SpanStatus.RUNNING

    def set_io(self, inputs: Optional[dict] = None, outputs: Optional[dict] = None) -> "Span":
        """补记输入输出（在 span 结束前调用，结束后补记不会重算耗时）。"""
        if inputs:
            self.inputs.update(inputs)
        if outputs:
            self.outputs.update(outputs)
        return self


class Trace(BaseModel):
    """一次运行的完整链路。"""

    trace_id: str = Field(default_factory=lambda: new_id("trace"))
    name: str = Field(default="")
    run_id: str = Field(default="", description="关联的 RunResult.run_id")
    started_ms: int = Field(default_factory=now_ms)
    ended_ms: int = Field(default=0)
    root_id: str = Field(default="", description="根 span id")
    spans: dict[str, Span] = Field(default_factory=dict, description="{span_id: Span}")

    @property
    def duration_ms(self) -> int:
        return max(0, (self.ended_ms or now_ms()) - self.started_ms)

    @property
    def root(self) -> Optional[Span]:
        return self.spans.get(self.root_id)

    def to_tree(self, span_id: str = "") -> dict:
        """导出成嵌套 JSON 结构（便于前端画火焰图）。"""
        node = self.spans.get(span_id or self.root_id)
        if node is None:
            return {}
        out = node.model_dump()
        out["children"] = [self.to_tree(cid) for cid in node.children]
        return out

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "name": self.name,
            "started_ms": self.started_ms,
            "ended_ms": self.ended_ms,
            "duration_ms": self.duration_ms,
            "span_count": len(self.spans),
            "tree": self.to_tree(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent,
                          default=str)

    # ------------------------------------------------------------------
    # 【为什么统计要遍历 span 而不是在写入时累加】
    #   并发写累加要加锁且容易漏；遍历是幂等的，
    #   跑几次结果都一样，适合进 CI 断言。
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """按状态与类型聚合，给监控看大盘。"""
        by_status: dict[str, int] = {}
        by_kind: dict[str, dict[str, Any]] = {}
        for sp in self.spans.values():
            by_status[sp.status.value] = by_status.get(sp.status.value, 0) + 1
            slot = by_kind.setdefault(sp.kind or "unknown", {"count": 0, "total_ms": 0, "max_ms": 0})
            slot["count"] += 1
            slot["total_ms"] += sp.duration_ms
            slot["max_ms"] = max(slot["max_ms"], sp.duration_ms)
        return {"by_status": by_status, "by_kind": by_kind,
                "span_count": len(self.spans), "duration_ms": self.duration_ms}

    def summary(self, limit: int = 400) -> str:
        """树形文本摘要：终端排查时直接 print 这个。"""
        lines = [f"trace {self.trace_id} [{self.name}] {self.duration_ms}ms "
                 f"spans={len(self.spans)}"]

        def walk(sid: str, prefix: str = "", is_last: bool = True) -> None:
            sp = self.spans.get(sid)
            if sp is None:
                return
            branch = ("└─ " if is_last else "├─ ") if prefix else ""
            tail = ""
            if sp.error:
                tail = f" error={sp.error[:60]}"
            elif sp.outputs:
                shown = ", ".join(f"{k}={str(v)[:limit // 8]}" for k, v in list(sp.outputs.items())[:3])
                tail = f" {shown}"
            lines.append(f"{prefix}{branch}{sp.name} [{sp.status.value}] {sp.duration_ms}ms{tail}")
            child_prefix = prefix + ("   " if is_last else "│  ")
            kids = sp.children
            for i, cid in enumerate(kids):
                walk(cid, child_prefix, i == len(kids) - 1)

        if self.root_id:
            walk(self.root_id)
        return "\n".join(lines)


class Tracer:
    """span 树的写入器。线程安全：可安全用于并行执行的批次。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        # 【为什么需要锁】
        #   批次是线程池并行跑的，多个 Agent 同时开 span、写 children。
        #   不加锁会出现字典并发写与"父 span 的 children 少记一个"，
        #   表现为线上偶尔少一条 span，且无法复现。
        self._lock = threading.RLock()
        self.traces: dict[str, Trace] = {}

    def new_trace(self, name: str, run_id: str = "") -> Trace:
        trace = Trace(name=name, run_id=run_id)
        with self._lock:
            self.traces[trace.trace_id] = trace
        return trace

    def get(self, trace_id: str) -> Optional[Trace]:
        return self.traces.get(trace_id)

    def start_span(self, trace: Trace, name: str, kind: str = "", parent_id: str = "",
                   **attributes) -> Span:
        """开一个 span；parent_id 非空时自动挂到父节点的 children 上。"""
        span = Span(trace_id=trace.trace_id, name=name, kind=kind,
                    parent_id=parent_id, attributes=attributes)
        with self._lock:
            trace.spans[span.span_id] = span
            if parent_id and parent_id in trace.spans:
                trace.spans[parent_id].children.append(span.span_id)
            elif not parent_id and not trace.root_id:
                trace.root_id = span.span_id
        return span

    def end_span(self, span: Span, status: SpanStatus = SpanStatus.OK,
                 outputs: Optional[dict] = None, error: str = "") -> Span:
        span.end_ms = now_ms()
        span.duration_ms = max(0, span.end_ms - span.start_ms)
        span.status = status
        if outputs:
            span.outputs.update(outputs)
        if error:
            span.error = error
        return span

    @contextmanager
    def span(self, trace: Trace, name: str, kind: str = "", parent_id: str = "",
             **attributes) -> Iterator[Span]:
        """上下文管理器写法：正常退出记 OK，抛异常记 ERROR 并原样向上抛。

        【为什么异常要原样抛出而不是吞掉】
            观测层的职责是"记录"，不是"兜底"。吞掉异常会让上层的降级逻辑
            永远不触发，表现为"trace 里一片红，但流程一路走到成功"。
        """
        if not self.enabled:
            # 关闭时返回一个游离 span：调用方照常写 inputs/outputs，不报错，
            # 只是不进树。这样"关追踪"不需要改任何调用点。
            yield Span(trace_id=trace.trace_id, name=name, kind=kind)
            return

        sp = self.start_span(trace, name, kind=kind, parent_id=parent_id, **attributes)
        started = time.time()
        try:
            yield sp
        except Exception as e:
            self.end_span(sp, SpanStatus.ERROR, error=f"{type(e).__name__}: {e}")
            raise
        else:
            self.end_span(sp, SpanStatus.OK)

    def finish(self, trace: Trace) -> Trace:
        trace.ended_ms = now_ms()
        # 【退出时把还在 RUNNING 的 span 收口】
        #   运行被打断（超时/abort）时有些 span 没走 end_span，
        #   不收口的话导出的树里永远挂着 running，看不出是"卡住了"还是"忘了关"。
        for sp in trace.spans.values():
            if not sp.done:
                self.end_span(sp, SpanStatus.TIMEOUT, error="trace 结束仍未关闭")
        return trace
