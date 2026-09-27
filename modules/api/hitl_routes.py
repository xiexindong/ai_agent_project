"""
Human-in-the-Loop HTTP 端点
============================
给审批页用的三个接口：看待办、提交决策、看留痕。

    GET  /hitl/pending          当前等着人拍板的请求（轮询）
    GET  /hitl/stream           SSE 长连接，一有待办就推一条（免轮询）
    POST /hitl/reply            提交决策（approve / modify / reject / abort）
    GET  /hitl/records          历史介入留痕（审计用）

【为什么必须单独成文件】
    和 pipeline_routes.py 同理：HITL 是一组独立可测的能力，
    单独成文件后既能挂到主 app，也能在测试里
    register_hitl_routes(FastAPI(), gate) 起一个干净的应用。

【为什么 gate 由外部传入而不是在这里 new】
    Agent 链路和审批端点必须用同一个 gate 实例，
    否则人在页面上点批准、链路那边等的却是另一个 store，永远收不到。
    这里只消费传入的 gate，不负责创建 —— 创建归 modules.hitl.get_global_gate。
"""
from __future__ import annotations

import asyncio
import json
import queue
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from modules.common.logger import get_logger
from modules.hitl.models import HumanAction

_logger = get_logger("api.hitl")


# 【什么】提交决策的请求体
# 【作用】action 是必须的；modified 只在 action=modify 时有意义
class HitlReplyRequest(BaseModel):
    """人工决策提交体。"""

    request_id: str = Field(..., description="待处理请求的 id")
    action: str = Field(..., description="approve / modify / reject / abort")
    modified: str = Field(default="", description="modify 时的改后内容（指令/参数/答案）")
    note: str = Field(default="", description="备注理由，进审计")


# 【什么】请求 → 前端卡片 的字段映射
# 【作用】/hitl/pending 与 /hitl/stream 共用同一份，避免两处字段漂移
def _pending_item(r) -> dict:
    return {
        "id": r.id, "kind": r.kind.value, "title": r.title,
        "reason": r.reason, "agent": r.agent,
        "context": r.context, "options": r.options,
        "default_action": r.default_action.value,
        "created_at": r.created_at,
        "wait_seconds": r.timeout_seconds,
    }


# 【什么】拼一条 SSE 帧
# 【作用】event 行让前端能按类型分流；ensure_ascii=False 保证中文不被转义
def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def register_hitl_routes(app: FastAPI, gate, auth_dependency, notifier=None) -> None:
    """把 HITL 端点挂到应用上。

    Args:
        app:              FastAPI 应用
        gate:             HumanGate 实例（与 Agent 链路共用同一个）
        auth_dependency:  鉴权依赖，如 RequirePerm("api:admin:*")
        notifier:         InterruptNotifier 实例（SSE 推送源）；None 时取进程级共享的那个
    """
    # 【为什么 notifier 允许外部传入】和 gate 同理：必须是链路正在用的那个，
    #   否则订阅者挂在另一个通知器上，永远收不到事件（表现为"连上了没反应"）。
    if notifier is None:
        try:
            from modules.hitl import get_global_notifier
            notifier = get_global_notifier()
        except Exception as e:
            _logger.warning(f"HITL 通知器不可用，/hitl/stream 将返回 503: {e}")
            notifier = None

    # 【什么】看待办：审批页轮询它
    # 【作用】只返回 WAITING 的；已处理的不出现，避免页面上挂着旧卡片
    @app.get("/hitl/pending")
    async def hitl_pending(limit: int = 50, user: dict = Depends(auth_dependency)):
        """当前等待人工拍板的请求列表。"""
        if gate is None:
            raise HTTPException(status_code=503, detail="HITL 未启用（HITL_ENABLED=false）")
        items = gate.pending(limit)
        return {"count": len(items), "items": [_pending_item(r) for r in items]}

    # 【什么】SSE 长连接：一有待办立刻推一条
    # 【作用】省掉审批页的轮询；连上瞬间先补发当前待办，
    #        避免"连接之前就抛出的请求"永远看不到。
    #   【浏览器 EventSource 的坑】它发不出自定义请求头，
    #     而本项目鉴权走 X-API-Key。网页端要用 fetch + ReadableStream，
    #     或由网关把 key 注入请求头；否则这里会直接 401/403。
    @app.get("/hitl/stream")
    async def hitl_stream(user: dict = Depends(auth_dependency)):
        """中断请求实时推送（text/event-stream）。"""
        if gate is None:
            raise HTTPException(status_code=503, detail="HITL 未启用（HITL_ENABLED=false）")
        if notifier is None:
            raise HTTPException(status_code=503, detail="HITL 通知器不可用")

        q = notifier.subscribe()
        backlog = [_pending_item(r) for r in gate.pending(limit=50)]

        async def _gen():
            try:
                for item in backlog:
                    yield _sse("interrupt", item)
                while True:
                    try:
                        req = q.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.5)          # 空转也让出事件循环，别空转烧 CPU
                        continue
                    yield _sse("interrupt", _pending_item(req))
            finally:
                notifier.unsubscribe(q)                    # 客户端断开必须摘掉，否则队列泄漏

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 【什么】提交决策
    # 【作用】人点按钮的落点；返回是否生效
    #   【为什么重复提交返回 409 而不是 200】
    #       网络重试可能把同一次点击发两遍。第二遍若被当成成功，
    #       调用方会以为"我改的内容生效了"，而真正生效的是第一次提交的内容 ——
    #       这种静默的不一致在审计上最难查。明确报冲突，让调用方去查最新状态。
    @app.post("/hitl/reply")
    async def hitl_reply(req: HitlReplyRequest, user: dict = Depends(auth_dependency)):
        """提交一次人工决策。"""
        if gate is None:
            raise HTTPException(status_code=503, detail="HITL 未启用（HITL_ENABLED=false）")
        try:
            action = HumanAction(req.action.strip().lower())
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"action 必须是 {[a.value for a in HumanAction]} 之一",
            )
        operator = str(user.get("username") or user.get("sub") or user.get("user_id") or "")
        ok = gate.submit(
            req.request_id, action, edited=req.modified,
            operator=operator, note=req.note,
        )
        if not ok:
            raise HTTPException(
                status_code=409,
                detail="请求不存在或已处理（可能重复提交，或已超时按默认动作兜底）",
            )
        _logger.info(f"人工决策已生效: {req.request_id} {action.value} by {operator or '-'}")
        return {"ok": True, "request_id": req.request_id, "action": action.value}

    # 【什么】看留痕
    # 【作用】审计用：谁在什么时候被为什么打断、最终按什么动作执行
    @app.get("/hitl/records")
    async def hitl_records(limit: int = 50, user: dict = Depends(auth_dependency)):
        """历史人工介入留痕。"""
        if gate is None:
            raise HTTPException(status_code=503, detail="HITL 未启用（HITL_ENABLED=false）")
        records = gate.records()
        records = records[-max(1, int(limit)):]
        return {
            "count": len(records),
            "items": [
                {
                    "request_id": r.request.id,
                    "kind": r.request.kind.value,
                    "title": r.request.title,
                    "status": r.request.status.value,
                    "final_action": r.final_action.value,
                    "operator": (r.reply.operator if r.reply else "system:timeout"),
                    "note": (r.reply.note if r.reply else ""),
                    "created_at": r.request.created_at,
                    "resolved_at": r.resolved_at,
                }
                for r in records
            ],
        }
