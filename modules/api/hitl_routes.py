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
from __future__ import annotations  # 允许在类型标注中使用字符串形式的前向引用

import asyncio  # 异步休眠：SSE 空转时让出事件循环
import json  # 序列化：SSE 事件帧的 data 字段
import queue  # 队列异常：notifier 订阅返回的 Queue 取空时抛 queue.Empty
from typing import Optional  # 类型标注：可选参数

from fastapi import Depends, FastAPI, HTTPException  # FastAPI 核心：依赖注入/应用/异常
from fastapi.responses import StreamingResponse  # SSE 流式响应
from pydantic import BaseModel, Field  # 请求体模型与字段校验

from modules.common.logger import get_logger  # 统一日志器工厂
from modules.hitl.models import HumanAction  # 人工决策动作枚举（approve/modify/reject/abort）

_logger = get_logger("api.hitl")  # 本模块日志器，名 "api.hitl"


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
def _pending_item(r) -> dict:  # 内部辅助函数：请求对象 → 前端卡片字典
    return {  # 字段映射：把内部模型转成前端直接渲染的格式
        "id": r.id, "kind": r.kind.value, "title": r.title,  # 基础标识：id/类型/标题
        "reason": r.reason, "agent": r.agent,  # 触发原因与来源 Agent
        "context": r.context, "options": r.options,  # 上下文与可选动作
        "default_action": r.default_action.value,  # 超时默认动作
        "created_at": r.created_at,  # 创建时间
        "wait_seconds": r.timeout_seconds,  # 已等待时长（前端据此显示倒计时）
    }


# 【什么】拼一条 SSE 帧
# 【作用】event 行让前端能按类型分流；ensure_ascii=False 保证中文不被转义
def _sse(event: str, data: dict) -> bytes:  # SSE 帧构造函数
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")  # event/data 两行 + 空行分隔，UTF-8 编码


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
    if notifier is None:  # 外部没传 notifier：尝试取进程级共享实例
        try:
            from modules.hitl import get_global_notifier  # 延迟导入：避免循环依赖
            notifier = get_global_notifier()  # 取全局唯一通知器
        except Exception as e:  # 通知器装配失败（如 HITL 未启用）
            _logger.warning(f"HITL 通知器不可用，/hitl/stream 将返回 503: {e}")  # 记录降级原因
            notifier = None  # 保持 None，后续 /hitl/stream 会返回 503

    # 【什么】看待办：审批页轮询它
    # 【作用】只返回 WAITING 的；已处理的不出现，避免页面上挂着旧卡片
    @app.get("/hitl/pending")
    async def hitl_pending(limit: int = 50, user: dict = Depends(auth_dependency)):
        """当前等待人工拍板的请求列表。"""
        if gate is None:  # HITL 未启用：返回 503 明确告知
            raise HTTPException(status_code=503, detail="HITL 未启用（HITL_ENABLED=false）")  # 503 服务不可用
        items = gate.pending(limit)  # 从 gate 取当前等待中的请求列表
        return {"count": len(items), "items": [_pending_item(r) for r in items]}  # 转成前端卡片格式返回

    # 【什么】SSE 长连接：一有待办立刻推一条
    # 【作用】省掉审批页的轮询；连上瞬间先补发当前待办，
    #        避免"连接之前就抛出的请求"永远看不到。
    #   【浏览器 EventSource 的坑】它发不出自定义请求头，
    #     而本项目鉴权走 X-API-Key。网页端要用 fetch + ReadableStream，
    #     或由网关把 key 注入请求头；否则这里会直接 401/403。
    @app.get("/hitl/stream")
    async def hitl_stream(user: dict = Depends(auth_dependency)):
        """中断请求实时推送（text/event-stream）。"""
        if gate is None:  # HITL 未启用：返回 503
            raise HTTPException(status_code=503, detail="HITL 未启用（HITL_ENABLED=false）")  # 503 服务不可用
        if notifier is None:  # 通知器不可用：返回 503
            raise HTTPException(status_code=503, detail="HITL 通知器不可用")  # 503 服务不可用

        q = notifier.subscribe()  # 订阅通知器：拿到一个专属队列，新中断会进这个队列
        backlog = [_pending_item(r) for r in gate.pending(limit=50)]  # 先取当前积压的待办，连上瞬间补发

        async def _gen():  # SSE 事件生成器
            try:
                for item in backlog:  # 先补发积压待办：避免"连接之前就抛出的请求"永远看不到
                    yield _sse("interrupt", item)  # 逐条推送给前端
                while True:  # 主循环：等新中断
                    try:
                        req = q.get_nowait()  # 非阻塞取队列：有就拿出来
                    except queue.Empty:  # 队列空：没新中断
                        await asyncio.sleep(0.5)          # 空转也让出事件循环，别空转烧 CPU
                        continue  # 继续轮询
                    yield _sse("interrupt", _pending_item(req))  # 有新中断：转成卡片格式推给前端
            finally:
                notifier.unsubscribe(q)                    # 客户端断开必须摘掉，否则队列泄漏

        return StreamingResponse(  # 返回 SSE 流式响应
            _gen(),  # 事件生成器
            media_type="text/event-stream",  # SSE 固定媒体类型
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # 禁缓存+关 Nginx 缓冲
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
        if gate is None:  # HITL 未启用：返回 503
            raise HTTPException(status_code=503, detail="HITL 未启用（HITL_ENABLED=false）")  # 503 服务不可用
        try:
            action = HumanAction(req.action.strip().lower())  # 字符串转枚举：strip+lower 容错大小写与空白
        except ValueError:  # 非法动作值：枚举转换失败
            raise HTTPException(
                status_code=422,
                detail=f"action 必须是 {[a.value for a in HumanAction]} 之一",
            )  # 422 参数校验失败
        operator = str(user.get("username") or user.get("sub") or user.get("user_id") or "")  # 操作者标识：多字段兜底
        ok = gate.submit(  # 提交决策到 gate
            req.request_id, action, edited=req.modified,  # 请求id/动作/修改内容
            operator=operator, note=req.note,  # 操作者与备注（进审计）
        )
        if not ok:  # 提交失败：请求不存在或已处理
            raise HTTPException(
                status_code=409,
                detail="请求不存在或已处理（可能重复提交，或已超时按默认动作兜底）",
            )  # 409 冲突：明确告知调用方去查最新状态
        _logger.info(f"人工决策已生效: {req.request_id} {action.value} by {operator or '-'}")  # 成功日志
        return {"ok": True, "request_id": req.request_id, "action": action.value}  # 返回确认信息

    # 【什么】看留痕
    # 【作用】审计用：谁在什么时候被为什么打断、最终按什么动作执行
    @app.get("/hitl/records")
    async def hitl_records(limit: int = 50, user: dict = Depends(auth_dependency)):
        """历史人工介入留痕。"""
        if gate is None:  # HITL 未启用：返回 503
            raise HTTPException(status_code=503, detail="HITL 未启用（HITL_ENABLED=false）")  # 503 服务不可用
        records = gate.records()  # 从 gate 取全部历史记录
        records = records[-max(1, int(limit)):]  # 只取最后 limit 条；max(1,...) 防止 limit=0 时取空
        return {  # 返回留痕列表
            "count": len(records),  # 记录总数
            "items": [  # 逐条转成前端展示格式
                {
                    "request_id": r.request.id,  # 请求标识
                    "kind": r.request.kind.value,  # 中断类型
                    "title": r.request.title,  # 标题
                    "status": r.request.status.value,  # 当前状态
                    "final_action": r.final_action.value,  # 最终执行动作
                    "operator": (r.reply.operator if r.reply else "system:timeout"),  # 操作者；无回复记系统超时
                    "note": (r.reply.note if r.reply else ""),  # 备注；无回复留空
                    "created_at": r.request.created_at,  # 创建时间
                    "resolved_at": r.resolved_at,  # 解决时间
                }
                for r in records  # 遍历记录列表
            ],
        }
