"""
请求级上下文：task_id / session_id / user_code
===============================================
双库架构（MySQL 业务库 + PG 会话库）没有外键、没有跨库事务，唯一能把两边数据
串起来的是"业务流水号" task_id。它必须在 HTTP 入口生成一次，之后所有落库动作
（PG 的 chat_history / tool_call_log、MySQL 的 sys_audit_log / sys_llm_usage）
自动带上，而不是靠函数层层透传——透传只要漏一层，链路就断了且没人知道。

【为什么用 ContextVar 而不是全局变量】
    /chat 里 Agent 是 asyncio.to_thread 丢到线程池跑的：
        ContextVar 的值会随 context 复制进子线程（读得到），且并发请求之间互不串号；
        全局变量在并发下会互相覆盖，A 用户的 task_id 会写到 B 用户的审计上。

【输入 → 输出对照】
    set_request_context(task_id="T-20260920-3f9c1a2b", session_id="s-1", user_code="U001")

    get_task_id()     -> "T-20260920-3f9c1a2b"
    get_session_id()  -> "s-1"
    snapshot()        -> {"task_id": "T-20260920-3f9c1a2b",
                          "session_id": "s-1", "user_code": "U001"}

【依赖警告】
    asyncio.to_thread 才会复制 context；如果以后改成裸 ThreadPoolExecutor.submit
    或 loop.run_in_executor 不带 context，子线程读到的会是空串，落库的 task_id 全空。
"""
import uuid
from datetime import datetime
from contextvars import ContextVar

_task_id: ContextVar[str] = ContextVar("task_id", default="")
_session_id: ContextVar[str] = ContextVar("session_id", default="")
_user_code: ContextVar[str] = ContextVar("user_code", default="")


def new_task_id() -> str:
    """生成全局业务流水号：T-年月日-8位随机hex，如 T-20260920-3f9c1a2b。

    同一轮对话在 MySQL 与 PG 里都用这一个号，是跨库排障的唯一串联键。
    """
    return f"T-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"


def set_request_context(task_id: str = "", session_id: str = "", user_code: str = "") -> None:
    """在入口处设置上下文；传空串表示"这一项不变"，方便只更新其中一项。"""
    if task_id:
        _task_id.set(task_id)
    if session_id:
        _session_id.set(session_id)
    if user_code:
        _user_code.set(user_code)


def get_task_id() -> str:
    return _task_id.get()


def get_session_id() -> str:
    return _session_id.get()


def get_user_code() -> str:
    return _user_code.get()


def snapshot() -> dict:
    """取一份快照，用于日志与 SSE 事件回传前端。"""
    return {"task_id": _task_id.get(), "session_id": _session_id.get(), "user_code": _user_code.get()}
