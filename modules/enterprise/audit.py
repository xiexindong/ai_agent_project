"""
审计与用量落库
==============
两张表：
    sys_audit_log  行为审计（谁在什么时候对什么资源做了什么，结果如何）
    sys_llm_usage  LLM 用量（按人/部门归集 token，做成本与预算控制）

【为什么拒绝也要记】
    result='deny' 的记录是安全审计的重点：一个普通员工反复尝试查全公司订单，
    说明要么是权限配错了，要么是有人在试探边界。只看 allow 的日志等于没有审计。

【为什么写失败不能抛异常】
    审计是旁路能力，它挂了不应该让对话挂掉。这里统一 try/except + 降级到应用日志，
    保证「审计不可用」永远不等于「服务不可用」。生产环境可改成投递到 MQ 异步落库。
"""
import json
from typing import Optional
from modules.common.logger import get_logger
from modules.common import context as req_ctx
from modules.enterprise.mysql_client import get_mysql_client, get_mysql_ro_client

_logger = get_logger("audit")


def _client():
    return get_mysql_client()


def _ro_client():
    """只读连接：账单与对账这类纯查询走它，代码层禁止写语句。"""
    return get_mysql_ro_client()


def log_audit(
    user_code: str,
    action: str,
    resource: str = "",
    result: str = "allow",
    dept_code: str = "",
    detail: Optional[dict] = None,
    ip: str = "",
    tenant_id: Optional[int] = None,
    session_id: str = "",
    task_id: str = "",
):
    """写一条审计日志；失败只告警。

    session_id / task_id 是跨库串联键：MySQL 审计与 PG 的 chat_history、
    tool_call_log 靠它们对齐同一次会话，否则排障只能按 user_code + 时间范围
    模糊匹配。两者都可由请求上下文自动补齐，调用方不用层层透传。
    """
    session_id = session_id or req_ctx.get_session_id()
    task_id = task_id or req_ctx.get_task_id()
    try:
        _client().execute(
            "INSERT INTO sys_audit_log (tenant_id, user_code, dept_code, action, resource, result, detail, ip, session_id, task_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                tenant_id,
                user_code or "anonymous",
                dept_code,
                action,
                resource,
                result,
                json.dumps(detail, ensure_ascii=False) if detail else None,
                ip,
                session_id,
                task_id,
            ),
        )
    except Exception as e:
        _logger.warning(f"审计写入失败（降级到应用日志）: {e} | {user_code} {action} {resource} {result}")
        if result == "deny":
            _logger.warning(f"[DENY] user={user_code} action={action} resource={resource} detail={detail}")


def log_llm_usage(
    user_code: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
    dept_code: str = "",
    session_id: str = "",
    task_id: str = "",
):
    """记录一次 LLM 调用的 token 消耗，用于部门账单与预算控制。"""
    session_id = session_id or req_ctx.get_session_id()
    task_id = task_id or req_ctx.get_task_id()
    try:
        _client().execute(
            "INSERT INTO sys_llm_usage (user_code, dept_code, session_id, task_id, model, "
            "prompt_tokens, completion_tokens, total_tokens, latency_ms) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                user_code or "anonymous",
                dept_code,
                session_id,
                task_id,
                model,
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,
                latency_ms,
            ),
        )
    except Exception as e:
        _logger.warning(f"用量写入失败: {e}")


def list_task_ids_since(hours: int = 24, limit: int = 5000) -> list:
    """取最近 N 小时落过库的 task_id（审计 ∪ 用量），供跨库对账比对。

    输入 hours=24 → 输出 ["T-20260920-3f9c1a2b", "T-20260920-91ad44e0", ...]
    没配 MySQL 时抛异常，由 reconcile 上层决定是告警还是跳过。
    """
    rows = _ro_client().query_all(
        "SELECT task_id FROM sys_audit_log "
        "WHERE task_id <> '' AND created_at >= NOW() - INTERVAL %s HOUR "
        "UNION "
        "SELECT task_id FROM sys_llm_usage "
        "WHERE task_id <> '' AND created_at >= NOW() - INTERVAL %s HOUR "
        "LIMIT %s",
        (int(hours), int(hours), int(limit)),
    )
    return [r["task_id"] for r in rows if r.get("task_id")]


def dept_token_bill(dept_code: str) -> Optional[dict]:
    """按部门汇总 token 消耗（管理员账单接口的数据源）。"""
    try:
        row = _ro_client().query_one(
            "SELECT COUNT(*) AS calls, SUM(total_tokens) AS tokens "
            "FROM sys_llm_usage WHERE dept_code=%s",
            (dept_code,),
        )
        return {"dept_code": dept_code, "calls": row["calls"] or 0, "tokens": int(row["tokens"] or 0)}
    except Exception as e:
        _logger.warning(f"账单查询失败: {e}")
        return None
