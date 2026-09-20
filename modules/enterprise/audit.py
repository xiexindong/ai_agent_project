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
from modules.enterprise.mysql_client import get_mysql_client

_logger = get_logger("audit")


def _client():
    return get_mysql_client()


def log_audit(
    user_code: str,
    action: str,
    resource: str = "",
    result: str = "allow",
    dept_code: str = "",
    detail: Optional[dict] = None,
    ip: str = "",
    tenant_id: Optional[int] = None,
):
    """写一条审计日志；失败只告警。"""
    try:
        _client().execute(
            "INSERT INTO sys_audit_log (tenant_id, user_code, dept_code, action, resource, result, detail, ip) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                tenant_id,
                user_code or "anonymous",
                dept_code,
                action,
                resource,
                result,
                json.dumps(detail, ensure_ascii=False) if detail else None,
                ip,
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
):
    """记录一次 LLM 调用的 token 消耗，用于部门账单与预算控制。"""
    try:
        _client().execute(
            "INSERT INTO sys_llm_usage (user_code, dept_code, session_id, model, "
            "prompt_tokens, completion_tokens, total_tokens, latency_ms) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                user_code or "anonymous",
                dept_code,
                session_id,
                model,
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,
                latency_ms,
            ),
        )
    except Exception as e:
        _logger.warning(f"用量写入失败: {e}")


def dept_token_bill(dept_code: str) -> Optional[dict]:
    """按部门汇总 token 消耗（管理员账单接口的数据源）。"""
    try:
        row = _client().query_one(
            "SELECT COUNT(*) AS calls, SUM(total_tokens) AS tokens "
            "FROM sys_llm_usage WHERE dept_code=%s",
            (dept_code,),
        )
        return {"dept_code": dept_code, "calls": row["calls"] or 0, "tokens": int(row["tokens"] or 0)}
    except Exception as e:
        _logger.warning(f"账单查询失败: {e}")
        return None
