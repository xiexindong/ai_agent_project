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
import json                                                       # 用于把 detail 字典序列化成 JSON 字符串入库
from typing import Optional                                       # 用于 dept_token_bill 返回类型标注
from modules.common.logger import get_logger                      # 统一日志器工厂
from modules.common import context as req_ctx                     # 请求上下文，自动补齐 session_id/task_id
from modules.enterprise.mysql_client import get_mysql_client, get_mysql_ro_client  # 读写/只读两个 MySQL 单例

_logger = get_logger("audit")                                     # 本模块日志器，名称固定为 audit


# 【什么】取读写客户端
# 【作用】审计与用量都是 INSERT，只能走读写连接
def _client():
    return get_mysql_client()                                     # 返回全局读写 MySQL 客户端


# 【什么】取只读客户端
# 【作用】账单与对账是纯查询，走只读连接，代码层还挡一道写语句闸门
def _ro_client():
    """只读连接：账单与对账这类纯查询走它，代码层禁止写语句。"""
    return get_mysql_ro_client()                                  # 返回全局只读 MySQL 客户端


# 【什么】写一条行为审计
# 【作用】放行与拒绝都留痕；写失败只降级到应用日志，绝不阻断主流程
def log_audit(
    user_code: str,                      # 工号；传空则落库为 anonymous
    action: str,                         # 动作名，如 auth / chat / tool_call
    resource: str = "",                  # 资源名，如 api_key:<16位摘要> / kb:finance
    result: str = "allow",               # 结果：allow / deny / error
    dept_code: str = "",                 # 部门码，账单与数据域过滤用
    detail: Optional[dict] = None,       # 明细字典，落库前 json.dumps 成文本
    ip: str = "",                        # 来源 IP
    tenant_id: Optional[int] = None,     # 租户 ID，可为空
    session_id: str = "",                # 会话号；留空自动取请求上下文
    task_id: str = "",                   # 流水号；留空自动取请求上下文
):
    """写一条审计日志；失败只告警。

    session_id / task_id 是跨库串联键：MySQL 审计与 PG 的 chat_history、
    tool_call_log 靠它们对齐同一次会话，否则排障只能按 user_code + 时间范围
    模糊匹配。两者都可由请求上下文自动补齐，调用方不用层层透传。
    """
    session_id = session_id or req_ctx.get_session_id()   # 没传就从请求上下文补，避免串联键为空
    task_id = task_id or req_ctx.get_task_id()            # 同上，保证与 PG 侧对得上
    try:
        # 【依赖警告】INSERT 的 10 个列与下面 VALUES 的 10 个 %s 必须一一对应，改列序会静默写错字段
        _client().execute(
            "INSERT INTO sys_audit_log (tenant_id, user_code, dept_code, action, resource, result, detail, ip, session_id, task_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                tenant_id,                                        # 租户 ID，未传则为 None
                user_code or "anonymous",                                        # 空工号统一记 anonymous
                dept_code,                                        # 部门码，按部门归集/过滤
                action,                                           # 动作名，如 auth/chat/tool_call
                resource,                                         # 资源名，如 api_key:<摘要> / kb:finance
                result,                                           # 结果：allow / deny / error
                json.dumps(detail, ensure_ascii=False) if detail else None,      # 明细转 JSON 文本；空明细写 NULL
                ip,                                               # 来源 IP
                session_id,                                       # 会话号，跨库串联键
                task_id,                                          # 流水号，跨库串联键
            ),
        )
    except Exception as e:
        _logger.warning(f"审计写入失败（降级到应用日志）: {e} | {user_code} {action} {resource} {result}")
        if result == "deny":
            # 拒绝的记录再单独打一条：越权尝试是安全审计的重点对象，不能随库一起丢
            _logger.warning(f"[DENY] user={user_code} action={action} resource={resource} detail={detail}")


# 【什么】记录一次 LLM 调用的用量
# 【作用】按人/部门归集 token，是部门账单与预算控制的数据源；写失败只告警
def log_llm_usage(
    user_code: str,                        # 工号；传空则落库为 anonymous
    model: str,                            # 模型名，如 deepseek-chat
    prompt_tokens: int = 0,                # 输入 token 数
    completion_tokens: int = 0,            # 输出 token 数
    latency_ms: int = 0,                   # 本次调用耗时（毫秒）
    dept_code: str = "",                   # 部门码，账单按部门归集
    session_id: str = "",                  # 会话号；留空自动取请求上下文
    task_id: str = "",                     # 流水号；留空自动取请求上下文
):
    """记录一次 LLM 调用的 token 消耗，用于部门账单与预算控制。"""
    session_id = session_id or req_ctx.get_session_id()   # 与审计同一套串联键
    task_id = task_id or req_ctx.get_task_id()            # 与审计同一套串联键
    try:
        _client().execute(
            "INSERT INTO sys_llm_usage (user_code, dept_code, session_id, task_id, model, "
            "prompt_tokens, completion_tokens, total_tokens, latency_ms) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                user_code or "anonymous",                          # 空工号统一记 anonymous
                dept_code,                                         # 部门码，按部门归集 token
                session_id,                                        # 会话号
                task_id,                                           # 流水号
                model,                                             # 模型名
                prompt_tokens,                                     # 输入 token 数
                completion_tokens,                                 # 输出 token 数
                prompt_tokens + completion_tokens,   # total_tokens = 输入 + 输出，与表里第 8 列对应
                latency_ms,                                        # 本次调用耗时（毫秒）
            ),
        )
    except Exception as e:
        _logger.warning(f"用量写入失败: {e}")


# 【什么】取最近 N 小时落过库的 task_id（审计 ∪ 用量）
# 【作用】跨库对账时拿它当 MySQL 侧的流水号集合；默认窗口 24 小时、上限 5000 条
def list_task_ids_since(hours: int = 24, limit: int = 5000) -> list:
    """取最近 N 小时落过库的 task_id（审计 ∪ 用量），供跨库对账比对。

    输入 hours=24 → 输出 ["T-20260920-3f9c1a2b", "T-20260920-91ad44e0", ...]
    没配 MySQL 时抛异常，由 reconcile 上层决定是告警还是跳过。
    """
    # 【两表 UNION】审计表 ∪ 用量表；UNION 自带去重，hours 要传两次（两表各一次）
    # 【边界】只取 task_id 非空的记录，空的号没法跟 PG 侧比对
    rows = _ro_client().query_all(
        "SELECT task_id FROM sys_audit_log "
        "WHERE task_id <> '' AND created_at >= NOW() - INTERVAL %s HOUR "
        "UNION "
        "SELECT task_id FROM sys_llm_usage "
        "WHERE task_id <> '' AND created_at >= NOW() - INTERVAL %s HOUR "
        "LIMIT %s",
        (int(hours), int(hours), int(limit)),   # hours 传两次，limit 传一次
    )
    return [r["task_id"] for r in rows if r.get("task_id")]   # 再过滤一次空号


# 【什么】按部门汇总 token 消耗
# 【作用】管理员账单接口的数据源；查失败返回 None，由上层决定怎么展示
def dept_token_bill(dept_code: str) -> Optional[dict]:
    """按部门汇总 token 消耗（管理员账单接口的数据源）。"""
    try:
        row = _ro_client().query_one(
            "SELECT COUNT(*) AS calls, SUM(total_tokens) AS tokens "
            "FROM sys_llm_usage WHERE dept_code=%s",
            (dept_code,),                                         # 按部门码过滤
        )
        # 【or 0 兜底】该部门没记录时 SUM/COUNT 返回 NULL，用 or 0 转成数字，接口不会吐 null
        return {"dept_code": dept_code, "calls": row["calls"] or 0, "tokens": int(row["tokens"] or 0)}
    except Exception as e:
        _logger.warning(f"账单查询失败: {e}")
        return None                                               # 查询失败返回 None，由上层决定降级展示
