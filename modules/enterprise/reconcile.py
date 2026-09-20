"""
跨库对账（最终一致性体检）
==========================
MySQL（业务/审计/用量）与 PG（会话/记忆/工具日志）之间没有事务，
"写 PG 成功、写 MySQL 失败"或反过来是常态。本模块按 task_id 比对两边，
把单边缺失的号找出来。

【为什么只报告不自动删】
    自动修复最容易做的是"把单边数据删掉"，可那一份可能就是唯一的真相。
    阶段1 只报告 + 告警；真要修，人看清楚哪边是脏数据再动手。
    大厂做法也是"定时对账 + 人工/半自动修复"，不是程序自己删。

【两种缺失各自意味着什么】
    pg_only    PG 有会话、MySQL 没有审计/用量
               → 多半是 MySQL 写入失败：账单少算钱、审计留白（合规风险）
    mysql_only MySQL 有审计/用量、PG 没有会话
               → 多半是会话已被归档清理（正常），或 PG 写入失败（会话丢失）

【输入 → 输出对照】
    reconcile(hours=24)
    → {"ok": True, "window_hours": 24,
       "pg_count": 120, "mysql_count": 118, "matched": 117,
       "pg_only": ["T-20260920-3f9c1a2b"], "mysql_only": [], "errors": []}

【依赖警告】
    RBAC_ENABLED=False 时 MySQL 侧根本不写审计与用量（见 llm._log_usage），
    这时候"MySQL 侧缺失"是设计如此，不是故障——本函数会跳过比对并注明原因。
"""
from modules.common.logger import get_logger
from config import get_settings

_logger = get_logger("reconcile")


def reconcile(hours: int = 24, limit: int = 5000) -> dict:
    """比对最近 N 小时内两边的 task_id，返回单边缺失清单。"""
    hours = int(hours)
    result = {
        "ok": True,
        "window_hours": hours,
        "pg_count": 0,
        "mysql_count": 0,
        "matched": 0,
        "pg_only": [],
        "mysql_only": [],
        "errors": [],
    }

    try:
        from modules.memory.pg_manager import PGManager

        pg_ids = set(PGManager().list_task_ids_since(hours, limit))
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"PG 不可用: {e}")
        return result
    result["pg_count"] = len(pg_ids)

    if not get_settings().rbac_enabled:
        result["note"] = "RBAC_ENABLED=False，MySQL 侧不落审计与用量，已跳过 MySQL 比对"
        return result

    try:
        from modules.enterprise.audit import list_task_ids_since

        mysql_ids = set(list_task_ids_since(hours, limit))
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"MySQL 不可用: {e}")
        return result
    result["mysql_count"] = len(mysql_ids)

    result["matched"] = len(pg_ids & mysql_ids)
    result["pg_only"] = sorted(pg_ids - mysql_ids)
    result["mysql_only"] = sorted(mysql_ids - pg_ids)

    if result["pg_only"] or result["mysql_only"]:
        _logger.warning(
            f"跨库对账不一致: pg_only={len(result['pg_only'])} mysql_only={len(result['mysql_only'])}"
        )
    return result
