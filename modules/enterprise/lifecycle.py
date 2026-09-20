"""
跨库数据生命周期治理
====================
MySQL 存组织主数据（谁在职），PostgreSQL 存 AI 运行数据（会话、记忆、工具日志）。
两库之间没有外键约束，所以「人在 MySQL 里离职了，数据在 PG 里还活着」是必然结果，
本模块负责把这条缝补上：

    ① sync    读 MySQL sys_user.status=0 的工号 → PG 侧会话打 archived=1 标记
    ② purge   标记超过 DATA_RETENTION_DAYS（默认 180 天）的会话真正删除

【为什么不做硬删除而是先打标】
    直接删会有两个问题：误删不可恢复、合规上无法证明"数据保留多久"。
    打标是软删除，保留期从标记时间开始计时，到期才落删，链路可审计。

【输入 → 输出对照】
    preview()
    → {"ok": True, "offboarded_users": ["U001"], "pending_sessions": {"U001": 3},
       "archived_users": [{"user_id": "U001", "sessions": 3, "archived_at": "2026-09-20 ..."}]}

    run_once()
    → {"ok": True, "archived_users": ["U001"], "sessions_archived": 3,
       "purged": {"retention_days": 180, "sessions_deleted": 0, "tool_logs_deleted": 0},
       "errors": []}

用法：
    python main.py offboard            # 只预览
    python main.py offboard --apply    # 执行归档 + 清理
    POST /admin/offboard/run           # 管理端触发
"""
from typing import Optional
from modules.common.logger import get_logger
from config import get_settings

_logger = get_logger("lifecycle")


def offboarded_users(tenant_code: str = "") -> list:
    """读 MySQL 里已离职（status=0）的用户工号列表。"""
    from modules.enterprise.mysql_client import get_mysql_ro_client

    if tenant_code:
        rows = get_mysql_ro_client().query_all(
            "SELECT u.user_code FROM sys_user u JOIN sys_tenant t ON t.id = u.tenant_id "
            "WHERE u.status = 0 AND t.tenant_code = %s",
            (tenant_code,),
        )
    else:
        rows = get_mysql_ro_client().query_all("SELECT user_code FROM sys_user WHERE status = 0")
    return [r["user_code"] for r in rows]


def preview(tenant_code: str = "") -> dict:
    """只对账不落改：报告哪些离职用户还有未归档会话、已归档的有多少。"""
    try:
        users = offboarded_users(tenant_code)
    except Exception as e:
        return {"ok": False, "error": f"MySQL 不可用: {e}"}

    result = {"ok": True, "offboarded_users": users, "pending_sessions": {}, "archived_users": []}
    try:
        from modules.memory.pg_manager import PGManager

        pg = PGManager()
        result["pending_sessions"] = pg.count_active_sessions(users)
        result["archived_users"] = pg.list_archived_users()
    except Exception as e:
        result["ok"] = False
        result["error"] = f"PG 不可用: {e}"
    return result


def run_once(retention_days: Optional[int] = None, tenant_code: str = "") -> dict:
    """执行一次完整处置：离职用户会话归档 + 超保留期数据清理。"""
    s = get_settings()
    if not s.offboard_archive_enabled:
        return {"ok": True, "skipped": "OFFBOARD_ARCHIVE_ENABLED=False"}

    days = int(retention_days) if retention_days is not None else int(s.data_retention_days)
    result = {"ok": True, "archived_users": [], "sessions_archived": 0, "purged": None, "errors": []}

    try:
        users = offboarded_users(tenant_code)
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"MySQL 不可用，无法读取离职名单: {e}")
        return result

    try:
        from modules.memory.pg_manager import PGManager

        pg = PGManager()
        if users:
            result["archived_users"] = users
            result["sessions_archived"] = pg.archive_user_sessions(users)

        # 先落归档再删：离职员工的数据到期后从 PG 清除，但证据要留在归档文件里。
        # 归档失败就不删——宁可多留几天，也不能删掉唯一一份记录。
        pending = pg.list_archived_session_ids(days)
        try:
            from modules.memory.archiver import archive_session_ids

            result["archived_files"] = archive_session_ids(pending)
        except Exception as e:
            result["ok"] = False
            result["errors"].append(f"归档失败，已跳过删除: {e}")
            return result

        result["purged"] = pg.purge_archived_data(days)
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"PG 处置失败: {e}")

    _logger.info(f"数据生命周期处置完成: {result}")
    return result
