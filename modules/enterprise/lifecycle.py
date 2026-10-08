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
from typing import Optional                                       # 用于 run_once 参数与返回类型
from modules.common.logger import get_logger                      # 统一日志器工厂
from config import get_settings                                   # 全局配置读取

_logger = get_logger("lifecycle")                                 # 本模块日志器，名称固定为 lifecycle


# 【什么】读 MySQL 里的离职名单
# 【作用】取 sys_user.status=0 的工号；传租户码就只查该租户下的离职人员
def offboarded_users(tenant_code: str = "") -> list:
    """读 MySQL 里已离职（status=0）的用户工号列表。"""
    # 【延迟导入】mysql_client 依赖配置与 PyMySQL，用到才导入，未配 MySQL 也不影响本模块导入
    from modules.enterprise.mysql_client import get_mysql_ro_client  # 只读 MySQL 客户端（查离职名单只读）

    if tenant_code:
        rows = get_mysql_ro_client().query_all(
            "SELECT u.user_code FROM sys_user u JOIN sys_tenant t ON t.id = u.tenant_id "
            "WHERE u.status = 0 AND t.tenant_code = %s",
            (tenant_code,),                                        # 按租户码过滤
        )
    else:
        # 【不带租户】全租户扫描，适合单租户部署
        rows = get_mysql_ro_client().query_all("SELECT user_code FROM sys_user WHERE status = 0")  # 全租户查询
    return [r["user_code"] for r in rows]                         # 只取 user_code 列，返回工号列表


# 【什么】预览：只报告不落改
# 【作用】告诉运维哪些离职用户还有未归档会话、已归档了哪些，执行前先看一眼
def preview(tenant_code: str = "") -> dict:
    """只对账不落改：报告哪些离职用户还有未归档会话、已归档的有多少。"""
    try:
        users = offboarded_users(tenant_code)                     # 读离职名单
    except Exception as e:
        return {"ok": False, "error": f"MySQL 不可用: {e}"}   # MySQL 读不到名单，直接返回失败

    # 【结果骨架】ok=是否跑通；pending_sessions 形如 {"U001": 3}；archived_users 是已归档清单
    result = {"ok": True, "offboarded_users": users, "pending_sessions": {}, "archived_users": []}  # 构造结果骨架
    try:
        from modules.memory.pg_manager import PGManager           # 延迟导入 PG 管理器

        pg = PGManager()                                          # 创建 PGManager 实例
        result["pending_sessions"] = pg.count_active_sessions(users)   # 每人还剩几个未归档会话
        result["archived_users"] = pg.list_archived_users()            # 已归档用户及归档时间
    except Exception as e:
        result["ok"] = False                                      # PG 异常，标记失败
        result["error"] = f"PG 不可用: {e}"                      # 写入错误说明
    return result                                                 # 返回预览报告字典


# 【什么】执行一次完整处置：归档 + 清理
# 【作用】离职用户会话打归档标记，超保留期的数据真正删除；retention_days 为空时取配置值
def run_once(retention_days: Optional[int] = None, tenant_code: str = "") -> dict:
    """执行一次完整处置：离职用户会话归档 + 超保留期数据清理。"""
    s = get_settings()                                            # 读取全局配置
    if not s.offboard_archive_enabled:
        return {"ok": True, "skipped": "OFFBOARD_ARCHIVE_ENABLED=False"}   # 总开关关闭：直接跳过

    # 【保留期】不传就用配置的 DATA_RETENTION_DAYS（默认 180 天）
    days = int(retention_days) if retention_days is not None else int(s.data_retention_days)  # 参数优先，否则取配置
    result = {"ok": True, "archived_users": [], "sessions_archived": 0, "purged": None, "errors": []}  # 结果骨架

    try:
        users = offboarded_users(tenant_code)                     # 读取离职名单
    except Exception as e:
        result["ok"] = False                                      # 读 MySQL 失败
        result["errors"].append(f"MySQL 不可用，无法读取离职名单: {e}")  # 记录错误
        return result                                             # 提前返回，后续不执行

    try:
        from modules.memory.pg_manager import PGManager           # 延迟导入 PG 管理器

        pg = PGManager()                                          # 创建 PGManager 实例
        if users:
            result["archived_users"] = users                                # 本次归档了哪些人
            result["sessions_archived"] = pg.archive_user_sessions(users)   # 归档了多少个会话

        # 先落归档再删：离职员工的数据到期后从 PG 清除，但证据要留在归档文件里。
        # 归档失败就不删——宁可多留几天，也不能删掉唯一一份记录。
        pending = pg.list_archived_session_ids(days)   # 已打标且超过 days 天的会话 ID
        try:
            from modules.memory.archiver import archive_session_ids  # 延迟导入归档函数

            result["archived_files"] = archive_session_ids(pending)   # 先落归档文件，证据留下来
        except Exception as e:
            result["ok"] = False                                    # 归档文件失败
            result["errors"].append(f"归档失败，已跳过删除: {e}")    # 记录错误并提前返回
            return result

        result["purged"] = pg.purge_archived_data(days)   # 归档成功后才真正删 PG 数据
    except Exception as e:
        result["ok"] = False                                      # PG 处置失败
        result["errors"].append(f"PG 处置失败: {e}")              # 记录错误

    _logger.info(f"数据生命周期处置完成: {result}")              # 处置完成后统一写 info 日志
    return result                                                 # 返回处置结果字典
