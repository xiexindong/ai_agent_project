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
from modules.common.logger import get_logger                      # 统一日志器工厂
from config import get_settings                                   # 全局配置读取

_logger = get_logger("reconcile")                                 # 本模块日志器，名称固定为 reconcile


# 【什么】跨库对账入口：按 task_id 比对 PG 与 MySQL
# 【作用】把单边缺失的流水号找出来；只报告不自动删，默认窗口 24 小时、上限 5000 条
def reconcile(hours: int = 24, limit: int = 5000) -> dict:
    """比对最近 N 小时内两边的 task_id，返回单边缺失清单。"""
    hours = int(hours)     # 转 int，避免字符串被拼进 SQL 的 INTERVAL
    result = {
        "ok": True,                # 是否跑通；任一侧不可用就置 False
        "window_hours": hours,     # 比对窗口（小时）
        "pg_count": 0,             # PG 侧流水号条数
        "mysql_count": 0,          # MySQL 侧流水号条数
        "matched": 0,              # 两边都有的条数
        "pg_only": [],             # 只有 PG 有：MySQL 侧写入可能失败
        "mysql_only": [],          # 只有 MySQL 有：PG 会话可能已被归档清理
        "errors": [],              # 异常信息列表
    }

    try:
        # 【延迟导入】PGManager 依赖 PG 驱动，用到才导入
        from modules.memory.pg_manager import PGManager            # 延迟导入 PG 管理器

        pg_ids = set(PGManager().list_task_ids_since(hours, limit))   # PG 侧的 task_id 集合
    except Exception as e:
        result["ok"] = False                                      # PG 读不到，标记失败
        result["errors"].append(f"PG 不可用: {e}")                # 记录错误
        return result                                             # 提前返回，不做后续比对
    result["pg_count"] = len(pg_ids)                              # 统计 PG 侧条数

    # 【跳过比对】RBAC 关闭时 MySQL 侧根本不写审计与用量，此时"MySQL 缺失"是设计如此
    if not get_settings().rbac_enabled:
        result["note"] = "RBAC_ENABLED=False，MySQL 侧不落审计与用量，已跳过 MySQL 比对"
        return result                                             # RBAC 关闭时直接返回，附 note 说明

    try:
        from modules.enterprise.audit import list_task_ids_since  # 延迟导入审计流水号查询

        mysql_ids = set(list_task_ids_since(hours, limit))   # MySQL 侧 = 审计 ∪ 用量
    except Exception as e:
        result["ok"] = False                                      # MySQL 读不到，标记失败
        result["errors"].append(f"MySQL 不可用: {e}")             # 记录错误
        return result                                             # 提前返回
    result["mysql_count"] = len(mysql_ids)                        # 统计 MySQL 侧条数

    result["matched"] = len(pg_ids & mysql_ids)              # 交集：两边都写了
    result["pg_only"] = sorted(pg_ids - mysql_ids)           # 差集排序，输出顺序稳定
    result["mysql_only"] = sorted(mysql_ids - pg_ids)        # 同上

    if result["pg_only"] or result["mysql_only"]:
        _logger.warning(
            f"跨库对账不一致: pg_only={len(result['pg_only'])} mysql_only={len(result['mysql_only'])}"
        )
    return result                                                 # 返回对账结果字典
