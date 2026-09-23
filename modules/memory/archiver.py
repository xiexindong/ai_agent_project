"""
会话冷热分离（大厂阶段3 的简化版）
==================================
PG 只留最近 SESSION_HOT_DAYS（默认 90 天）的热会话，更老的先导出成 JSONL 归档
文件、再从 PG 删掉，避免"会话一直留在库里，半年后几十 G、查询越来越慢"。

大厂把归档丢对象存储（S3 / OSS），这里先落本地目录：接口形状与对象存储一致
（一次写一份不可变文件），将来换 OSS 只需改 _write_archive 一个函数。

【为什么不能直接 DELETE】
    直接删最省事，但审计和争议回溯一定会问"半年前那次对话到底说了什么"。
    先落归档再删：PG 体积可控，证据也还在。

【目录结构】
    archives/2026-09-20/s-3f9c1a2b.jsonl
    一行一个会话：{"session_id": ..., "history": [...], "tool_logs": [...]}

【输入 → 输出对照】
    archive_expired(retention_days=90)
    → {"ok": True, "retention_days": 90, "archived": 2, "deleted": 2,
       "files": ["archives/2026-09-20/s-3f9c1a2b.jsonl", ...]}

【依赖警告】
    删除走 chat_session 主键，chat_history 靠外键 ON DELETE CASCADE 连带删除；
    review_queue 不删（治理证据），但它里的 history_id 会变成悬空 ID，
    查审核队列时不要再回连 chat_history。
"""
# 【什么】标准库与项目内依赖
# 【作用】json 写 JSONL、datetime 取当天目录名、Path 拼路径建目录、get_settings 读归档配置
import json
from datetime import datetime
from pathlib import Path
from modules.common.logger import get_logger
from config import get_settings

_logger = get_logger("archiver")                    # 模块级日志器，前缀 archiver


# 【什么】把一个会话写成一份 JSONL 归档文件
# 【作用】删库之前先把证据落盘；将来换对象存储只改这一个函数
def _write_archive(session_id: str, history: list, tool_logs: list) -> str:
    """把一个会话写成一份 JSONL 归档文件，返回路径。

    default=str 是关键：created_at 是 datetime、tool_input 是 dict，
    json.dumps 默认只认识基础类型，不加它会在 datetime 上抛 TypeError。
    """
    s = get_settings()
    day = datetime.now().strftime("%Y-%m-%d")       # 当天日期做子目录名，如 2026-09-20
    out_dir = Path(s.archive_dir) / day             # 根目录默认 archives，见配置项 ARCHIVE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)      # 目录不存在就建，已存在不报错
    path = out_dir / f"{session_id}.jsonl"          # 一个会话一个文件，文件名即会话号
    record = {
        "session_id": session_id,                   # 会话号，回查时用它定位
        "archived_at": datetime.now().isoformat(),  # 归档时刻，可读时间串
        "history": history,                         # 该会话的全部对话页
        "tool_logs": tool_logs,                     # 该会话的全部工具流水
    }
    with path.open("a", encoding="utf-8") as f:     # 追加写：同一天重复归档同一会话会追加成多行
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return str(path)


# 【什么】只导出不删除地归档单个会话
# 【作用】给"删之前先留证据"的场景用；会话没数据就返回空串，不产生空文件
def archive_session(session_id: str) -> str:
    """只导出不删除：把一个会话的对话页与工具流水写成归档文件。

    输入 "s-3f9c1a2b" → 输出 "archives/2026-09-20/s-3f9c1a2b.jsonl"
    会话没有数据 → 返回 ""（不产生空文件）
    """
    from modules.memory.pg_manager import PGManager

    pg = PGManager()
    history = pg.fetch_history_by_session(session_id)        # 对话页列表
    tool_logs = pg.fetch_tool_logs_by_session(session_id)    # 工具流水列表
    if not history and not tool_logs:                        # 两边都空 = 会话没内容，不落空文件
        return ""
    return _write_archive(session_id, history, tool_logs)


# 【什么】批量只导出不删除地归档多个会话
# 【作用】离职清理、手工归档用；单个会话失败只记告警，不影响其余会话
def archive_session_ids(session_ids: list) -> list:
    """批量只导出不删除：给"删之前先留证据"的场景用（离职清理、手工归档）。

    输入 ["s-1", "s-2"] → 输出 ["archives/2026-09-20/s-1.jsonl", ...]
    单个会话失败只告警，不影响其它会话。
    """
    from modules.memory.pg_manager import PGManager

    pg = PGManager()
    files = []
    for sid in session_ids:
        try:
            history = pg.fetch_history_by_session(sid)
            tool_logs = pg.fetch_tool_logs_by_session(sid)
            if history or tool_logs:
                files.append(_write_archive(sid, history, tool_logs))
        except Exception as e:
            _logger.warning(f"会话 {sid} 归档失败: {e}")
    return files


# 【什么】冷热分离主入口：超期会话先归档再删除
# 【作用】控制 PG 体积同时保留可回溯证据；归档开关关闭时整段跳过
def archive_expired(retention_days: int = 0, limit: int = 500) -> dict:
    """冷热分离主入口：超期会话导出归档 → 从 PG 删除。"""
    s = get_settings()
    days = int(retention_days or s.session_hot_days)     # 传 0 就用配置的 SESSION_HOT_DAYS（默认 90 天）
    if not s.archive_enabled:
        return {"ok": True, "skipped": "ARCHIVE_ENABLED=False", "retention_days": days}

    result = {"ok": True, "retention_days": days, "archived": 0, "deleted": 0,
              "files": [], "errors": []}      # days 是本次实际保留天数；archived / deleted 为计数
    try:
        from modules.memory.pg_manager import PGManager

        pg = PGManager()
        session_ids = pg.list_expired_session_ids(days, limit)  # 最多取 limit（默认 500）个超期会话
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"PG 不可用: {e}")
        return result

    result["files"] = archive_session_ids(session_ids)      # 先落盘，拿到归档文件路径列表
    result["archived"] = len(result["files"])               # 成功归档的会话数 = 文件路径条数
    if len(result["files"]) != len(session_ids):
        _logger.warning(
            f"归档数量不一致: 待归档 {len(session_ids)} 实际 {len(result['files'])}，"
            f"未归档的会话本次不删除"
        )

    if session_ids:
        result["deleted"] = pg.delete_sessions(session_ids) or 0    # 对话页靠外键 ON DELETE CASCADE 连带删
    _logger.info(f"冷热分离完成: archived={result['archived']} deleted={result['deleted']}")
    return result
