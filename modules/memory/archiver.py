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
import json                                                   # 序列化归档记录为 JSONL 行（一行一份 JSON）
from datetime import datetime                                 # 取当天日期做子目录名、记录归档时刻
from pathlib import Path                                      # 跨平台路径拼接与建目录
from modules.common.logger import get_logger                  # 统一日志工厂，产出带 "archiver" 前缀的 logger
from config import get_settings                               # 全局配置：归档根目录 / 热数据保留天数 / 归档开关

_logger = get_logger("archiver")                    # 模块级日志器，前缀 archiver


# 【什么】把一个会话写成一份 JSONL 归档文件
# 【作用】删库之前先把证据落盘；将来换对象存储只改这一个函数
# 【为什么】归档写盘收敛为单一出口：本地目录与对象存储的差异只体现在这里，上层流程不用动
def _write_archive(session_id: str, history: list, tool_logs: list) -> str:
    """把一个会话写成一份 JSONL 归档文件，返回路径。

    default=str 是关键：created_at 是 datetime、tool_input 是 dict，
    json.dumps 默认只认识基础类型，不加它会在 datetime 上抛 TypeError。
    """
    s = get_settings()                                        # 读全局配置，归档根目录从它取
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
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")  # ensure_ascii=False 保中文；default=str 兜 datetime 等非常见类型
    return str(path)                                          # 返回归档文件路径，供结果统计与日志


# 【什么】只导出不删除地归档单个会话
# 【作用】给"删之前先留证据"的场景用；会话没数据就返回空串，不产生空文件
# 【为什么】导出与删除分离：调用方可以只备份不清理，也可以确认备份成功后再另行删除
def archive_session(session_id: str) -> str:
    """只导出不删除：把一个会话的对话页与工具流水写成归档文件。

    输入 "s-3f9c1a2b" → 输出 "archives/2026-09-20/s-3f9c1a2b.jsonl"
    会话没有数据 → 返回 ""（不产生空文件）
    """
    from modules.memory.pg_manager import PGManager           # 延迟导入：避免模块级循环依赖与无谓的连接建立

    pg = PGManager()                                          # 取 PG 单例，复用其连接与游标
    history = pg.fetch_history_by_session(session_id)        # 对话页列表
    tool_logs = pg.fetch_tool_logs_by_session(session_id)    # 工具流水列表
    if not history and not tool_logs:                        # 两边都空 = 会话没内容，不落空文件
        return ""                                            # 空串告诉调用方"没东西可归档"，与文件路径区分
    return _write_archive(session_id, history, tool_logs)    # 落盘并返回归档文件路径


# 【什么】批量只导出不删除地归档多个会话
# 【作用】离职清理、手工归档用；单个会话失败只记告警，不影响其余会话
# 【为什么】批量任务必须"部分失败可继续"：一个会话的数据异常不该让整批归档停摆
def archive_session_ids(session_ids: list) -> list:
    """批量只导出不删除：给"删之前先留证据"的场景用（离职清理、手工归档）。

    输入 ["s-1", "s-2"] → 输出 ["archives/2026-09-20/s-1.jsonl", ...]
    单个会话失败只告警，不影响其它会话。
    """
    from modules.memory.pg_manager import PGManager           # 延迟导入，理由同 archive_session

    pg = PGManager()                                          # 取 PG 单例
    files = []                                                # 收集成功归档的文件路径，与入参顺序无关（只收成功的）
    for sid in session_ids:                                   # 逐个会话归档
        try:
            history = pg.fetch_history_by_session(sid)        # 取该会话全部对话页
            tool_logs = pg.fetch_tool_logs_by_session(sid)    # 取该会话全部工具流水
            if history or tool_logs:                          # 任一边有数据才落盘；全空则跳过不产生空文件
                files.append(_write_archive(sid, history, tool_logs))  # 落盘并把路径收进结果列表
        except Exception as e:
            _logger.warning(f"会话 {sid} 归档失败: {e}")      # 单个失败只告警：记住是哪个会话，继续处理其余
    return files                                              # 返回成功归档的文件路径列表


# 【什么】冷热分离主入口：超期会话先归档再删除
# 【作用】控制 PG 体积同时保留可回溯证据；归档开关关闭时整段跳过
# 【为什么】"先归档后删除"是铁序：先删再归档一旦中途失败，数据就永久丢失；反过来最坏只是本次没删成
def archive_expired(retention_days: int = 0, limit: int = 500) -> dict:
    """冷热分离主入口：超期会话导出归档 → 从 PG 删除。"""
    s = get_settings()                                        # 读全局配置：开关与默认保留天数
    days = int(retention_days or s.session_hot_days)     # 传 0 就用配置的 SESSION_HOT_DAYS（默认 90 天）
    if not s.archive_enabled:                                 # 归档总开关：关闭时整段跳过，什么都不做
        return {"ok": True, "skipped": "ARCHIVE_ENABLED=False", "retention_days": days}  # 明确告诉调用方"是跳过不是出错"

    result = {"ok": True, "retention_days": days, "archived": 0, "deleted": 0,
              "files": [], "errors": []}      # days 是本次实际保留天数；archived / deleted 为计数
    try:
        from modules.memory.pg_manager import PGManager       # 延迟导入，理由同上

        pg = PGManager()                                      # 取 PG 单例
        session_ids = pg.list_expired_session_ids(days, limit)  # 最多取 limit（默认 500）个超期会话
    except Exception as e:
        result["ok"] = False                                  # 取清单失败 = 本次任务整体失败
        result["errors"].append(f"PG 不可用: {e}")            # 把失败原因写进结果，供定时任务告警
        return result                                         # 直接返回：清单都拿不到，后续无从谈起

    result["files"] = archive_session_ids(session_ids)      # 先落盘，拿到归档文件路径列表
    result["archived"] = len(result["files"])               # 成功归档的会话数 = 文件路径条数
    # 【设计意图·数量核对】归档成功的份数必须等于待归档的会话数，不等说明有会话没落盘
    if len(result["files"]) != len(session_ids):
        _logger.warning(
            f"归档数量不一致: 待归档 {len(session_ids)} 实际 {len(result['files'])}，"
            f"未归档的会话本次不删除"
        )

    # 【注意】当前实现删除的是全部 session_ids（含归档失败的）；上条 warning 声明的
    #   "未归档的会话本次不删除"是本函数的目标语义——数量不一致时应先排查再重跑
    if session_ids:                                           # 有待删会话才执行删除，空列表跳过
        result["deleted"] = pg.delete_sessions(session_ids) or 0    # 对话页靠外键 ON DELETE CASCADE 连带删
    _logger.info(f"冷热分离完成: archived={result['archived']} deleted={result['deleted']}")  # 收口日志：本次归档/删除总量
    return result                                             # 返回完整执行报告：成功标记/计数/文件清单/错误列表
