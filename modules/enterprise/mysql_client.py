"""
MySQL 访问层
============
极简封装：连接复用 + 断线自动重连 + 字典游标。

【为什么不用 ORM】
    权限查询就几条固定 SQL，且要能一眼看懂、方便 DBA 评审；ORM 在这里只会增加
    心智负担。真要复杂关联查询时再上 SQLAlchemy 不迟。

【三个方法覆盖全部场景】
    query_all  查多行 → list[dict]
    query_one  查单行 → dict | None
    execute    写操作 → 影响行数

【降级设计】
    所有方法在连接不可用时 raise，由上层（RBACService / audit）决定是否降级，
    本层不做静默吞异常——静默吞会让"权限系统挂了"伪装成"没权限"，非常危险。
"""
from typing import Any, Optional
from modules.common.logger import get_logger
from config import get_settings

_logger = get_logger("mysql")


class MySQLClient:
    """带自动重连的 MySQL 客户端（单例使用，见 get_mysql_client）。"""

    def __init__(self):
        s = get_settings()
        self._conn_args = dict(
            host=s.mysql_host,
            port=s.mysql_port,
            user=s.mysql_user,
            password=s.mysql_password,
            database=s.mysql_database,
            charset="utf8mb4",
            autocommit=True,          # 权限库都是单行写入，无需显式事务
            connect_timeout=5,
            read_timeout=10,
        )
        self._conn = None

    # ==================== 连接管理 ====================

    def _get_conn(self):
        """取连接；未连接或已断开则重建（最多重试一次）。"""
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError as e:            # pragma: no cover - 依赖未装
            raise RuntimeError("未安装 PyMySQL，请执行: pip install pymysql") from e

        if self._conn is None:
            self._conn = pymysql.connect(cursorclass=DictCursor, **self._conn_args)
            return self._conn

        try:
            self._conn.ping(reconnect=True)     # MySQL 8 小时断连的经典坑
        except Exception:
            _logger.warning("MySQL 连接已断开，尝试重连")
            self._conn = pymysql.connect(cursorclass=DictCursor, **self._conn_args)
        return self._conn

    def ping(self) -> bool:
        """健康检查：启动自检用，失败返回 False 而非抛异常。"""
        try:
            self._get_conn()
            return True
        except Exception as e:
            _logger.warning(f"MySQL 不可用: {e}")
            return False

    # ==================== 查询 ====================

    def query_all(self, sql: str, args: tuple = ()) -> list:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall()

    def query_one(self, sql: str, args: tuple = ()) -> Optional[dict]:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchone()

    def execute(self, sql: str, args: tuple = ()) -> int:
        conn = self._get_conn()
        with conn.cursor() as cur:
            return cur.execute(sql, args)

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


_client: Optional[MySQLClient] = None


def get_mysql_client() -> MySQLClient:
    """全局单例客户端（懒连接，未用到就不建连接）。"""
    global _client
    if _client is None:
        _client = MySQLClient()
    return _client
