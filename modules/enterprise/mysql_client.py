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

【两种连接：读写 / 只读】
    get_mysql_client()    读写，只给审计与用量落库用
    get_mysql_ro_client() 只读，权限/知识库授权/账单/对账查询用它，
                          并且代码层只放行 SELECT/SHOW/EXPLAIN（见 _guard）。
    对应大厂的"AI 只读账号访问业务库"：数据库 GRANT 是底线，
    代码层闸门保证换环境（用同一个高权限账号自测）时也不会失效。

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

    def __init__(self, read_only: bool = False):
        s = get_settings()
        self._read_only = read_only
        user, password = s.mysql_user, s.mysql_password
        if read_only and s.mysql_readonly_user:
            user, password = s.mysql_readonly_user, s.mysql_readonly_password
        self._conn_args = dict(
            host=s.mysql_host,
            port=s.mysql_port,
            user=user,
            password=password,
            database=s.mysql_database,
            charset="utf8mb4",
            autocommit=True,
            connect_timeout=5,
            read_timeout=10,
        )
        self._conn = None


    def _get_conn(self):
        """取连接；未连接或已断开则重建（最多重试一次）。"""
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError as e:
            raise RuntimeError("未安装 PyMySQL，请执行: pip install pymysql") from e

        if self._conn is None:
            self._conn = pymysql.connect(cursorclass=DictCursor, **self._conn_args)
            return self._conn

        try:
            self._conn.ping(reconnect=True)
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


    _READ_OK = ("SELECT", "SHOW", "DESC", "DESCRIBE", "EXPLAIN")

    def _guard(self, sql: str) -> None:
        """只读闸门：只放行 SELECT/SHOW/EXPLAIN，其余一律拒绝（代码层兜底）。

        【为什么要代码层再拦一道】
            数据库账号的 GRANT 是最后一道防线，但本地/测试环境常常图省事用同一个
            高权限账号，GRANT 形同虚设。代码层拦一道，换环境也不会失效。

        【输入 → 输出对照】
            _guard("SELECT * FROM sys_user")        → 放行（不抛）
            _guard("INSERT INTO sys_user ...")      → PermissionError: 只读账号禁止执行 INSERT 语句
            _guard("UPDATE sys_user SET status=0")  → PermissionError: 只读账号禁止执行 UPDATE 语句
        """
        if not self._read_only:
            return
        stripped = sql.strip()
        head = stripped.lstrip("(").split()[0].upper() if stripped else ""
        if head not in self._READ_OK:
            raise PermissionError(f"只读账号禁止执行 {head or '空'} 语句: {stripped[:60]}")

    def query_all(self, sql: str, args: tuple = ()) -> list:
        self._guard(sql)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall()

    def query_one(self, sql: str, args: tuple = ()) -> Optional[dict]:
        self._guard(sql)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchone()

    def execute(self, sql: str, args: tuple = ()) -> int:
        self._guard(sql)
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
_ro_client: Optional[MySQLClient] = None


def get_mysql_client() -> MySQLClient:
    """全局读写单例（懒连接，未用到就不建连接）：审计与用量落库走它。"""
    global _client
    if _client is None:
        _client = MySQLClient()
    return _client


def get_mysql_ro_client() -> MySQLClient:
    """全局只读单例：权限、知识库授权、账单与对账查询一律走它。

    AI 服务读业务数据（用户/角色/权限/知识库授权）只应该 SELECT，
    用只读客户端 + 代码层闸门，等价于"AI 只能看不能改业务库"。
    """
    global _ro_client
    if _ro_client is None:
        _ro_client = MySQLClient(read_only=True)
    return _ro_client
