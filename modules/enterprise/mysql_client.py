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
from typing import Any, Optional                                  # 引入 Any（conn_args 字典值类型）与 Optional（连接槽位类型）
from modules.common.logger import get_logger                       # 统一日志器工厂
from config import get_settings                                    # 全局配置读取，延迟导入避免循环依赖

_logger = get_logger("mysql")                                      # 本模块日志器，名称固定为 mysql


# 【什么】MySQL 访问客户端（单例使用，见 get_mysql_client）
# 【作用】三个方法覆盖全部场景：query_all 查多行、query_one 查单行、execute 写操作
class MySQLClient:
    """带自动重连的 MySQL 客户端（单例使用，见 get_mysql_client）。"""

    # 【什么】构造客户端：只算参数，不建连接（懒连接）
    # 【作用】read_only=True 时换成只读账号，并把连接参数固化到 _conn_args
    def __init__(self, read_only: bool = False):
        s = get_settings()                                         # 读取全局配置，取数据库连接参数
        self._read_only = read_only                                  # True = 只读客户端，走 _guard 闸门
        user, password = s.mysql_user, s.mysql_password              # 默认用读写账号

        if read_only and s.mysql_readonly_user:
            # 【换账号】配了 mysql_readonly_user 就换成它，没配则沿用读写账号
            user, password = s.mysql_readonly_user, s.mysql_readonly_password  # 替换成只读账号与密码

        # 【类型说明】这里必须是 Any 而不是 str | int：
        #   下面 pymysql.connect(**self._conn_args) 会把字典解包成 host/port/user/... 等形参，
        #   它们的类型各不相同（int/str/bool 都有），写成联合类型会导致解包时逐形参报类型不兼容
        self._conn_args: dict[str, Any] = dict(
            host=s.mysql_host,          # 主机地址
            port=s.mysql_port,          # 端口
            user=user,                  # 账号（只读客户端会被换成只读账号）
            password=password,          # 密码
            database=s.mysql_database,  # 默认库
            charset="utf8mb4",          # 字符集：支持 4 字节字符与完整中文
            autocommit=True,            # 每条写语句自动提交，不手工开事务
            connect_timeout=5,          # 建连超时 5 秒，超时即判定不可用
            read_timeout=10,            # 读超时 10 秒，慢查询最多等这么久
        )
        self._conn = None               # 连接对象；None 表示还没建过连接


    # 【什么】取可用连接，断了就重建
    # 【作用】所有查询/写入都从这里拿连接；重连逻辑集中在一处，上层不用关心
    def _get_conn(self):
        """取连接；未连接或已断开则重建（最多重试一次）。"""
        # 【延迟导入】PyMySQL 是可选依赖，用到才 import，未安装也不影响模块导入
        try:
            import pymysql                                            # 引入 PyMySQL 驱动（可选依赖，延迟加载）
            from pymysql.cursors import DictCursor                     # 引入字典游标类（返回 dict 行）
        except ImportError as e:
            raise RuntimeError("未安装 PyMySQL，请执行: pip install pymysql") from e  # 驱动缺失时抛明确异常

        if self._conn is None:
            # 【首次建连】DictCursor 让每行直接返回 dict，按字段名取值
            self._conn = pymysql.connect(cursorclass=DictCursor, **self._conn_args)   # 按参数字典创建带字典游标的连接
            return self._conn                                          # 首次建连后直接返回

        try:
            self._conn.ping(reconnect=True)   # 复用前先探活，断了 pymysql 自己重连
        except Exception:
            _logger.warning("MySQL 连接已断开，尝试重连")
            self._conn = pymysql.connect(cursorclass=DictCursor, **self._conn_args)   # ping 也失败 → 整体重建
        return self._conn                                              # 返回可用连接对象

    # 【什么】健康检查
    # 【作用】启动自检与 RBAC 可用性探测用；失败返回 False 而不是抛异常
    def ping(self) -> bool:
        """健康检查：启动自检用，失败返回 False 而非抛异常。"""
        try:
            self._get_conn()                                           # 尝试建连/复用，成功即连通
            return True                                                # 建连成功，标记可用
        except Exception as e:
            _logger.warning(f"MySQL 不可用: {e}")
            return False                                               # 建连/探活失败，标记不可用


    # 【什么】只读账号允许的头关键字白名单（5 个）
    # 【作用】_guard 拿它做前缀判定，不在表里的语句一律拒绝
    _READ_OK = ("SELECT", "SHOW", "DESC", "DESCRIBE", "EXPLAIN")     # 只读客户端允许的 5 个 SQL 起始关键字

    # 【什么】只读闸门：拦掉非查询语句
    # 【作用】数据库 GRANT 之外再挡一道，换环境用高权限账号时也不会误写
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
            return                      # 读写客户端不做任何限制
        stripped = sql.strip()          # 去首尾空白
        # 【取头关键字】跳过开头的左括号再取第一个词并转大写；空 SQL 得到 ""
        head = stripped.lstrip("(").split()[0].upper() if stripped else ""  # 取 SQL 第一个关键字并大写
        if head not in self._READ_OK:
            # 【拒绝】错误信息里只带前 60 个字符，不把整条 SQL 打进日志
            raise PermissionError(f"只读账号禁止执行 {head or '空'} 语句: {stripped[:60]}")  # 抛异常拦截非查询语句

    # 【什么】查多行
    # 【作用】返回 list[dict]；无匹配行时是空 list，不是 None
    def query_all(self, sql: str, args: tuple = ()) -> list:
        self._guard(sql)            # 只读闸门
        conn = self._get_conn()     # 取/建连接
        with conn.cursor() as cur:  # 上下文管理器，游标自动关闭
            cur.execute(sql, args)                                     # 执行 SQL 并绑定参数
            return list(cur.fetchall())   # 全部结果行；pymysql 返回 tuple，统一成 list 交付

    # 【什么】查单行
    # 【作用】返回 dict 或 None；惯用法是 `if not row: 拒绝`
    def query_one(self, sql: str, args: tuple = ()) -> Optional[dict]:
        self._guard(sql)                                               # 只读闸门
        conn = self._get_conn()                                        # 取/建连接
        with conn.cursor() as cur:                                     # 上下文管理器，游标自动关闭
            cur.execute(sql, args)                                     # 执行 SQL 并绑定参数
            return cur.fetchone()   # 只取第一行；没有就是 None

    # 【什么】执行写语句
    # 【作用】返回受影响行数；autocommit=True 下执行完即落库
    def execute(self, sql: str, args: tuple = ()) -> int:
        self._guard(sql)                                               # 只读闸门
        conn = self._get_conn()                                        # 取/建连接
        with conn.cursor() as cur:                                     # 上下文管理器，游标自动关闭
            return cur.execute(sql, args)   # 返回值即受影响行数

    # 【什么】关闭连接
    # 【作用】置回 None，下次查询会重新建连；关闭本身失败也不抛
    def close(self):
        if self._conn is not None:                                     # 有连接才尝试关闭
            try:
                self._conn.close()                                     # 释放底层 TCP 连接
            except Exception:
                pass                   # 关闭失败也无所谓，反正要丢弃这条连接
            self._conn = None                                          # 置回 None，下次查询会重新建连


# 【什么】两个全局单例槽位
# 【作用】读写与只读各存一个实例，全进程复用同一条连接，避免反复建连
_client: Optional[MySQLClient] = None       # 读写客户端
_ro_client: Optional[MySQLClient] = None    # 只读客户端


# 【什么】取全局读写单例
# 【作用】审计与用量落库走它；懒连接，从没写过库就不建连接
def get_mysql_client() -> MySQLClient:
    """全局读写单例（懒连接，未用到就不建连接）：审计与用量落库走它。"""
    global _client                                                 # 声明改模块级变量
    if _client is None:                                            # 首次调用才创建
        _client = MySQLClient()                                    # 创建读写客户端并缓存
    return _client                                                 # 返回全局唯一读写客户端


# 【什么】取全局只读单例
# 【作用】权限、知识库授权、账单与对账查询一律走它，等价于"AI 只能看不能改业务库"
def get_mysql_ro_client() -> MySQLClient:
    """全局只读单例：权限、知识库授权、账单与对账查询一律走它。

    AI 服务读业务数据（用户/角色/权限/知识库授权）只应该 SELECT，
    用只读客户端 + 代码层闸门，等价于"AI 只能看不能改业务库"。
    """
    global _ro_client                                              # 声明改模块级变量
    if _ro_client is None:                                         # 首次调用才创建
        _ro_client = MySQLClient(read_only=True)                   # 创建只读客户端并缓存
    return _ro_client                                              # 返回全局唯一只读客户端
