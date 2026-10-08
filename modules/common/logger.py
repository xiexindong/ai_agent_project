"""
日志封装模块
============
基于 loguru 实现全链路日志：控制台彩色输出 + 文件轮转持久化。
支持按模块隔离日志、自动捕获未处理异常。
"""
import sys                                    # 标准库：取 sys.stderr 作为控制台日志的输出目标
from pathlib import Path                      # 标准库：跨平台路径对象，用来拼出 logs 目录
from loguru import logger                     # loguru 全局 logger：所有 sink（输出目标）都注册在它身上
from config import get_settings               # 项目配置入口：读取 LOG_LEVEL 等日志配置项

# 【什么】日志目录：项目根/logs（本文件往上三级 = modules/common/logger.py → 项目根）
# 【作用】所有日志文件集中在一个目录，便于挂 volume 与统一清理
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"   # 拼出 <项目根>/logs 的绝对路径
_LOG_DIR.mkdir(parents=True, exist_ok=True)  # 递归创建目录；parents=True 补全中间层，exist_ok=True 避免已存在时报错

# 【什么】"是否已配置过"的开关
# 【作用】保证 logger.add 只跑一次；重复 add 同一目标会让每条日志写两遍
_configured = False                            # False = 尚未装配 sink；装配完成后置 True


# 【什么】装配 loguru 的三个 sink（控制台 + 全量文件 + 错误文件）
# 【作用】运行期第一次取 logger 时完成装配，之后直接复用
# 【为什么】sink 注册是全局副作用，必须集中在唯一入口执行并用开关防重，否则日志会重复输出或丢失
def _configure_logger():
    """配置 loguru 日志：控制台 + 文件轮转。"""
    global _configured                         # 声明要改模块级开关，否则函数内赋值会变成局部变量
    if _configured:                            # 已装配过就直接短路返回
        return                                 # 防重关键：重复 add 会让每条日志写两遍
    s = get_settings()                         # 读一次配置对象，后面取 log_level 用
    # 【先 remove】清掉 loguru 自带的默认 stderr handler，否则每条日志在控制台打两遍
    logger.remove()                            # 移除所有已注册 handler（含默认的），从干净状态开始装配

    logger.add(                                # 注册 sink 1：控制台彩色输出
        sys.stderr,          # 输出目标：标准错误流，不与 stdout 的 SSE 数据流混在一起
        level=s.log_level,   # 级别下限，取配置的 LOG_LEVEL
        format=(             # 控制台格式：带颜色标签，文件里不认这些标签所以用另一套
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "          # 时间戳段，绿色显示
            "<level>{level: <8}</level> | "                         # 级别段，左对齐占 8 字符宽
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"  # 模块:行号 - 消息正文
        ),
    )
    logger.add(                                # 注册 sink 2：全量日志文件，按天切分
        _LOG_DIR / "agent_{time:YYYY-MM-DD}.log",  # 全量日志，文件名按天分
        level=s.log_level,                         # 与控制台同一级别下限
        rotation="00:00",                          # 每天 0 点切新文件
        retention="7 days",                        # 全量日志保留 7 天，超时自动删
        compression="zip",                         # 切出的旧文件压缩成 zip 省磁盘
        encoding="utf-8",                          # 显式 utf-8，中文日志不乱码
        format=(                                   # 文件格式：不带颜色标签（写进文件会是乱码）
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "           # 时间戳 | 级别（定宽对齐）
            "{name}:{line} - {message}"                             # 模块:行号 - 消息正文
        ),
    )
    logger.add(                                # 注册 sink 3：错误日志文件，只收 ERROR 及以上
        _LOG_DIR / "error_{time:YYYY-MM-DD}.log",  # 错误日志：只收 ERROR 及以上
        level="ERROR",                             # 固定 ERROR，不受 LOG_LEVEL 影响
        rotation="00:00",                          # 每天 0 点切新文件
        retention="30 days",                       # 错误日志保留 30 天，比全量更久便于追溯
        compression="zip",                         # 同上，压缩归档
        encoding="utf-8",                          # 同上，中文不乱码
    )
    _configured = True                         # 装配完成置开关；之后任何调用都走前面的短路返回


# 【什么】取一个绑定了模块名的 logger
# 【作用】各模块用自己的名字取 logger，日志里可按模块名快速定位来源
# 【为什么】统一从这一个入口取 logger，保证"先装配、再使用"的顺序不会被破坏
def get_logger(name: str = "agent"):
    """获取带模块标识的 logger 实例。"""
    # 【先装配再 bind】首次调用时才真正 add sink，之后再调直接命中 _configured 短路
    _configure_logger()                        # 确保 sink 已装配；幂等，重复调用无副作用
    # 【bind 返回新对象】不改全局 logger，各模块拿到的实例互不干扰
    return logger.bind(module=name)            # 给日志记录挂上 module=<模块名> 字段后返回


# 【什么】导入本模块即完成装配
# 【作用】任何模块 `from ... import get_logger` 后拿到的 logger 都已装好 sink
_configure_logger()                            # 模块导入时立即装配一次，兜底保证日志可用
