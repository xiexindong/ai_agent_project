"""
日志封装模块
============
基于 loguru 实现全链路日志：控制台彩色输出 + 文件轮转持久化。
支持按模块隔离日志、自动捕获未处理异常。
"""
import sys
from pathlib import Path
from loguru import logger
from config import get_settings

# 【什么】日志目录：项目根/logs（本文件往上三级 = modules/common/logger.py → 项目根）
# 【作用】所有日志文件集中在一个目录，便于挂 volume 与统一清理
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# 【什么】"是否已配置过"的开关
# 【作用】保证 logger.add 只跑一次；重复 add 同一目标会让每条日志写两遍
_configured = False


# 【什么】装配 loguru 的三个 sink（控制台 + 全量文件 + 错误文件）
# 【作用】运行期第一次取 logger 时完成装配，之后直接复用
def _configure_logger():
    """配置 loguru 日志：控制台 + 文件轮转。"""
    global _configured
    if _configured:
        return
    s = get_settings()
    # 【先 remove】清掉 loguru 自带的默认 stderr handler，否则每条日志在控制台打两遍
    logger.remove()

    logger.add(
        sys.stderr,          # 输出目标：标准错误流，不与 stdout 的 SSE 数据流混在一起
        level=s.log_level,   # 级别下限，取配置的 LOG_LEVEL
        format=(             # 控制台格式：带颜色标签，文件里不认这些标签所以用另一套
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )
    logger.add(
        _LOG_DIR / "agent_{time:YYYY-MM-DD}.log",  # 全量日志，文件名按天分
        level=s.log_level,                         # 与控制台同一级别下限
        rotation="00:00",                          # 每天 0 点切新文件
        retention="7 days",                        # 全量日志保留 7 天，超时自动删
        compression="zip",                         # 切出的旧文件压缩成 zip 省磁盘
        encoding="utf-8",                          # 显式 utf-8，中文日志不乱码
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{line} - {message}"
        ),
    )
    logger.add(
        _LOG_DIR / "error_{time:YYYY-MM-DD}.log",  # 错误日志：只收 ERROR 及以上
        level="ERROR",                             # 固定 ERROR，不受 LOG_LEVEL 影响
        rotation="00:00",                          # 每天 0 点切新文件
        retention="30 days",                       # 错误日志保留 30 天，比全量更久便于追溯
        compression="zip",                         # 同上，压缩归档
        encoding="utf-8",                          # 同上，中文不乱码
    )
    _configured = True


# 【什么】取一个绑定了模块名的 logger
# 【作用】各模块用自己的名字取 logger，日志里可按模块名快速定位来源
def get_logger(name: str = "agent"):
    """获取带模块标识的 logger 实例。"""
    # 【先装配再 bind】首次调用时才真正 add sink，之后再调直接命中 _configured 短路
    _configure_logger()
    # 【bind 返回新对象】不改全局 logger，各模块拿到的实例互不干扰
    return logger.bind(module=name)


# 【什么】导入本模块即完成装配
# 【作用】任何模块 `from ... import get_logger` 后拿到的 logger 都已装好 sink
_configure_logger()
