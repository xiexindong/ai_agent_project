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

# ---- 日志目录 ----
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_configured = False


def _configure_logger():
    """配置 loguru 日志：控制台 + 文件轮转。"""
    global _configured
    if _configured:
        return
    s = get_settings()
    logger.remove()  # 清除默认handler

    # 控制台输出：彩色、精简
    logger.add(
        sys.stderr,
        level=s.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )
    # 文件输出：按天轮转，保留7天
    logger.add(
        _LOG_DIR / "agent_{time:YYYY-MM-DD}.log",
        level=s.log_level,
        rotation="00:00",       # 每天午夜轮转
        retention="7 days",     # 保留7天
        compression="zip",      # 自动压缩
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{line} - {message}"
        ),
    )
    # 错误日志单独文件，方便快速定位
    logger.add(
        _LOG_DIR / "error_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )
    _configured = True


def get_logger(name: str = "agent"):
    """获取带模块标识的 logger 实例。"""
    _configure_logger()
    return logger.bind(module=name)


# 启动时自动配置
_configure_logger()
