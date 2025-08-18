from pathlib import Path
import time
from loguru import logger

configured_identifiers = set()
g_flag_init = False

def configure_logger(
    identifier="default",
    level="DEBUG",
    rotation="10000 MB",
    retention="10 days",
    parent_dir=None,
):
    """
    配置日志记录器，支持多标识符隔离和单例模式

    Args:
        identifier: 日志标识符（用于区分不同日志文件）
        level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        rotation: 日志轮转条件（大小或时间）
        retention: 日志保留策略
        parent_dir: 自定义日志父目录（可选）
    """
    global configured_identifiers
    global g_flag_init
    # 路径处理
    project_path = Path.cwd() if parent_dir is None else Path(parent_dir)
    base_log_path = project_path / "logs" / identifier
    base_log_path.mkdir(parents=True, exist_ok=True)

    # 生成当天日志文件名（使用Path保证跨平台兼容性）
    log_file = base_log_path / f"{time.strftime('%Y-%m-%d')}.log"
    
    if not g_flag_init:
        # 总日志文件路径
        total_log_path = project_path / "logs" / "total"
        total_log_path.mkdir(parents=True, exist_ok=True)
        total_log_file = total_log_path / f"{time.strftime('%Y-%m-%d')}.log"
        logger.add(
            sink=str(total_log_file),
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            level=level,
        )
        g_flag_init = True
    # 保证每个identifier只添加一次handler
    if identifier not in configured_identifiers:
        try:
            # 添加日志处理器
            logger.add(
                sink=str(log_file),
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
                level=level,
                filter=lambda record: record["extra"].get("identifier") == identifier,
            )
            
            configured_identifiers.add(identifier)
        except Exception as e:
            logger.error(f"Failed to configure logger: {str(e)}")
            raise

    # 返回绑定后的日志记录器
    return logger.bind(identifier=identifier)
