import logging
import os
from logging.handlers import RotatingFileHandler
from app.config import Config


def setup_logging(log_path=None, force=False, max_bytes=None):
    """
    Configure logging with rotation.

    Args:
        log_path (str, optional): Path to log file. Defaults to Config.LOG_PATH.
        force (bool): If True, clear existing handlers.
        max_bytes (int, optional): Max file size before rotation. Defaults to Config.MAX_LOG_SIZE.

    Returns:
        logging.Logger: Configured logger.
    """
    log_path = log_path or Config.LOG_PATH
    logger = logging.getLogger('revenue_predictor')
    logger.setLevel(logging.INFO)

    if force or not logger.handlers:
        if force:
            logger.handlers.clear()
            print(f"setup_logging: Cleared existing handlers for: {log_path}")
        log_dir = os.path.dirname(log_path) or '.'
        print(f"setup_logging: Creating directory: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)
        print(f"setup_logging: Using log path: {log_path}")
        try:
            if not os.path.exists(log_path):
                print(f"setup_logging: Touching log file: {log_path}")
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write('')
            print(f"setup_logging: Initializing handler for: {log_path}")
            max_bytes = max_bytes if max_bytes is not None else getattr(
                Config, 'MAX_LOG_SIZE', 1024 * 1024)
            handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=5,
                encoding='utf-8'
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            print(
                f"setup_logging: File handler added for: {log_path} with maxBytes={max_bytes}")
            logger.info(f"setup_logging: Logging initialized to: {log_path}")
            handler.flush()
        except Exception as e:
            print(
                f"setup_logging: Failed to initialize logging to {log_path}: {str(e)}")
            raise
    else:
        print(f"setup_logging: Logger already has handlers, using: {log_path}")
    return logger
