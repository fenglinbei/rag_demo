from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import AppConfig


def setup_logging(config: AppConfig) -> None:
    """Configure root logging for both console and file outputs."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    formatter = logging.Formatter(log_format)
    root_logger.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    root_logger.addHandler(stream_handler)

    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
