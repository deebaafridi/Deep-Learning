"""Centralized logging infrastructure for DualCNN.

Provides thread-safe logger initialization with simultaneous console streaming
and file output while preventing handler accumulation across multiple calls.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def get_logger(
    name: str = "solution",
    log_file: Optional[Union[str, Path]] = "solution.log",
    level: int = logging.INFO,
    format_str: str = "%(asctime)s [%(levelname)s] %(message)s",
) -> logging.Logger:
    """Configures and retrieves a centralized logger instance.

    Ensures that multiple invocations with the same logger name do not result
    in duplicate log messages by checking existing attached handlers.

    Args:
        name: Name identifier for the logger instance (defaults to 'solution').
        log_file: Filesystem path to an optional output log file. If None,
            file logging is skipped.
        level: Logging verbosity threshold (defaults to logging.INFO).
        format_str: Format template for log records.

    Returns:
        A configured logging.Logger instance ready for use across modules.

    Example:
        >>> logger = get_logger("trainer", log_file="train.log")
        >>> logger.info("Model initialization complete.")
    """
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handler registration if get_logger is called repeatedly
    if not logger.handlers:
        formatter = logging.Formatter(format_str)

        # 1. Console stream handler (writing to standard output)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

        # 2. File output handler (appending with UTF-8 encoding)
        if log_file:
            try:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
                file_handler.setFormatter(formatter)
                file_handler.setLevel(level)
                logger.addHandler(file_handler)
            except (OSError, IOError) as exc:
                logger.warning(f"Could not initialize file logger at '{log_file}': {exc}")

    return logger
