"""
Central logging configuration and debug decorator.

Provides structured logging with file and console handlers, plus a decorator
for automatic function-level observability.
"""

import functools
import logging
import traceback
from pathlib import Path
from time import time
from typing import Any, Callable, TypeVar

# Type variable for function return types
F = TypeVar("F", bound=Callable[..., Any])

# Log file path
LOG_FILE = Path(__file__).resolve().parent.parent / "system_debug.log"

# Configure root logger
_logger = logging.getLogger("profitability_engine")
_logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers
if not _logger.handlers:
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(module)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    _logger.addHandler(console_handler)

    # File handler (DEBUG level)
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(module)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    _logger.addHandler(file_handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Optional module name. If None, uses the caller's module.

    Returns:
        Logger instance configured with file and console handlers.
    """
    if name:
        return logging.getLogger(f"profitability_engine.{name}")
    return _logger


def debug_watcher(func: F) -> F:
    """
    Decorator that logs function entry, execution time, and exceptions.

    Logs:
    - Function start with arguments
    - Function completion with execution time
    - Full traceback on exceptions (to file only)

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function with logging.
    """
    logger = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__name__
        start_time = time()

        # Log function entry with arguments
        args_str = ", ".join([str(arg)[:100] for arg in args[:3]])  # Limit arg display
        kwargs_str = ", ".join([f"{k}={str(v)[:50]}" for k, v in list(kwargs.items())[:3]])
        params_str = ", ".join(filter(None, [args_str, kwargs_str]))
        logger.info(f"Starting {func_name}... ({params_str})")

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Log successful completion
            elapsed = time() - start_time
            logger.info(f"Completed {func_name} in {elapsed:.3f} seconds.")

            return result

        except Exception as e:
            # Log full traceback to file (DEBUG level)
            elapsed = time() - start_time
            error_msg = f"Exception in {func_name} after {elapsed:.3f} seconds: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Full traceback for {func_name}:\n{traceback.format_exc()}")

            # Re-raise to maintain normal error handling
            raise

    return wrapper  # type: ignore[return-value]
