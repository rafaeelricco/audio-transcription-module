import logging
import threading
from typing import Optional


class Logger:
    """
    Thread-safe logger with support for multiple output formats.

    This logger wraps Python's built-in logging module and adds features
    like success/failure indicators and verbosity control.
    """

    VERBOSE = False
    _lock = threading.Lock()
    _logger = None

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        """Get or create the logger instance."""
        if cls._logger is None:
            with cls._lock:
                if cls._logger is None:
                    cls._logger = logging.getLogger("audio-to-text")
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter("%(message)s")
                    handler.setFormatter(formatter)
                    cls._logger.addHandler(handler)
                    cls._logger.setLevel(logging.INFO)
        return cls._logger

    @classmethod
    def set_verbose(cls, verbose: bool) -> None:
        """
        Set the verbosity level for the logger.

        Args:
            verbose (bool): If True, debug messages will be shown
        """
        with cls._lock:
            cls.VERBOSE = verbose
            logger = cls._get_logger()
            logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    @classmethod
    def log(cls, success: bool, message: str, level: str = "info") -> None:
        """
        Log a message with appropriate formatting and level.

        Args:
            success (bool): Whether the operation was successful
            message (str): The message to log
            level (str): Log level - one of: info, debug, warning, error
        """
        if level == "debug" and not cls.VERBOSE:
            return

        prefix = "✓" if success else "✗"

        level_prefixes = {
            "debug": "[DEBUG]",
            "warning": "[WARNING]",
            "error": "[ERROR]",
            "info": "",
        }

        level_prefix = level_prefixes.get(level, "")
        formatted_message = f"{prefix} {level_prefix} {message}"

        logger = cls._get_logger()

        with cls._lock:
            if level == "error":
                logger.error(formatted_message)
            elif level == "warning":
                logger.warning(formatted_message)
            elif level == "debug":
                logger.debug(formatted_message)
            else:
                logger.info(formatted_message)

    @classmethod
    def setup_file_logging(
        cls, log_file: str, format_string: Optional[str] = None
    ) -> None:
        """
        Set up logging to a file in addition to the console.

        Args:
            log_file (str): Path to the log file
            format_string (str, optional): Custom format string for log messages
        """
        with cls._lock:
            logger = cls._get_logger()

            file_handler = logging.FileHandler(log_file, encoding="utf-8")

            if format_string is None:
                format_string = "%(asctime)s - %(levelname)s - %(message)s"

            formatter = logging.Formatter(format_string)
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
