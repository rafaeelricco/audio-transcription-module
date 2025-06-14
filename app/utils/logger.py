import logging
import threading

from typing import Optional


class Logger:
    """
    Thread-safe logger with support for multiple output formats.

    This class provides a singleton logging interface that wraps Python's built-in
    logging module with additional features such as thread safety, visual success/failure
    indicators, and verbosity control. It supports both console and file output.

    This logger is designed for use throughout the application to provide consistent
    log formatting and behavior across all modules.
    """

    VERBOSE = False
    _lock = threading.Lock()
    _logger = None

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        """
        Get or create the logger instance.

        Creates the logger if it doesn't exist yet, following the singleton pattern.
        This internal method handles the thread-safe initialization of the logger.

        Returns:
            logging.Logger: The configured logger instance
        """
        if cls._logger is None:
            with cls._lock:
                if cls._logger is None:
                    cls._logger = logging.getLogger("audio-to-text")
                    cls._logger.setLevel(logging.ERROR)
        return cls._logger

    @classmethod
    def set_verbose(cls, verbose: bool) -> None:
        """
        Set the verbosity level for the logger.

        Controls whether debug-level messages are displayed. When verbose is True,
        debug messages are shown; otherwise, they are suppressed.

        Args:
            verbose (bool): If True, debug messages will be shown; if False,
                           only info level and higher messages will be shown
        """
        with cls._lock:
            cls.VERBOSE = verbose
            logger = cls._get_logger()
            logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    @classmethod
    def log(cls, success: bool, message: str, level: str = "info") -> None:
        """
        Log a message with appropriate formatting and level.

        Logs a message with visual indicators for success/failure status and
        proper log level formatting. Messages are automatically prefixed with
        symbols: ✓ for success and ✗ for failure.

        Args:
            success (bool): Whether the operation was successful (affects the prefix symbol)
            message (str): The message text to log
            level (str): Log level - one of: 'info', 'debug', 'warning', 'error'.
                        Debug messages are only shown when verbose mode is enabled.
        """
        # Skip debug messages if not in verbose mode
        if level == "debug" and not cls.VERBOSE:
            return

        # Choose prefix based on success status
        if level == "error":
            prefix = "✗"
        elif success:
            prefix = "✓"
        else:
            prefix = "✗"

        # Format message with level and prefix
        if level == "error":
            formatted_message = f"{prefix} [ERROR] {message}"
        elif level == "warning":
            formatted_message = f"{prefix} [WARNING] {message}"
        elif level == "debug":
            formatted_message = f"{prefix} [DEBUG] {message}"
        else:  # info
            formatted_message = f"{prefix} {message}"

        # Print to console immediately for user feedback
        print(formatted_message)

        # Also log to the internal logger if needed
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

        Configures the logger to output messages to both the console and a specified
        file, optionally with a custom format string.

        Args:
            log_file (str): Path to the log file where messages will be written
            format_string (str, optional): Custom format string for log messages.
                                          Defaults to a timestamp-prefixed format if None.
        """
        with cls._lock:
            logger = cls._get_logger()

            file_handler = logging.FileHandler(log_file, encoding="utf-8")

            if format_string is None:
                format_string = "%(asctime)s - %(levelname)s - %(message)s"

            formatter = logging.Formatter(format_string)
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
