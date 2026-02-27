import logging
import os
import sys
from logging.handlers import RotatingFileHandler


# ---------- COLOR DEFINITIONS ----------
class LogColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    WHITE = "\033[97m"
    BLUE = "\033[38;5;39m"
    GREEN = "\033[38;5;34m"
    YELLOW = "\033[38;5;226m"
    RED = "\033[38;5;196m"
    MAGENTA = "\033[38;5;201m"


class ColorFormatter(logging.Formatter):
    """
    Custom formatter to add colors based on log level.
    """

    FORMATS = {
        logging.DEBUG: LogColors.BLUE
        + "%(asctime)s - %(name)s - DEBUG - %(message)s"
        + LogColors.RESET,
        logging.INFO: LogColors.WHITE
        + "%(asctime)s - %(name)s - INFO - %(message)s"
        + LogColors.RESET,
        logging.WARNING: LogColors.YELLOW
        + "%(asctime)s - %(name)s - WARNING - %(message)s"
        + LogColors.RESET,
        logging.ERROR: LogColors.RED
        + "%(asctime)s - %(name)s - ERROR - %(message)s"
        + LogColors.RESET,
        logging.CRITICAL: LogColors.MAGENTA
        + LogColors.BOLD
        + "%(asctime)s - %(name)s - CRITICAL - %(message)s"
        + LogColors.RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(
            record.levelno,
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# ---------- MAIN LOGGER FUNCTION ----------
def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Returns a singleton logger with:
    - Colorful console logs
    - Rotating file logs stored in /logs directory
    """

    logger = logging.getLogger(name)
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    if not logger.handlers:
        logger.setLevel(level)

        # ---------- CREATE LOG DIRECTORY ----------
        project_root = os.getcwd()  # Your project base directory
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file_path = os.path.join(log_dir, "mmm_pipeline.log")

        # ---------- CONSOLE HANDLER (COLORFUL) ----------
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorFormatter())

        # ---------- FILE HANDLER (ROTATING) ----------
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=5,  # Keep last 5 log files
            encoding="utf-8",
        )

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Add both handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Avoid duplicate logs
        logger.propagate = False

    return logger
