import logging
from datetime import datetime


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[37m",  # White
        logging.INFO: "\033[34m",  # Blue
        logging.WARNING: "\033[93m",  # Orange
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        log_time = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        message = f"{log_time} [{record.levelname}] {record.name}: {record.getMessage()}"

        return f"{color}{message}{self.RESET}"


class Logger:
    def __init__(
        self,
        name: str,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(ColorFormatter())
            self.logger.addHandler(handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
