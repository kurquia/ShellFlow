import logging


class CustomFormatter(logging.Formatter):
    # Define color codes
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    # Define log format
    FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

    def __init__(self):
        # Pass the format string to the parent class
        super().__init__(self.FORMAT)

    def format(self, record):
        # Add colors to the levelname based on log level
        if record.levelno == logging.ERROR:
            record.levelname = f"{self.RED}{record.levelname}{self.RESET}"
        elif record.levelno == logging.WARNING:
            record.levelname = f"{self.YELLOW}{record.levelname}{self.RESET}"
        elif record.levelno == logging.INFO:
            record.levelname = f"{self.GREEN}{record.levelname}{self.RESET}"
        elif record.levelno == logging.DEBUG:
            record.levelname = f"{self.BLUE}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(name, level=logging.INFO):
    """
    Create and configure a logger with color-coded output.

    Parameters:
    - name: str, name of the logger (default: __name__).
    - level: int, logging level (default: logging.INFO).

    Returns:
    - logger: logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # Avoid adding multiple handlers
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
    else:
        logger.setLevel(level)  # Update the level if the logger already exists
    return logger
