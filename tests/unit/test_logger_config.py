import logging

from shellflow import get_logger


def test_logger():
    logger = get_logger("test_logger")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.debug(
        "This is a debug message."
    )  # Ensure this doesn't show up if the level is INFO


def test_logging_levels():
    logger = get_logger("test_logger", level=logging.DEBUG)
    logger.debug("This debug message should appear.")
    logger.setLevel(logging.WARNING)
    logger.info("This info message should NOT appear.")
    logger.warning("This warning message should appear.")


if __name__ == "__main__":
    print("INFO MODE")
    test_logger()
    print("DEBUG & WARNING MODE")
    test_logging_levels()
