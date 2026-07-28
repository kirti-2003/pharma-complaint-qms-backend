import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    """
    Configure application-wide console logging.

    DEBUG logs are enabled during development.
    INFO logs are used when debug mode is disabled.
    """

    log_level = logging.DEBUG if settings.debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger for a module.

    Example:
        logger = get_logger(__name__)
    """

    return logging.getLogger(name)