import logging
from functools import cache

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


@cache
def get_logger(name: str = "talhao") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = RichHandler(
        console=_console,
        show_time=True,
        show_level=True,
        show_path=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%Y-%m-%d %H:%M:%S]"))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    return logger
