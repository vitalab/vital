import logging
from pathlib import Path
from typing import Union


def configure_logging(
    level: Union[int, str] = logging.WARNING, filename: Path = None, print_to_console: bool = True
) -> None:
    """Configures a standardized way of logging for the library.

    Args:
        level: Minimal level of events to log.
        filename: Path to the file loggers should write to, if wanted.
        print_to_console: Whether the loggers should display the messages to the console.
    """
    handlers = []
    # No additional handler is set in case of `print_to_console`
    # because `basicConfig` logs to console by default
    # Adding a `logging.StreamHandler()` will only cause duplicated logs
    # TODO Find a way to disable `StreamHandler` only (when `print_to_console=False`)

    if filename:
        handlers.append(logging.FileHandler(str(filename), mode="w"))

    logging.basicConfig(level=level, handlers=handlers)
