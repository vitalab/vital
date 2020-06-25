import logging
from pathlib import Path
from typing import Union


def configure_logging(level: Union[int, str] = logging.WARNING,
                      filename: Path = None, print_to_console: bool = True):
    """Configures a standardized way of logging for the library.

    Args:
        level: minimal level of events to log.
        filename: path to the file loggers should write to, if wanted.
        print_to_console: whether the loggers should display the messages to the console.
    """
    handlers = []
    if filename:
        handlers.append(logging.FileHandler(str(filename), mode='w'))
    if print_to_console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, handlers=handlers)
