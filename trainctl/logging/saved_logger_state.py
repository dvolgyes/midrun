"""The saved handlers, propagate flag and level of a standard-library logger."""

import logging


class SavedLoggerState:
    """A standard-library logger's handlers/propagate/level, saved for restoration."""

    def __init__(self, std_logger: logging.Logger) -> None:
        self.std_logger = std_logger
        self.handlers = list(std_logger.handlers)
        self.propagate = std_logger.propagate
        self.level = std_logger.level

    def restore(self) -> None:
        """Reinstates the exact handlers/propagate/level captured at save time."""
        self.std_logger.handlers = list(self.handlers)
        self.std_logger.propagate = self.propagate
        self.std_logger.setLevel(self.level)
