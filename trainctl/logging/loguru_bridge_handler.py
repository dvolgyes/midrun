"""A logging handler that forwards standard-library records into Loguru."""

import inspect
import logging
import threading

from loguru import logger

# Standard-library level names that Loguru also defines, with the same numbers.
_SHARED_LEVEL_NAMES = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class LoguruBridgeHandler(logging.Handler):
    """Forwards one standard-library logger's records into Loguru as one session.

    Guards against reentrancy: a Loguru sink that itself logs through a bridged
    standard-library logger would otherwise recurse forever; the guard drops that
    nested call instead of forwarding it.

    A standard level keeps its name (`INFO`, not `Level 20`), and the Loguru record
    names the caller that the standard record names (so `stacklevel` is honored).
    A record whose caller frame is not on the current stack (for example, one that a
    queue listener handles) keeps this handler as its caller.

    Attributes:
        session_id: The owning `LoggingSession`'s identity, used to detect an
            overlapping install onto an already-bridged logger.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(level=logging.NOTSET)
        self.session_id = session_id
        self._guard = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        """Forwards `record` into Loguru, unless this thread is already forwarding."""
        if getattr(self._guard, "active", False):
            return
        self._guard.active = True
        try:
            bound = logger.bind(
                trainctl_session=self.session_id,
                std_logger_name=record.name,
                std_pathname=record.pathname,
                std_lineno=record.lineno,
                std_funcname=record.funcName,
            )
            level = (
                record.levelname
                if record.levelname in _SHARED_LEVEL_NAMES
                else record.levelno
            )
            caller = (record.pathname, record.lineno, record.funcName)
            frame, depth = inspect.currentframe(), 0
            while frame is not None and caller != (
                frame.f_code.co_filename,
                frame.f_lineno,
                frame.f_code.co_name,
            ):
                frame, depth = frame.f_back, depth + 1
            if frame is None:
                depth = 0
            del frame
            bound.opt(exception=record.exc_info, depth=depth).log(
                level, record.getMessage()
            )
        finally:
            self._guard.active = False
