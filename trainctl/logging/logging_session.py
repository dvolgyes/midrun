"""The standard-logging bridge and Loguru run-file sink of one training run."""

from pathlib import Path
from typing import Any

from loguru import logger

from trainctl.logging.loguru_bridge_handler import LoguruBridgeHandler
from trainctl.logging.resolve_std_logger import resolve_std_logger
from trainctl.logging.saved_logger_state import SavedLoggerState

BRIDGED_LOGGERS: tuple[str, ...] = (
    "lightning.pytorch",
    "pytorch_lightning",
    "lightning.fabric",
)


class LoggingSession:
    """Owns one training run's standard-logging bridge and its Loguru run-file sink.

    Detaches the bridged loggers' own handlers so nothing double-logs, forwards their
    records into Loguru bound with `session_id`, and (optionally) attaches a Loguru file
    sink filtered to that same `session_id` -- so a later, unrelated session in the same
    process cannot leak into this run's file. `install`/`close` are each idempotent.

    `install` never changes a bridged logger's own level: Python's logging module
    computes a message's effective threshold (walking to the nearest ancestor with a
    set level) independently of `propagate`, so a bridged logger still needs its own
    level set below a message's severity for that message to reach this bridge at all.
    Lightning sets its own loggers to `INFO` at import time; this class respects that
    policy rather than overriding it.

    Attributes:
        session_id: This session's identity, bound onto every forwarded record and the
            run-file sink's filter.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._saved: list[SavedLoggerState] = []
        self._handler: LoguruBridgeHandler | None = None
        self._file_sink_id: int | None = None

    def install(self, logger_names: tuple[str, ...] = BRIDGED_LOGGERS) -> None:
        """Detaches `logger_names`' own handlers and installs the Loguru bridge.

        A second call on an already-installed session is a no-op. Installing onto a
        logger already bridged by a *different* session is rejected outright --
        detected from the handler instance already present, not a separate registry --
        so overlapping sessions can never silently shadow one another.

        Args:
            logger_names: Standard-library logger names to bridge; `"root"` refers to
                the actual root logger (see `resolve_std_logger`).

        Raises:
            RuntimeError: A logger in `logger_names` is already bridged by a different,
                still-open `LoggingSession`.
        """
        if self._handler is not None:
            return
        # Names can repeat or alias one logger ("" and "root"): save each logger once,
        # or a second save records the bridged state and close() restores that.
        std_loggers = {resolve_std_logger(name): name for name in logger_names}
        for std_logger, name in std_loggers.items():
            for existing in std_logger.handlers:
                if isinstance(existing, LoguruBridgeHandler):
                    raise RuntimeError(
                        f"logger {name!r} is already bridged by session "
                        f"{existing.session_id!r}; close that session first"
                    )
        handler = LoguruBridgeHandler(self.session_id)
        for std_logger in std_loggers:
            self._saved.append(SavedLoggerState(std_logger))
            for existing in list(std_logger.handlers):
                std_logger.removeHandler(existing)
            std_logger.addHandler(handler)
            std_logger.propagate = False
        self._handler = handler

    def attach_run_file(self, path: Path) -> None:
        """Adds a Loguru file sink at `path`, filtered to this session's own records.

        A second call is a no-op. I/O failures (e.g. an unwritable parent directory)
        propagate as-is from `Path.mkdir`/`loguru.logger.add` -- the caller decides
        whether that is fatal.
        """
        if self._file_sink_id is not None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        session_id = self.session_id
        self._file_sink_id = logger.add(
            str(path),
            filter=lambda record: record["extra"].get("trainctl_session") == session_id,
            level="DEBUG",
            diagnose=False,
            backtrace=False,
        )

    def bind(self) -> Any:
        """Returns a Loguru logger bound to this session's id.

        Trainctl code that wants its own diagnostics routed into this session's run
        file (`attach_run_file`) must log through this bound logger rather than the
        bare module-level `loguru.logger`.
        """
        return logger.bind(trainctl_session=self.session_id)

    def close(self) -> None:
        """Restores every intercepted logger and removes the run-file sink.

        Idempotent -- safe to call from both the normal-teardown and the
        `on_exception` paths without double-restoring or erroring on a second call.
        """
        for saved in self._saved:
            saved.restore()
        self._saved = []
        self._handler = None
        if self._file_sink_id is not None:
            logger.remove(self._file_sink_id)
            self._file_sink_id = None
