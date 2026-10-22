"""Installs a logging session bridge that includes the root logger."""

from trainctl.logging.logging_session import BRIDGED_LOGGERS, LoggingSession


def install_root_bridge(session: LoggingSession) -> None:
    """Installs `session`'s bridge including the actual root logger.

    For applications that want *all* standard-library logging -- not just Lightning's
    namespaces -- unified into the same run file. Call this instead of
    `session.install()`, not in addition to it: `install` is a no-op once a session
    already holds a handler, so a later call would silently skip adding root.
    """
    session.install(logger_names=(*BRIDGED_LOGGERS, "root"))
