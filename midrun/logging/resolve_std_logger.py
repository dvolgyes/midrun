"""Resolves a logger name to a standard-library logger."""

import logging


def resolve_std_logger(name: str) -> logging.Logger:
    """Resolves `name` to a standard-library logger, special-casing `"root"`.

    `logging.getLogger("root")` is *not* the actual root logger (it is an ordinary
    child logger named `"root"`); only `logging.getLogger()` with no argument returns
    it. `"root"` is this module's sentinel name for that logger, matching the same
    convention used by `logging.config`'s dictConfig schema.
    """
    return logging.getLogger() if name == "root" else logging.getLogger(name)
