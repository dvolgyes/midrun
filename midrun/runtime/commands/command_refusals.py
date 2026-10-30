"""The thread-safe set of command kinds that the command queue refuses."""

import threading


class CommandRefusals:
    """Thread-safe record of the refused command kinds and the reason for each.

    A stop refuses later holds; a new run or stage clears the refusals.

    Attributes:
        _lock: Guards `_reasons`.
        _reasons: The refusal reason of each refused kind.
    """

    _lock: threading.Lock
    _reasons: dict[str, str]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reasons = {}

    def refuse(self, kind: str, reason: str) -> None:
        """Refuses `kind` until `clear`; a later reason replaces an earlier one."""
        with self._lock:
            self._reasons[kind] = reason

    def clear(self) -> None:
        """Accepts again every kind that `refuse` refused."""
        with self._lock:
            self._reasons.clear()

    def reason(self, kind: str) -> str | None:
        """Returns why `kind` is refused, or None if it is accepted."""
        with self._lock:
            return self._reasons.get(kind)
