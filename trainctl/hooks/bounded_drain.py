"""Drains one child-process pipe without letting it fill."""

import threading
from typing import Any

_READ_CHUNK_SIZE = 4096


class BoundedDrain:
    """Drains one pipe to EOF on its own thread, streaming chunks to Loguru.

    Streams at most the first `max_bytes` to the log, then one line that counts the
    bytes it did not log. Retains the last `max_bytes` as the tail; keeps draining past
    that limit (setting `truncated`) so the child's pipe never fills and deadlocks it.
    """

    _tail: bytearray

    def __init__(self, stream: Any, *, tag: str, max_bytes: int, log: Any) -> None:
        self._stream = stream
        self._tag = tag
        self._max_bytes = max_bytes
        self._log = log
        self._tail = bytearray()
        self._total = 0
        self.truncated = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Starts the drain thread."""
        self._thread.start()

    def join(self) -> None:
        """Waits for the drain thread to finish (the stream has hit EOF)."""
        self._thread.join()

    def tail(self) -> str:
        """Returns the retained (possibly truncated) decoded tail."""
        return bytes(self._tail).decode("utf-8", errors="replace")

    def _run(self) -> None:
        try:
            for chunk in iter(lambda: self._stream.read(_READ_CHUNK_SIZE), b""):
                self._record(chunk)
        finally:
            self._stream.close()
        if self.truncated:
            self._log.info(
                "[hook {}] {} more bytes not logged (hooks_max_output_bytes={})",
                self._tag,
                self._total - self._max_bytes,
                self._max_bytes,
            )

    def _record(self, chunk: bytes) -> None:
        unlogged = max(self._max_bytes - self._total, 0)
        if unlogged:
            self._log.info(
                "[hook {}] {}",
                self._tag,
                chunk[:unlogged].decode("utf-8", errors="replace"),
            )
        self._total += len(chunk)
        self._tail += chunk
        if len(self._tail) > self._max_bytes:
            del self._tail[: len(self._tail) - self._max_bytes]
            self.truncated = True
