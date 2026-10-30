"""Drains one child-process pipe without letting it fill."""

import codecs
import threading
from typing import Any

_READ_CHUNK_SIZE = 4096


class BoundedDrain:
    """Drains one pipe to EOF on its own thread, streaming its lines to Loguru.

    Streams at most the first `max_bytes` to the log, one record per line, then one
    line that counts the bytes it did not log. A line that the limit cuts is logged
    up to the limit. Retains the last `max_bytes` as the tail; keeps draining past
    that limit (setting `truncated`) so the child's pipe never fills and deadlocks it.
    """

    _tail: bytearray
    _decoder: codecs.IncrementalDecoder
    _partial_line: str

    def __init__(self, stream: Any, *, tag: str, max_bytes: int, log: Any) -> None:
        self._stream = stream
        self._tag = tag
        self._max_bytes = max_bytes
        self._log = log
        self._tail = bytearray()
        self._total = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._partial_line = ""
        self.truncated = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Starts the drain thread."""
        self._thread.start()

    def join(self, *, timeout: float | None = None) -> bool:
        """Waits up to `timeout` seconds (`None`: forever) for the stream to hit EOF.

        Returns:
            Whether the drain thread has finished.
        """
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def tail(self) -> str:
        """Returns the retained (possibly truncated) decoded tail."""
        return bytes(self._tail).decode("utf-8", errors="replace")

    def _run(self) -> None:
        try:
            # `read1` returns what the pipe has now; `read` would wait for a full chunk.
            for chunk in iter(lambda: self._stream.read1(_READ_CHUNK_SIZE), b""):
                self._record(chunk)
        finally:
            self._stream.close()
        self._log_lines(self._decoder.decode(b"", final=True), final=True)
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
            limit_reached = len(chunk) >= unlogged
            text = self._decoder.decode(chunk[:unlogged], final=limit_reached)
            self._log_lines(text, final=limit_reached)
        self._total += len(chunk)
        self._tail += chunk
        if len(self._tail) > self._max_bytes:
            del self._tail[: len(self._tail) - self._max_bytes]
            self.truncated = True

    def _log_lines(self, text: str, *, final: bool) -> None:
        lines = (self._partial_line + text).split("\n")
        self._partial_line = lines.pop()
        if final and self._partial_line:
            lines.append(self._partial_line)
            self._partial_line = ""
        for line in lines:
            self._log.info("[hook {}] {}", self._tag, line)
