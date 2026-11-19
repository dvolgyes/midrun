"""Cross-thread hand-off for cheap, on-demand live views computed on the training thread.

Purpose-built for lazy FUSE files (see `midrun.fuse.filesystem`): a reader thread
blocks in `LiveCaptureBroker.request` until the training thread calls `serve` from
inside a safe point, computes the callable there (where `current_step`/`pl_module` are
valid), and wakes the caller. Unlike `Command`, a live capture is never persisted to
`ArtifactStore` and never appears in `GET /commands` -- it is a transient read, not an
audited action.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING, Literal, cast

from loguru import logger

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule


@dataclass(eq=False)
class _PendingCapture:
    """One request and its synchronized lifecycle.

    :ivar fn: Computation to run on the training thread.
    :ivar deadline: Latest time the training thread may claim this request.
    :ivar event: Signals that the computation finished or the request expired.
    :ivar result: Computation result, if successful.
    :ivar error: Callback exception transferred to the requesting thread.
    :ivar state: `pending` until claimed or expired; a claimed callback runs to completion.
    """

    fn: Callable[["LightningModule"], object]
    deadline: float
    event: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: BaseException | None = None
    state: Literal["pending", "claimed", "expired"] = "pending"


class LiveCaptureBroker:
    """Thread-safe hand-off of one-shot computations to the training thread.

    :ivar _lock: Serializes request admission, claim, and expiry.
    :ivar _pending: Requests not yet detached by the training thread.
    """

    _lock: threading.Lock
    _pending: list[_PendingCapture]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending = []

    def request[T](self, fn: Callable[["LightningModule"], T], timeout: float) -> T:
        """Blocks until `fn(pl_module)` runs on the training thread, or `timeout` elapses.

        Args:
            fn: Called with `pl_module` from the training thread at the next safe point
                `serve` is invoked for. Any exception it raises is re-raised here, in
                the calling thread.
            timeout: Seconds to wait for the result.

        Returns:
            `fn`'s return value.

        Raises:
            TimeoutError: The reader's deadline elapsed. A callback already claimed
                by the training thread can still finish after the reader times out.
        """
        pending = _PendingCapture(fn=fn, deadline=monotonic() + timeout)
        with self._lock:
            self._pending.append(pending)
        if not pending.event.wait(max(0.0, pending.deadline - monotonic())):
            with self._lock:
                if pending.state == "claimed":
                    raise TimeoutError(
                        "live capture timed out after the training thread claimed it"
                    )
                pending.state = "expired"
                if pending in self._pending:
                    self._pending.remove(pending)
        if pending.state == "expired":
            raise TimeoutError(
                "live capture timed out before the training thread claimed it"
            )
        if pending.error is not None:
            raise pending.error
        return cast("T", pending.result)

    def serve(self, pl_module: "LightningModule") -> None:
        """Runs and wakes every pending request. Call only from the training thread."""
        with self._lock:
            batch, self._pending = self._pending, []
        for item in batch:
            with self._lock:
                if item.state == "expired" or monotonic() >= item.deadline:
                    item.state = "expired"
                    item.event.set()
                    continue
                item.state = "claimed"
            try:
                item.result = item.fn(pl_module)
            except Exception as exc:  # noqa: BLE001 -- boundary: a live-view failure must not crash training
                logger.warning("midrun live capture failed: {}", exc)
                item.error = exc
            finally:
                item.event.set()
