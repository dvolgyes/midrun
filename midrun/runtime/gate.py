"""Cooperative hold gate coordinating the Lightning training thread and REST.

The gate itself only tracks whether a hold is active and provides the events used to
wait for and signal release. Deciding *when* a hold should start is the command
scheduler's job (`runtime.commands`): a `hold` command becomes due at a safe point like
any other command, and its handler calls `RunGate.enter`.
"""

import threading
import time
from dataclasses import dataclass, replace

from midrun.runtime.commands.command_queue import CommandQueue


@dataclass(frozen=True)
class ActiveHold:
    """A hold currently blocking the training thread.

    :ivar command_id: The `hold` command whose execution entered this hold.
    :ivar safe_point: The safe point at which the hold was entered.
    :ivar epoch: Epoch at hold entry.
    :ivar step: Global step at hold entry.
    :ivar started_at: Unix timestamp of hold entry.
    :ivar joined_ids: The later holds that became due while this hold was active.
        The release of any of them, or of `command_id`, releases the run.
    """

    command_id: str
    safe_point: str
    epoch: int
    step: int
    started_at: float
    joined_ids: tuple[str, ...] = ()


class RunGate:
    """Tracks the active hold and the events used to wait for/signal release.

    :ivar _lock: Serializes gate state and control pause transitions.
    :ivar _active_hold: The hold blocking training, if any.
    :ivar _release_event: Signals that the active hold was released.
    :ivar _control_event: Wakes the training thread when control state changes.
    """

    _lock: threading.RLock
    _active_hold: ActiveHold | None
    _release_event: threading.Event
    _control_event: threading.Event

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_hold = None
        self._release_event = threading.Event()
        self._control_event = threading.Event()

    def hold_transition(self) -> threading.RLock:
        """Serializes hold dispatch with pause and continue requests."""
        return self._lock

    def request_pause(self, queue: CommandQueue) -> tuple[str, bool]:
        """Returns an active hold ID or arms a control pause."""
        with self._lock:
            if self._active_hold is not None:
                return self._active_hold.command_id, False
            command, queued = queue.submit_pause()
        self.notify()
        return command.id, queued

    def continue_run(self, queue: CommandQueue) -> tuple[bool, list[str]]:
        """Cancels a queued pause and releases the active hold atomically."""
        with self._lock:
            cancelled = queue.cancel_pause()
            return self.release(), cancelled

    def enter(
        self, command_id: str, safe_point: str, epoch: int, step: int
    ) -> ActiveHold:
        """Marks the gate held, or joins `command_id` to the active hold.

        Must be called from the training thread.

        Returns:
            The active hold. Its `command_id` is not `command_id` if this call joined it.
        """
        hold = ActiveHold(
            command_id=command_id,
            safe_point=safe_point,
            epoch=epoch,
            step=step,
            started_at=time.time(),
        )
        with self._lock:
            if self._active_hold is not None:
                hold = replace(
                    self._active_hold,
                    joined_ids=(*self._active_hold.joined_ids, command_id),
                )
            self._active_hold = hold
            self._release_event.clear()
        return hold

    def is_held(self) -> bool:
        """Returns whether the gate currently blocks training progress."""
        with self._lock:
            return self._active_hold is not None

    def active_hold(self) -> ActiveHold | None:
        """Returns the active hold, or None."""
        with self._lock:
            return self._active_hold

    def release(self, *, command_id: str | None = None) -> bool:
        """Releases the active hold if the supplied primary or joined ID matches.

        Returns:
            True if released. Without an ID, releases any active hold.
        """
        with self._lock:
            hold = self._active_hold
            if hold is None or (
                command_id is not None
                and command_id != hold.command_id
                and command_id not in hold.joined_ids
            ):
                return False
            self._active_hold = None
            self._release_event.set()
            self._control_event.set()
            return True

    def notify(self) -> None:
        """Wakes a training thread blocked in `wait_for_control_event` (e.g. new command)."""
        self._control_event.set()

    def wait_for_control_event(self, timeout: float) -> None:
        """Blocks the calling thread until notified or `timeout` elapses."""
        self._control_event.wait(timeout)
        self._control_event.clear()
