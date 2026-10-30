"""Runs a Python snippet inside a live process through gdb call-injection."""

from typing import Any

from midrun.debugger.gdb_capture.schedule_pending_call_and_wait import (
    schedule_pending_call_and_wait,
)
from midrun.debugger.hold_client import HoldClient


def gdb_capture(
    pid: int,
    python_snippet: str,
    *,
    poll_timeout_s: float = 10.0,
    max_attempts: int = 20,
    gdb_bin: str = "gdb",
    hold_client: HoldClient | None = None,
    hold_command_id: str | None = None,
) -> dict[str, Any]:
    """Runs `python_snippet` inside `pid` via gdb call-injection; peer of `pyspy_capture.capture`.

    When `hold_client`/`hold_command_id` are given, refuses to attach unless that
    hold is confirmed currently active (see `HoldClient.require_hold_active`).
    This does not eliminate the residual deadlock risk documented at module
    level -- it only converts it from unrecoverable to recoverable/retryable.
    """
    if hold_client is not None and hold_command_id is not None:
        hold_client.require_hold_active(hold_command_id)
    return schedule_pending_call_and_wait(
        pid,
        python_snippet,
        poll_timeout_s=poll_timeout_s,
        max_attempts=max_attempts,
        gdb_bin=gdb_bin,
    )
