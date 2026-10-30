"""Schedules a Python snippet in a live process through gdb, with retries."""

import tempfile
import time
from pathlib import Path

from trainctl.debugger.gdb_capture.attempt_pending_call import attempt_pending_call
from trainctl.debugger.gdb_capture.check_gdb_available import check_gdb_available
from trainctl.debugger.gdb_capture.evaluate_mi_expression import (
    GdbAbandonedEvaluationError,
)
from trainctl.debugger.gdb_capture.gdb_attach_error import GdbAttachError


def schedule_pending_call(
    pid: int,
    python_snippet: str,
    *,
    max_attempts: int = 20,
    gdb_bin: str = "gdb",
    directory: Path | None = None,
) -> Path:
    """Attaches to `pid`, enqueues `python_snippet` via `Py_AddPendingCall`, detaches immediately -- the snippet itself runs later, on the target's own schedule.

    Retries on `GdbAbandonedEvaluationError` (see that class's docstring); each
    retry re-attaches fresh rather than reusing the aborted session.

    Args:
        pid: Target process id. Must already be the real interpreter process, not
            a launcher wrapper (e.g. `uv run python ...` does not execve -- resolve
            the target's own `os.getpid()`-reported value instead, such as
            trainctl's `RankInfo.pid`).
        python_snippet: Python source executed in a fresh, isolated globals dict
            (`exec(source, {})`) inside the target's own interpreter -- never in
            `__main__`'s real globals, which can silently clobber the target's own
            top-level variables of the same name.
        max_attempts: Retry budget for abandoned evaluations.
        gdb_bin: `gdb` executable to invoke.
        directory: Where the snippet file is created; the system temp directory
            when None.

    Returns:
        The snippet file, which the caller deletes once the call is serviced. It
        is deleted here when scheduling fails.

    Raises:
        GdbUnavailableError: `gdb_bin` is missing or unusable.
        GdbAttachError: every attempt failed for a non-retryable reason, or the
            retry budget was exhausted.
    """
    check_gdb_available(gdb_bin)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=directory, delete=False
    ) as f:
        # An exception escaping a pending call raises SystemError in whatever
        # code the target runs next, so the snippet's own errors stop here.
        f.write(
            f"try:\n    exec({python_snippet!r}, {{}})\n"
            "except Exception:\n    __import__('traceback').print_exc()\n"
        )
        snippet_path = Path(f.name)

    last_error: Exception | None = None
    scheduled = False
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                attempt_pending_call(pid, snippet_path, gdb_bin)
                scheduled = True
                return snippet_path
            except GdbAbandonedEvaluationError as exc:
                last_error = exc
                time.sleep(0.1 * attempt)
        raise GdbAttachError(f"gave up after {max_attempts} attempts: {last_error}")
    finally:
        if not scheduled:
            snippet_path.unlink(missing_ok=True)
