"""Schedules a Python snippet through gdb and waits for its JSON report."""

import contextlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from midrun.debugger.gdb_capture.schedule_pending_call import (
    schedule_pending_call,
)


def schedule_pending_call_and_wait(
    pid: int,
    python_snippet: str,
    *,
    poll_timeout_s: float = 10.0,
    max_attempts: int = 20,
    gdb_bin: str = "gdb",
) -> dict[str, Any]:
    """`schedule_pending_call()` plus polling for the snippet's own JSON report to appear.

    `python_snippet` must write a JSON report to the path given by the literal
    substring `{out_path}` (used as a Python expression, e.g. `open({out_path},
    "w")`), which this function replaces with a quoted string literal before
    scheduling. A plain substring replacement is used (not `str.format`) so the
    snippet is free to contain its own literal `{`/`}` (dict/set literals,
    f-strings, ...) without colliding with the placeholder.

    Raises:
        TimeoutError: the target's eval loop never serviced the pending call
            within `poll_timeout_s` -- e.g. it is genuinely stuck in a call that
            never returns control to the interpreter -- or the report did not
            become complete JSON in that time.
    """
    # A private (0700) directory: no other user can plant or swap the report or
    # the snippet. A late call can recreate files during cleanup; those stay.
    with tempfile.TemporaryDirectory(
        prefix="midrun-gdb-", ignore_cleanup_errors=True
    ) as private_dir:
        out_path = Path(private_dir) / "report.json"
        _ = schedule_pending_call(
            pid,
            python_snippet.replace("{out_path}", repr(str(out_path))),
            max_attempts=max_attempts,
            gdb_bin=gdb_bin,
            directory=Path(private_dir),
        )
        deadline = time.monotonic() + poll_timeout_s
        while time.monotonic() < deadline:
            if out_path.exists():
                # The snippet creates the report before it finishes writing it.
                with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError):
                    return dict(json.loads(out_path.read_text(encoding="utf-8")))
            time.sleep(0.05)
        raise TimeoutError(
            f"pid {pid} never serviced the pending call within {poll_timeout_s}s "
            "-- it may be stuck in a call that never returns to the interpreter"
        )
