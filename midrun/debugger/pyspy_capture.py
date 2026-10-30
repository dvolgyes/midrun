"""Safe, read-only capture mode using py-spy.

py-spy's dump mode pauses the target just long enough to read its memory and
reconstruct Python frames -- it never makes the target execute injected code, so
it is structurally immune to the CUDA/autograd deadlock class documented in
`gdb_capture`'s module docstring. Peer of `gdb_capture.gdb_capture`: equally
supported, not a "safe default" for a "risky opt-in" -- reach for this whenever a
read-only stack/locals dump is enough, and for `gdb_capture` when you need to
mutate live state or call an arbitrary live function instead.
"""

import json
import subprocess
from typing import Any

from midrun.debugger.descendant_pids import descendant_pids
from midrun.debugger.hold_client import HoldClient


class PySpyError(RuntimeError):
    """py-spy failed to produce a stack dump."""


def capture(
    pid: int,
    *,
    locals_verbosity: int = 0,
    native: bool = False,
    nonblocking: bool = False,
    subprocesses: bool = False,
    py_spy_bin: str = "py-spy",
    hold_client: HoldClient | None = None,
    hold_command_id: str | None = None,
) -> dict[str, Any]:
    """Captures a stack dump of `pid` via `py-spy dump --json`.

    Args:
        pid: Target process id.
        locals_verbosity: 0 for no locals, 1 for `-l`, 2+ for `-ll` (more detail).
        native: Include native (C/C++/Cython) frames.
        nonblocking: Read memory without pausing the target -- less consistent, but
            removes even the brief pause py-spy's default mode uses.
        subprocesses: Also dump every live descendant process of `pid`, one py-spy
            run each (`py-spy dump --json --subprocesses` reports only `pid`). The
            processes are paused one after another, not at the same time.
        py_spy_bin: `py-spy` executable to invoke, resolved via `PATH`.
        hold_client: When given with `hold_command_id`, refuses to run unless that
            hold is confirmed currently active (see `HoldClient.require_hold_active`).
            Optional here (unlike `gdb_capture`) since a read-only dump carries no
            deadlock risk of its own; still recommended for the coherent-collective
            reasons in the Chapter 2 design doc's "cooperative attach" workflow.
        hold_command_id: The hold command id to check when `hold_client` is given.

    Returns:
        `{"pid": pid, "threads": [...]}` -- `threads` is py-spy's own per-thread dump
        list (each entry has `pid`, `thread_id`, `active`, `owns_gil`, `frames`, ...).
        With `subprocesses`, `threads` also holds the threads of the descendants, and
        `failed_subprocesses` lists `{"pid", "error"}` for each descendant that py-spy
        could not dump (it exited, is not Python, or refused the attach).

    Raises:
        PySpyError: py-spy exited non-zero, timed out, or produced unparsable output
            for `pid` itself.
        HoldNotActiveError: a hold check was requested but the hold isn't active.
    """
    if hold_client is not None and hold_command_id is not None:
        hold_client.require_hold_active(hold_command_id)
    options = ["--json", *["-l"] * max(locals_verbosity, 0)]
    if native:
        options.append("--native")
    if nonblocking:
        options.append("--nonblocking")
    result: dict[str, Any] = {"pid": pid, "threads": _dump(py_spy_bin, pid, options)}
    if subprocesses:
        failed: list[dict[str, Any]] = []
        for child in descendant_pids(pid):
            try:
                result["threads"].extend(_dump(py_spy_bin, child, options))
            except PySpyError as exc:
                failed.append({"pid": child, "error": str(exc)})
        result["failed_subprocesses"] = failed
    return result


def _dump(py_spy_bin: str, pid: int, options: list[str]) -> list[dict[str, Any]]:
    """Runs one `py-spy dump` of `pid` and returns its parsed per-thread list.

    Raises:
        PySpyError: py-spy exited non-zero, timed out, or produced unparsable output.
    """
    argv = [py_spy_bin, "dump", "--pid", str(pid), *options]
    try:
        result = subprocess.run(  # noqa: S603 -- operator-specified py-spy binary and pid
            argv, capture_output=True, text=True, timeout=30, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise PySpyError(f"py-spy dump timed out after {exc.timeout}s") from exc
    if result.returncode != 0:
        raise PySpyError(
            f"py-spy dump failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    try:
        threads: list[dict[str, Any]] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PySpyError(f"py-spy dump produced unparsable output: {exc}") from exc
    return threads
