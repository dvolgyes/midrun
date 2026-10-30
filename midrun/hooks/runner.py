"""Synchronous, supervised execution of one hook script.

Training waits for the script; concurrent stdout/stderr draining here is an
implementation detail to avoid pipe deadlock, not asynchronous dispatch. Exit status
is data (`check=False`): only an actual launch I/O failure is caught. `KeyboardInterrupt`
contains the child's process group before re-raising, never swallowing it.
"""

import contextlib
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from midrun.hooks.bounded_drain import BoundedDrain

_GRACE_PERIOD_S = 5.0
_DRAIN_DEADLINE_S = 1.0


@dataclass(frozen=True)
class ExecutionResult:
    """The outcome of one supervised hook script invocation.

    Attributes:
        phase: `"completed"`, `"timeout"`, or `"launch_error"`.
        exit_code: The child's exit code, or `None` if it was killed or never launched.
        signal_number: The signal that killed the child, or `None`.
        duration_s: Wall-clock seconds from launch attempt to reap.
        stdout_tail: Up to the configured byte budget of stdout, decoded with `replace`.
        stderr_tail: Same, for stderr.
        stdout_truncated: Whether stdout exceeded the retained-tail budget.
        stderr_truncated: Whether stderr exceeded the retained-tail budget.
        error: A human-readable launch failure description, else `None`.
    """

    phase: str
    exit_code: int | None
    signal_number: int | None
    duration_s: float
    stdout_tail: str
    stderr_tail: str
    stdout_truncated: bool
    stderr_truncated: bool
    error: str | None


def run_script(
    script: Path,
    argv_extra: list[str],
    *,
    cwd: Path,
    shell: Path,
    timeout_s: float | None,
    max_output_bytes: int,
    log: Any,
) -> ExecutionResult:
    """Runs `shell script *argv_extra` in its own process group and waits for it.

    Args:
        script: Absolute path to the hook script (already validated as enabled).
        argv_extra: Extra positional arguments after `script` (the params/manifest path).
        cwd: Working directory for the child -- the live `hooks/` root.
        shell: The validated Bash executable.
        timeout_s: Seconds to wait before terminating the child; `None` waits forever.
        max_output_bytes: Retained-tail budget per stream.
        log: A Loguru-compatible logger for streamed output and termination warnings.

    Returns:
        The completed, killed, or launch-failed result. Never raises for the child's
        own exit status or signal death; propagates only `KeyboardInterrupt`, after
        containing the child's process group first.
    """
    start = time.monotonic()
    try:
        # Running an operator-enabled hook script is this module's purpose; the
        # execute bit is the sole privilege boundary.
        process = subprocess.Popen(  # noqa: S603  # pylint: disable=consider-using-with
            [str(shell), str(script), *argv_extra],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        return ExecutionResult(
            phase="launch_error",
            exit_code=None,
            signal_number=None,
            duration_s=time.monotonic() - start,
            stdout_tail="",
            stderr_tail="",
            stdout_truncated=False,
            stderr_truncated=False,
            error=str(exc),
        )

    # No `with process:` -- its exit closes the pipes, which blocks while a drain
    # thread is still reading one; each drain closes its own pipe at EOF instead.
    stdout_drain = BoundedDrain(
        process.stdout,
        tag=f"{script.name} stdout",
        max_bytes=max_output_bytes,
        log=log,
    )
    stderr_drain = BoundedDrain(
        process.stderr,
        tag=f"{script.name} stderr",
        max_bytes=max_output_bytes,
        log=log,
    )
    stdout_drain.start()
    stderr_drain.start()

    timed_out = False
    try:
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_group(process, log)
        except KeyboardInterrupt:
            _terminate_group(process, log)
            raise
    finally:
        if not _contain_after_exit(process, (stdout_drain, stderr_drain)):
            log.warning(
                "hook process group {}: an output pipe is still open {}s after the "
                "script exited (held by a process outside the group); no longer "
                "waiting for it",
                process.pid,
                _DRAIN_DEADLINE_S,
            )

    duration = time.monotonic() - start
    exit_code = process.returncode if process.returncode >= 0 else None
    signal_number = -process.returncode if process.returncode < 0 else None
    return ExecutionResult(
        phase="timeout" if timed_out else "completed",
        exit_code=exit_code,
        signal_number=signal_number,
        duration_s=duration,
        stdout_tail=stdout_drain.tail(),
        stderr_tail=stderr_drain.tail(),
        stdout_truncated=stdout_drain.truncated,
        stderr_truncated=stderr_drain.truncated,
        error=None,
    )


def _contain_after_exit(
    process: subprocess.Popen[bytes], drains: tuple[BoundedDrain, ...]
) -> bool:
    """Kills what is left of the reaped script's process group, then joins the drains.

    A member of the group that survived the script (a background job, or a child
    that ignored SIGTERM) gets SIGKILL. A process that left the group (e.g. via
    `setsid`) can still hold a pipe open; the drains are joined only until a
    deadline, so it cannot block training.

    Returns:
        Whether every drain reached EOF before the deadline.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    deadline = time.monotonic() + _DRAIN_DEADLINE_S
    finished = [
        drain.join(timeout=max(deadline - time.monotonic(), 0.0)) for drain in drains
    ]
    return all(finished)


def _terminate_group(process: subprocess.Popen[bytes], log: Any) -> None:
    """Sends SIGTERM to the child's process group, escalating to SIGKILL after a grace period."""
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_GRACE_PERIOD_S)
        return
    except subprocess.TimeoutExpired:
        pass
    log.warning(
        "hook process group {} did not exit within {}s of SIGTERM; sending SIGKILL",
        pgid,
        _GRACE_PERIOD_S,
    )
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()
