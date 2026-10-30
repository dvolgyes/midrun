"""Builds the result of a hook whose payload could not be captured."""

from midrun.hooks.runner import ExecutionResult


def capture_failed_result(reason: str) -> ExecutionResult:
    """Returns a `capture_failed` result: the script was never launched."""
    return ExecutionResult(
        phase="capture_failed",
        exit_code=None,
        signal_number=None,
        duration_s=0.0,
        stdout_tail="",
        stderr_tail="",
        stdout_truncated=False,
        stderr_truncated=False,
        error=reason,
    )
