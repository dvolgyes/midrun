"""Decides whether a hook script run failed."""

from trainctl.hooks.runner import ExecutionResult

_FAILED_PHASES = ("timeout", "launch_error", "capture_failed")


def is_hook_failure(result: ExecutionResult) -> bool:
    """Returns whether the run timed out, did not launch, failed its capture, or exited non-zero."""
    return result.phase in _FAILED_PHASES or (
        result.phase == "completed" and result.exit_code != 0
    )
