"""Logs one failed hook script run with its full context."""

from pathlib import Path
from typing import Any

from trainctl.hooks.payload.occurrence_context import OccurrenceContext
from trainctl.hooks.runner import ExecutionResult


def log_hook_failure(
    script: Path,
    modality: str,
    context: OccurrenceContext,
    occurrence_id: str,
    invocation_id: str,
    result: ExecutionResult,
    log: Any,
) -> None:
    """Logs `result` as an error, with the script, occurrence, rank, and output tails."""
    log.error(
        "hook failed: hook={} modality={} script={} session={} occurrence={} "
        "invocation={} rank={} stage={} epoch={} global_step={} phase={} exit_code={} "
        "signal={} duration_s={:.3f} error={} stdout_truncated={} stderr_truncated={} "
        "stdout_tail={!r} stderr_tail={!r}",
        context.hook,
        modality,
        script,
        context.session_id,
        occurrence_id,
        invocation_id,
        context.rank,
        context.stage,
        context.epoch,
        context.global_step,
        result.phase,
        result.exit_code,
        result.signal_number,
        result.duration_s,
        result.error,
        result.stdout_truncated,
        result.stderr_truncated,
        result.stdout_tail,
        result.stderr_tail,
    )
