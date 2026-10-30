"""Runs the `save_optimizer_summary` command."""

from typing import TYPE_CHECKING, Any

from midrun.encode_json import encode_json
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.handlers.optimizer_summary_payload import (
    optimizer_summary_payload,
)

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def handle_save_optimizer_summary(
    runtime: "MidrunRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes the optimizer summary as a debug capture in standard JSON.

    NaN and infinite learning rates become `null` (see `encode_json`).
    """
    del pl_module, command, safe_point
    payload = optimizer_summary_payload(trainer)
    return runtime.artifact_store.write_debug_capture(
        "optimizer-summary", {"optimizer.json": encode_json(payload).decode()}
    )
