"""Runs the `save_optimizer_summary` command."""

import json
from typing import TYPE_CHECKING, Any

from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint
from trainctl.runtime.handlers.optimizer_summary_payload import (
    optimizer_summary_payload,
)

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_save_optimizer_summary(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes the optimizer summary as a debug capture."""
    del pl_module, command, safe_point
    payload = optimizer_summary_payload(trainer)
    return runtime.artifact_store.write_debug_capture(
        "optimizer-summary", {"optimizer.json": json.dumps(payload, indent=2)}
    )
