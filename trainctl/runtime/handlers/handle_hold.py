"""Runs the `hold` command: pauses the run at the current safe point."""

from typing import TYPE_CHECKING, Any

from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_hold(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    """Marks the run gate held for `command` at `safe_point`."""
    del pl_module
    runtime.gate.enter(
        command.id, str(safe_point), trainer.current_epoch, trainer.global_step
    )
    return {"hold": "entered", "reason": command.args["reason"]}
