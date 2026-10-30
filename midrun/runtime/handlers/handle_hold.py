"""Runs the `hold` command: pauses the run at the current safe point."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def handle_hold(
    runtime: "MidrunRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Marks the run gate held for `command` at `safe_point`."""
    del pl_module
    runtime.gate.enter(
        command.id, str(safe_point), trainer.current_epoch, trainer.global_step
    )
    return {"hold": "entered", "reason": command.args["reason"]}
