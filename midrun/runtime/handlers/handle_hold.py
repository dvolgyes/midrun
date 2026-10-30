"""Runs the `hold` command: pauses the run at the current safe point."""

from typing import Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint


def handle_hold(
    runtime: CommandRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Marks the run gate held for `command` at `safe_point`, or joins the active hold."""
    del pl_module
    hold = runtime.gate.enter(
        command.id, str(safe_point), trainer.current_epoch, trainer.global_step
    )
    if hold.command_id != command.id:
        return {
            "hold": "joined",
            "active_hold_id": hold.command_id,
            "reason": command.args["reason"],
        }
    return {"hold": "entered", "reason": command.args["reason"]}
