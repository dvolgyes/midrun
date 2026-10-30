"""Runs the `stop` command: stops the run."""

from typing import Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint


def handle_stop(
    runtime: CommandRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Asks the trainer to stop, refuses holds from now on and releases the active hold."""
    del pl_module, command, safe_point
    trainer.should_stop = True
    # Cancel queued holds before the release, so no later safe point holds the run again.
    runtime.command_queue.refusals.refuse("hold", "the run is stopping")
    cancelled = runtime.command_queue.end_queued("cancelled", kind="hold")
    runtime.gate.release()
    return {"stop": "requested", "cancelled_holds": cancelled}
