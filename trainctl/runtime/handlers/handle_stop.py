"""Runs the `stop` command: stops the run."""

from typing import TYPE_CHECKING, Any

from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_stop(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    """Asks the trainer to stop, cancels the queued holds and releases the active hold."""
    del pl_module, command, safe_point
    trainer.should_stop = True
    # Cancel queued holds before the release, so no later safe point holds the run again.
    queue = runtime.command_queue
    cancelled = [
        held.id
        for held in queue.list_all()
        if held.kind == "hold" and queue.cancel(held.id)
    ]
    runtime.gate.release()
    return {"stop": "requested", "cancelled_holds": cancelled}
