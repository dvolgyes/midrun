"""Runs the `torchview_capture` command."""

from typing import TYPE_CHECKING, Any

from trainctl.runtime import forward_input
from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint
from trainctl.runtime.graph_inspectors.torchview_capture import torchview_capture

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_torchview_capture(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes a torchview graph of the model for the in-flight batch as a debug capture.

    Raises:
        ValueError: No batch is in flight.
    """
    del trainer, command, safe_point
    if runtime.current_step.batch is None:
        raise ValueError("torchview_capture requires an in-flight batch")
    target = forward_input.resolve(pl_module, runtime.current_step.batch)
    files = torchview_capture(pl_module, target)
    return runtime.artifact_store.write_debug_capture("torchview-graph", files)
