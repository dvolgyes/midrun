"""Runs the `torchview_capture` command."""

from typing import Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.graph_inspectors.torchview_capture import torchview_capture
from midrun.runtime.run_on_in_flight_batch import run_on_in_flight_batch


def handle_torchview_capture(
    runtime: CommandRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes a torchview graph of the model for the in-flight batch as a debug capture.

    Raises:
        ValueError: No batch is in flight, or the parameters are sharded.
    """
    del command, safe_point
    files = run_on_in_flight_batch(
        lambda target: torchview_capture(pl_module, target),
        runtime,
        trainer,
        pl_module,
        "torchview_capture",
    )
    return runtime.artifact_store.write_debug_capture("torchview-graph", files)
