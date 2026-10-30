"""Runs the `torchinfo_capture` command."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.run_on_in_flight_batch import run_on_in_flight_batch
from midrun.runtime.torchinfo_inspector.torchinfo_rich_summary import (
    torchinfo_rich_summary,
)

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def handle_torchinfo_capture(
    runtime: "MidrunRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes a torchinfo summary of the model for the in-flight batch as a debug capture.

    Raises:
        ValueError: No batch is in flight, or the parameters are sharded.
    """
    del command, safe_point
    files = run_on_in_flight_batch(
        lambda target: torchinfo_rich_summary(pl_module, target),
        runtime,
        trainer,
        pl_module,
        "torchinfo_capture",
    )
    return runtime.artifact_store.write_debug_capture("torchinfo-summary", files)
