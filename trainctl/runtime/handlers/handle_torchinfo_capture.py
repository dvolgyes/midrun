"""Runs the `torchinfo_capture` command."""

from typing import TYPE_CHECKING, Any

from trainctl.runtime import forward_input
from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint
from trainctl.runtime.torchinfo_inspector.torchinfo_rich_summary import (
    torchinfo_rich_summary,
)

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_torchinfo_capture(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    """Writes a torchinfo summary of the model for the in-flight batch as a debug capture.

    Raises:
        ValueError: No batch is in flight.
    """
    del trainer, command, safe_point
    if runtime.current_step.batch is None:
        raise ValueError("torchinfo_capture requires an in-flight batch")
    target = forward_input.resolve(pl_module, runtime.current_step.batch)
    files = torchinfo_rich_summary(pl_module, target)
    return runtime.artifacts.write_debug_capture("torchinfo-summary", files)
