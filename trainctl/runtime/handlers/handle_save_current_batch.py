"""Runs the `save_current_batch` command."""

from typing import TYPE_CHECKING, Any

from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_save_current_batch(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    """Writes the in-flight batch with `torch.save` as a debug capture."""
    del pl_module, trainer, command, safe_point
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(runtime.current_step.batch, buffer)
    return runtime.artifacts.write_debug_capture(
        "current-batch", {"batch.pt": buffer.getvalue()}
    )
