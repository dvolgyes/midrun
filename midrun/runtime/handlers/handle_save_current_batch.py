"""Runs the `save_current_batch` command."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def handle_save_current_batch(
    runtime: "MidrunRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Writes the in-flight batch with `torch.save` as a debug capture.

    Raises:
        ValueError: No batch is in flight.
    """
    del pl_module, trainer, command, safe_point
    if runtime.current_step.batch is None:
        raise ValueError("save_current_batch requires an in-flight batch")
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(runtime.current_step.batch, buffer)
    return runtime.artifact_store.write_debug_capture(
        "current-batch", {"batch.pt": buffer.getvalue()}
    )
