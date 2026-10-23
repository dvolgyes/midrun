"""Runs the `checkpoint` command: saves a Lightning checkpoint."""

from typing import TYPE_CHECKING, Any

from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def handle_checkpoint(
    runtime: "TrainctlRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Saves a checkpoint on this rank, at the path that rank 0 selects for the label."""
    del pl_module, safe_point
    label = command.args["label"]
    # Every rank saves, as Lightning requires, so every rank must use rank 0's path.
    path = runtime.rank_sync.broadcast(runtime.artifact_store.checkpoint_path(label))
    trainer.save_checkpoint(str(path))
    return {"checkpoint_path": str(path)}
