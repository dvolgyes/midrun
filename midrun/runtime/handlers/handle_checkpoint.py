"""Runs the `checkpoint` command: saves a Lightning checkpoint."""

from typing import TYPE_CHECKING, Any

from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def handle_checkpoint(
    runtime: "MidrunRuntime",
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Saves a checkpoint on this rank, at the path that rank 0 selects for the label."""
    del pl_module, safe_point
    label = command.args["label"]
    # Every rank saves, as Lightning requires, so every rank must use rank 0's path.
    # Only rank 0 reserves it, so the other ranks leave no empty files.
    reserved = (
        runtime.artifact_store.reserve_checkpoint_path(label)
        if runtime.on_rank_zero
        else None
    )
    path = runtime.rank_sync.broadcast(reserved)
    try:
        trainer.save_checkpoint(str(path))
    except BaseException:
        # An empty or partial file must not look like a saved checkpoint.
        if reserved is not None:
            reserved.unlink(missing_ok=True)
        raise
    return {"checkpoint_path": str(path)}
