"""Runs the `checkpoint` command: saves a Lightning checkpoint."""

from pathlib import Path
from typing import Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint


def handle_checkpoint(
    runtime: CommandRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Saves a checkpoint on this rank, at the path that rank 0 selects for the label.

    A failure on one rank does not leave the other ranks in a collective: every rank
    raises when rank 0 cannot reserve the path, and a rank whose save raises still
    enters the barrier that ends Lightning's save.
    """
    del pl_module, safe_point
    label = command.args["label"]
    # Every rank saves, as Lightning requires, so every rank must use rank 0's path.
    # Only rank 0 reserves it, so the other ranks leave no empty files.
    reserved: Path | None = None
    reserve_error: Exception | None = None
    if runtime.on_rank_zero:
        try:
            reserved = runtime.artifact_store.reserve_checkpoint_path(label)
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:  # noqa: BLE001 -- re-raised after the broadcast
            reserve_error = exc
    path = runtime.rank_sync.broadcast(reserved)
    if reserve_error is not None:
        raise reserve_error
    if path is None:
        raise RuntimeError("rank 0 could not reserve a checkpoint path")
    try:
        trainer.save_checkpoint(str(path))
    except BaseException as exc:
        # An empty or partial file must not look like a saved checkpoint.
        if reserved is not None:
            reserved.unlink(missing_ok=True)
        # Lightning's save ends with a barrier on every rank, which a raise skips.
        if isinstance(exc, Exception):
            trainer.strategy.barrier("midrun.handle_checkpoint")
        raise
    return {"checkpoint_path": str(path)}
