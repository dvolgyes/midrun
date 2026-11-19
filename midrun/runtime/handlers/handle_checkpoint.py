"""Runs the `checkpoint` command: saves a Lightning checkpoint."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.rank_agreement import RankAgreement

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


def handle_checkpoint(
    runtime: CommandRuntime,
    pl_module: "LightningModule",
    trainer: "Trainer",
    command: Command,
    safe_point: SafePoint | None,
) -> dict[str, Any]:
    """Saves a checkpoint on this rank, at the path that rank 0 selects for the label.

    Lightning's public save method includes a final barrier. Distributed saves use
    the Trainer's checkpoint connector to finish preparation on every rank before
    starting strategy I/O. Failures inside a strategy-owned collective are not
    recoverable here.
    """
    del pl_module, safe_point
    label = command.args["label"]
    # Every rank saves, as Lightning requires, so every rank must use rank 0's path.
    # Only rank 0 reserves it, so the other ranks leave no empty files.
    reserved: Path | None = None
    with RankAgreement(runtime.rank_sync, "checkpoint reservation"):
        if runtime.on_rank_zero:
            reserved = runtime.artifact_store.reserve_checkpoint_path(label)
    path = runtime.rank_sync.broadcast(reserved)
    if path is None:
        raise RuntimeError("rank 0 could not reserve a checkpoint path")
    try:
        if runtime.rank_sync.world_size == 1:
            trainer.save_checkpoint(str(path))
        else:
            with RankAgreement(runtime.rank_sync, "checkpoint preparation"):
                checkpoint = trainer._checkpoint_connector.dump_checkpoint(None)
            with RankAgreement(runtime.rank_sync, "checkpoint write"):
                trainer.strategy.save_checkpoint(checkpoint, str(path))
    except BaseException:
        if reserved is not None:
            reserved.unlink(missing_ok=True)
        raise
    return {"checkpoint_path": str(path)}
