"""Runs the `checkpoint` command: saves a Lightning checkpoint."""

from contextlib import ExitStack
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
    """Saves privately on every rank, then publishes the checkpoint on rank zero.

    Lightning's public save method includes a final barrier. Distributed saves use
    the Trainer's checkpoint connector to finish preparation on every rank before
    starting strategy I/O. Failures inside a strategy-owned collective are not
    recoverable here.
    Rank zero exits the staging context after save agreement; all ranks agree
    on the outcome of publication and cleanup before returning.
    """
    del pl_module, safe_point
    label = command.args["label"]
    paths: tuple[Path, Path] | None = None
    with (
        RankAgreement(runtime.rank_sync, "checkpoint publication"),
        ExitStack() as stack,
    ):
        with RankAgreement(runtime.rank_sync, "checkpoint reservation"):
            if runtime.on_rank_zero:
                paths = stack.enter_context(
                    runtime.artifact_store.stage_checkpoint(label)
                )
        paths = runtime.rank_sync.broadcast(paths)
        if paths is None:
            raise RuntimeError("rank 0 could not reserve a checkpoint path")
        staged, path = paths
        _save_checkpoint(runtime, trainer, staged)
    return {"checkpoint_path": str(path)}


def _save_checkpoint(runtime: CommandRuntime, trainer: "Trainer", staged: Path) -> None:
    """Waits for every rank's write, including Lightning's background writer.

    AsyncCheckpointIO teardown joins pending writes and raises their errors.
    Lightning lazily recreates the executor on the next save.
    """
    checkpoint = None
    if runtime.rank_sync.world_size > 1:
        with RankAgreement(runtime.rank_sync, "checkpoint preparation"):
            checkpoint = trainer._checkpoint_connector.dump_checkpoint(None)
    with RankAgreement(runtime.rank_sync, "checkpoint write"):
        try:
            if checkpoint is None:
                trainer.save_checkpoint(str(staged))
            else:
                trainer.strategy.save_checkpoint(checkpoint, str(staged))
        finally:
            checkpoint_io = trainer.strategy.checkpoint_io
            if isinstance(checkpoint_io, runtime.backend.PL.plugins.AsyncCheckpointIO):
                checkpoint_io.teardown()
