"""Runs the `checkpoint` command: saves a Lightning checkpoint."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from midrun.runtime.command_specs.command_runtime import CommandRuntime
from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


@contextmanager
def _agree_on_phase(runtime: CommandRuntime, phase: str) -> Iterator[None]:
    """Reports a local exception to every rank before any rank advances."""
    error = None
    try:
        yield
    # pylint: disable-next=broad-exception-caught
    except Exception as exc:  # noqa: BLE001 -- exchange ordinary checkpoint failures
        error = f"{type(exc).__name__}: {exc}"
    for rank, message in enumerate(runtime.rank_sync.all_gather(error)):
        if message is not None:
            raise RuntimeError(f"checkpoint {phase} failed on rank {rank}: {message}")


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
        if runtime.rank_sync.world_size == 1:
            trainer.save_checkpoint(str(path))
        else:
            with _agree_on_phase(runtime, "preparation"):
                checkpoint = trainer._checkpoint_connector.dump_checkpoint(None)
            with _agree_on_phase(runtime, "write"):
                trainer.strategy.save_checkpoint(checkpoint, str(path))
    except BaseException:
        if reserved is not None:
            reserved.unlink(missing_ok=True)
        raise
    return {"checkpoint_path": str(path)}
