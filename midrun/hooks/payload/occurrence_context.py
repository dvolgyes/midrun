"""The identity of one Lightning callback firing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OccurrenceContext:
    """Identity shared by every script invoked for one Lightning callback firing.

    Attributes:
        session_id: This run's hooks session id (see `bootstrap_hooks_tree`).
        hook: The exact `Callback` method name, e.g. `on_train_batch_start`.
        stage: `trainer.state.stage` as a string, or `None` if unavailable.
        epoch: `trainer.current_epoch`, or `None` if unavailable.
        global_step: `trainer.global_step`, or `None` if unavailable.
        batch_idx: The batch index Lightning passed, or `None` for non-batch hooks.
        dataloader_idx: The dataloader index Lightning passed, or `None` otherwise.
        rank: `trainer.global_rank`.
        world_size: `trainer.world_size`.
    """

    session_id: str
    hook: str
    stage: str | None
    epoch: int | None
    global_step: int | None
    batch_idx: int | None
    dataloader_idx: int | None
    rank: int
    world_size: int
