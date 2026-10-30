"""A scheduled control-plane command and its outcome."""

from dataclasses import dataclass
from typing import Any

from midrun.runtime.commands.command_when import CommandWhen


@dataclass
class Command:
    """A scheduled control-plane command and its execution outcome.

    Attributes:
        id: Unique command id.
        kind: One of `hold`, `stop`, `checkpoint`, `snapshot`, `set_learning_rate`,
            `reset_momentum`, `set_dataloader_finite_check`, `save_current_batch`,
            `save_current_loss`, `save_partial_gradients`, `save_optimizer_summary`,
            `release_exception`, `torchinfo_capture`, `profile`, `torchview_capture`,
            `torchviz_capture`, `torchlens_capture`. CUDA memory snapshots and
            allocator-history toggling are not commands -- see `GET
            /debug/cuda-memory/snapshot` and `GET /debug/cuda-memory/history`, which
            don't touch the Trainer/model and so don't need training-thread mediation.
        args: Handler-specific arguments.
        when: The scheduling condition that makes this command due.
        created_at: Unix timestamp of submission.
        status: `queued`, `executing`, `succeeded`, `failed`, `cancelled`, or
            `expired` (still queued when the stage stopped).
        result: Handler result payload, once succeeded.
        error: Failure message, once failed or expired.
    """

    id: str
    kind: str
    args: dict[str, Any]
    when: CommandWhen
    created_at: float
    status: str = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None
