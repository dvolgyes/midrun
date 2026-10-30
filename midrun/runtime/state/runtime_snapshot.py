"""One immutable snapshot of the runtime state."""

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor

from midrun.runtime.state.exception_breakpoint_info import ExceptionBreakpointInfo
from midrun.runtime.state.rank_info import RankInfo
from midrun.runtime.state.scalar_metric import ScalarMetric


@dataclass(frozen=True)
class RuntimeSnapshot:
    """A fully immutable point-in-time view of a training run, safe for cross-thread reads.

    Attributes:
        generation: Monotonically increasing publish counter.
        updated_at: Unix timestamp of this snapshot's publication.
        run_id: Unique identifier for this training run.
        status: One of `initializing`, `running`, `held`, `exception_held`, `stopping`,
            `finished`, `failed`.
        stage: Lightning trainer stage (`fit`, `validate`, `test`, `predict`), when known.
        phase: Current training phase (`train`, `validate`, `test`, `predict`, `idle`).
        epoch: Current epoch.
        global_step: Current global step.
        batch_idx: Current batch index within the epoch, when known.
        world_size: Number of distributed processes.
        ranks: Static metadata for every known rank.
        metrics: Latest scalar metrics keyed by Lightning log key.
        learning_rates: Current learning rate per optimizer param group; a
            tensor learning rate stays a tensor (see `host_snapshot`).
        pending_commands: Number of commands still queued.
        hold_state: `none` or `held`.
        rest_host: REST bind host, once started.
        rest_port: REST bind port, once started.
        torch_debug_url: PyTorch distributed debug frontend URL, once started.
        artifacts_path: This run's checkpoint/snapshot/debug-capture directory, once
            resolved -- distinct per run (colocated under Lightning's own
            `trainer.log_dir` when available), so distinguishes concurrent runs that
            share a host and launch directory.
        lightning_backend: Selected Lightning backend family name.
        lightning_version: Selected Lightning backend version string.
        exception: The active backward-exception breakpoint, when `status ==
            "exception_held"`; `None` otherwise.
        backward_interception_supported: Whether the active strategy is known to route
            backward through `LightningModule.backward()`, making
            `break_on_backward_exception` effective.
        backward_interception_reason: Why interception is unsupported, when
            `backward_interception_supported` is False.
    """

    generation: int
    updated_at: float

    run_id: str
    status: str
    stage: str | None
    phase: str

    epoch: int
    global_step: int
    batch_idx: int | None

    world_size: int
    ranks: tuple[RankInfo, ...]

    metrics: Mapping[str, ScalarMetric]
    learning_rates: tuple[float | Tensor, ...]

    pending_commands: int
    hold_state: str

    rest_host: str | None
    rest_port: int | None

    torch_debug_url: str | None
    artifacts_path: str | None

    lightning_backend: str
    lightning_version: str

    exception: ExceptionBreakpointInfo | None
    backward_interception_supported: bool
    backward_interception_reason: str | None
