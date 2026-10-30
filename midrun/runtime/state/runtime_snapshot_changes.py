"""The field overrides that `StateStore.update` accepts."""

from collections.abc import Mapping
from typing import TypedDict

from midrun.runtime.state.exception_breakpoint_info import ExceptionBreakpointInfo
from midrun.runtime.state.rank_info import RankInfo
from midrun.runtime.state.scalar_metric import ScalarMetric


class RuntimeSnapshotChanges(TypedDict, total=False):
    """The `RuntimeSnapshot` fields that a caller can override, with their types.

    `StateStore.update` sets `generation` and `updated_at` itself, so they are not
    here. The other keys and types match `RuntimeSnapshot`; mypy rejects a
    mismatch at the `dataclasses.replace` call.

    Attributes:
        run_id: See `RuntimeSnapshot`.
        status: See `RuntimeSnapshot`.
        stage: See `RuntimeSnapshot`.
        phase: See `RuntimeSnapshot`.
        epoch: See `RuntimeSnapshot`.
        global_step: See `RuntimeSnapshot`.
        batch_idx: See `RuntimeSnapshot`.
        world_size: See `RuntimeSnapshot`.
        ranks: See `RuntimeSnapshot`.
        metrics: See `RuntimeSnapshot`.
        learning_rates: See `RuntimeSnapshot`.
        pending_commands: See `RuntimeSnapshot`.
        hold_state: See `RuntimeSnapshot`.
        rest_host: See `RuntimeSnapshot`.
        rest_port: See `RuntimeSnapshot`.
        torch_debug_url: See `RuntimeSnapshot`.
        artifacts_path: See `RuntimeSnapshot`.
        lightning_backend: See `RuntimeSnapshot`.
        lightning_version: See `RuntimeSnapshot`.
        exception: See `RuntimeSnapshot`.
        backward_interception_supported: See `RuntimeSnapshot`.
        backward_interception_reason: See `RuntimeSnapshot`.
    """

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
    learning_rates: tuple[float, ...]
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
