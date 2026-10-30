"""The part of the runtime that command handlers use."""

from typing import Any, Protocol

from midrun.config import MidrunConfig
from midrun.lightning_backend import LightningBackend
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.commands.command_queue import CommandQueue
from midrun.runtime.current_step import CurrentStep
from midrun.runtime.finite_guard import DataloaderFiniteGuard
from midrun.runtime.gate import RunGate
from midrun.runtime.hparam_registry.tunable_hparam_spec import TunableHparamSpec
from midrun.runtime.profiler import ProfilerRun
from midrun.runtime.rank_sync import RankSync
from midrun.runtime.state.exception_breakpoint_info import ExceptionBreakpointInfo
from midrun.runtime.state.state_store import StateStore


class CommandRuntime(Protocol):
    """The runtime state that a command handler reads and changes.

    `MidrunRuntime` implements it. The handlers depend on this interface, not on the
    runtime module, which imports them.

    Attributes:
        config: Effective `MidrunConfig` for this run.
        backend: The Lightning backend the composed model uses.
        state: Published `RuntimeSnapshot` store.
        command_queue: Submitted control-plane commands.
        gate: Cooperative hold gate.
        current_step: Live batch/loss/exception for the in-flight training step.
        finite_guard: Opt-in dataloader-batch finite-value checker.
        rank_sync: Control-plane collectives with the other ranks.
        profiler_run: The armed/active bounded `torch.profiler` run, if any.
        lightning_profiler: The Lightning profiler that `set_lightning_profiler`
            installed, if any.
        lightning_profiler_level: The level `lightning_profiler` was last set to.
        exception_hold: The active exception hold on rank 0, or None.
        tunable_hparam_specs: The declared tunable hparams of the model class.
    """

    config: MidrunConfig
    backend: LightningBackend
    state: StateStore
    command_queue: CommandQueue
    gate: RunGate
    current_step: CurrentStep
    finite_guard: DataloaderFiniteGuard
    rank_sync: RankSync
    profiler_run: ProfilerRun | None
    lightning_profiler: Any | None
    lightning_profiler_level: str
    exception_hold: ExceptionBreakpointInfo | None
    tunable_hparam_specs: dict[str, TunableHparamSpec]

    @property
    def artifact_store(self) -> ArtifactStore:
        """The artifact store of the run."""
        ...

    @property
    def on_rank_zero(self) -> bool:
        """Whether this process is global rank 0."""
        ...
