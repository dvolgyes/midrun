"""Owns Midrun's runtime services and dispatches Lightning-hook-driven state
publication and command execution.

The Lightning training thread is the only thread that mutates or deeply inspects
Trainer/model state. REST and FUSE only read published `RuntimeSnapshot`s or enqueue
`Command`s; this module is what actually executes them, from inside Lightning hooks.
"""

import os
import secrets
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, SupportsIndex

import torch
from loguru import logger

from midrun.config import MidrunConfig
from midrun.hooks.session import HandleResult, HookSession
from midrun.lightning_backend import LightningBackend
from midrun.logging.logging_session import LoggingSession
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.command_args.validate_command_args import validate_command_args
from midrun.runtime.commands.command import Command
from midrun.runtime.commands.command_queue import CommandQueue
from midrun.runtime.commands.next_id import next_id
from midrun.runtime.cuda_memory.cuda_memory_history_state import (
    CudaMemoryHistoryState,
)
from midrun.runtime.cuda_memory.set_cuda_memory_history import (
    set_cuda_memory_history,
)
from midrun.runtime.current_step import CurrentStep
from midrun.runtime.default_base_dir import default_base_dir
from midrun.runtime.distributed_initialized import distributed_initialized
from midrun.runtime.events import SafePoint
from midrun.runtime.finite_guard import DataloaderFiniteGuard, FiniteCheckReport
from midrun.runtime.gate import RunGate
from midrun.runtime.handlers.command_handlers import COMMAND_HANDLERS
from midrun.runtime.hparam_registry.collect_tunable_hparams import (
    collect_tunable_hparams,
)
from midrun.runtime.hparam_registry.describe_hparam import describe_hparam
from midrun.runtime.hparam_registry.get_field import get_field
from midrun.runtime.hparam_registry.hparam_value import HparamValue
from midrun.runtime.hparam_registry.resolve_hparam_path import resolve_hparam_path
from midrun.runtime.hparam_registry.tunable_hparam_spec import TunableHparamSpec
from midrun.runtime.is_rank_zero import is_rank_zero
from midrun.runtime.live_capture import LiveCaptureBroker
from midrun.runtime.phase_of_stage import phase_of_stage
from midrun.runtime.pipeline_pressure import PipelinePressureTracker
from midrun.runtime.profiler import ProfilerRun
from midrun.runtime.rank_sync import RankSync
from midrun.runtime.record_bad_batch import record_bad_batch
from midrun.runtime.runtime_services import RuntimeServices
from midrun.runtime.stage_sessions import StageSessions
from midrun.runtime.state.exception_breakpoint_info import ExceptionBreakpointInfo
from midrun.runtime.state.initial_snapshot import initial_snapshot
from midrun.runtime.state.rank_info import RankInfo
from midrun.runtime.state.scalar_metric import ScalarMetric
from midrun.runtime.state.state_store import StateStore
from midrun.runtime.torchinfo_inspector.basic_summary_json import basic_summary_json
from midrun.runtime.torchinfo_inspector.basic_summary_text import basic_summary_text
from midrun.torch_debug.manager import TorchDebugManager

if TYPE_CHECKING:
    from fastapi import FastAPI

    from midrun.fuse.filesystem import MidrunFS

_ALLOWED_DURING_HOLD = frozenset(
    {
        "checkpoint",
        "snapshot",
        "set_learning_rate",
        "reset_momentum",
        "set_dataloader_finite_check",
        "set_hparam",
        "stop",
        "torchinfo_capture",
        "profile",
        "torchview_capture",
        "torchviz_capture",
        "torchlens_capture",
        "set_lightning_profiler",
    }
)
# The command kinds that every rank runs, because they change state that each rank
# holds a copy of. The other kinds act on rank 0's services and run there only.
_ALL_RANK_KINDS = frozenset(
    {
        "stop",
        "checkpoint",
        "set_learning_rate",
        "reset_momentum",
        "set_dataloader_finite_check",
        "set_hparam",
    }
)
_ALLOWED_DURING_EXCEPTION = frozenset(
    {
        "save_current_batch",
        "save_current_loss",
        "save_partial_gradients",
        "save_optimizer_summary",
        "release_exception",
        "torchviz_capture",
    }
)
_KNOWN_SAFE_BACKWARD_STRATEGIES = frozenset(
    {"SingleDeviceStrategy", "DDPStrategy", "FSDPStrategy"}
)


class MidrunRuntime:
    """Coordinates state publication, command execution, and Midrun's services.

    Attributes:
        config: Effective `MidrunConfig` for this run.
        backend: The Lightning backend the composed model uses.
        run_id: Unique id for this training run.
        started_at: Unix timestamp this runtime object was constructed.
        state: Published `RuntimeSnapshot` store.
        command_queue: Submitted control-plane commands.
        gate: Cooperative hold gate.
        artifacts: Checkpoint/snapshot/debug-capture disk store; created once
            `setup_environment` resolves its final directory, not before.
        torch_debug: PyTorch distributed debug server manager.
        pipeline: Rolling per-batch phase-timing tracker.
        current_step: Live batch/loss/exception for the in-flight training step.
        finite_guard: Opt-in dataloader-batch finite-value checker.
        cuda_memory_history: Current CUDA allocator-history recording configuration.
        profiler_run: The armed/active bounded `torch.profiler` run, if any.
        lightning_profiler: The Lightning profiler instance currently installed as
            `trainer.profiler` by `set_lightning_profiler`, if any. Kept as a direct
            reference so its live `.summary()` can be read without a `trainer` handle.
        lightning_profiler_level: `off`, `simple`, `advanced`, or `pytorch` -- the
            level `lightning_profiler` was last set to.
        live_captures: Broker for cheap, non-persisted live views served at
            `SafePoint.BEFORE_OPTIMIZER_STEP` -- see
            `midrun.runtime.virtual_fs.live_files.LIVE_FILES`.
        tunable_hparams: Current value/type of each hparam the model opted into via
            `midrun_tunable_hparams`, refreshed at every safe point and in a hold.
            A tensor's value is read when this property is read, not at the safe
            point. Entries declared via the dict form also carry `min`/`max`/`label`.
        rank_sync: Control-plane collectives with the other ranks of a distributed
            run.
        services: The REST, FUSE and torch debug servers and their handles.
        sessions: The logging session and hooks session of the current stage.
    """

    _unreadable_hparams_reported: set[str]
    rank_sync: RankSync
    profiler_run: ProfilerRun | None

    def __init__(self, config: MidrunConfig, backend: LightningBackend) -> None:
        self.config: MidrunConfig = config
        self.backend: LightningBackend = backend
        # commands.next_id is process-local, so two processes would share a run id.
        self.run_id: str = (
            f"run-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{secrets.token_hex(4)}"
        )
        self.started_at: float = time.time()
        self.state: StateStore = StateStore(
            initial_snapshot(self.run_id, backend.name, backend.version)
        )
        self.command_queue: CommandQueue = CommandQueue(validate_command_args)
        self.gate: RunGate = RunGate()
        self.artifacts: ArtifactStore | None = None
        self.torch_debug: TorchDebugManager = TorchDebugManager(
            port=config.torch_debug_port, port_search=config.torch_debug_port_search
        )

        self._lock = threading.Lock()
        self._started = False
        self._is_rank_zero = True

        self.services = RuntimeServices(
            config,
            self.state,
            self.torch_debug,
            run_id=self.run_id,
            build_app=self._build_rest_app,
            build_fs=self._build_fuse_ops,
        )
        self.sessions = StageSessions(config, self.run_id)

        self._model_class_name: str | None = None
        self._model_hparams: dict[str, Any] = {}
        self._model_summary: str = ""

        self._local_rank_info: RankInfo | None = None
        self.rank_sync = RankSync()

        self.pipeline: PipelinePressureTracker = PipelinePressureTracker(
            window=config.pipeline_pressure_window
        )
        self.current_step: CurrentStep = CurrentStep()
        self.finite_guard: DataloaderFiniteGuard = DataloaderFiniteGuard(
            enabled=config.dataloader_finite_check_enabled
        )
        self._exception_hold: ExceptionBreakpointInfo | None = None
        self._model_summary_json: dict[str, Any] = {}
        self.cuda_memory_history: CudaMemoryHistoryState = CudaMemoryHistoryState(
            enabled=False, max_entries=None, started_at=None
        )
        self.profiler_run = None
        self.lightning_profiler: Any | None = None
        self.lightning_profiler_level: str = "off"
        self.live_captures: LiveCaptureBroker = LiveCaptureBroker()
        self._tunable_hparams: dict[str, tuple[HparamValue, torch.Tensor | None]] = {}
        self._tunable_hparam_specs: dict[str, TunableHparamSpec] = {}
        self._unreadable_hparams_reported = set()

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        """Rejects pickling and copying with a message that names the fix."""
        raise TypeError(
            "MidrunRuntime cannot be pickled or copied, because it owns threads,"
            " locks and servers. Lightning pickles the model to start the processes"
            " of strategy='ddp_spawn' and the other spawn-based strategies, which"
            " midrun does not support; use strategy='ddp'."
        )

    @property
    def rest_url(self) -> str | None:
        """The reachable REST base URL, once started."""
        return self.services.rest_url

    @property
    def _rest_port(self) -> int | None:
        return self.services.rest_port

    @property
    def fuse_mountpoint(self) -> Path | None:
        """The active FUSE mountpoint, once started."""
        return self.services.fuse_mountpoint

    @property
    def tunable_hparams(self) -> dict[str, HparamValue]:
        """See the class attributes.

        The REST and FUSE threads read this, so a safe point never copies a tensor to
        the host and waits for its device.
        """
        return {
            name: described
            if tensor is None
            else replace(described, value=describe_hparam(tensor).value)
            for name, (described, tensor) in self._tunable_hparams.items()
        }

    @property
    def _hook_session(self) -> HookSession | None:
        return self.sessions.hook_session

    @property
    def _logging_session(self) -> LoggingSession | None:
        return self.sessions.logging_session

    def _build_rest_app(self) -> "FastAPI":
        from midrun.rest.app import build_app

        return build_app(self)

    def _build_fuse_ops(self) -> "MidrunFS":
        from midrun.fuse.filesystem import MidrunFS

        return MidrunFS(self)

    @property
    def artifact_store(self) -> ArtifactStore:
        """The artifact store that `setup_environment` creates.

        Raises:
            RuntimeError: `setup_environment` has not created the store yet.
        """
        if self.artifacts is None:
            raise RuntimeError(
                "midrun: the artifact store does not exist until setup_environment runs"
            )
        return self.artifacts

    @property
    def on_rank_zero(self) -> bool:
        """Whether this process is global rank 0, as `setup_environment` found."""
        return self._is_rank_zero

    # ---- Lightning lifecycle ------------------------------------------------

    def setup_environment(self, pl_module: Any, stage: str) -> None:
        """Starts Midrun's services. Idempotent; safe to call for every Lightning stage."""
        del stage
        with self._lock:
            if self._started:
                return
            self._started = True
        # A new run or stage accepts the holds that a stop of the last one refused.
        self.command_queue.refusals.clear()

        self._is_rank_zero = is_rank_zero()
        self.rank_sync = RankSync.create()
        self._gather_ranks(pl_module)
        self._detect_backward_interception(pl_module)

        # `trainer.log_dir` is a collective under DDP, so every rank resolves it here,
        # before any rank-0-only start.
        base_dir = default_base_dir(pl_module, self.run_id)
        self._guarded(
            "artifact directory", lambda: self._resolve_artifact_dir(base_dir)
        )

        # torch.distributed.debug needs an initialized process group; a plain
        # single-process run has none, and must still get its REST server.
        torch_debug = self.config.torch_debug_enabled and distributed_initialized()
        if self.config.torch_debug_enabled and not torch_debug:
            logger.info(
                "midrun: pytorch distributed debug server skipped: "
                "torch.distributed is not initialized"
            )
        if self._is_rank_zero and torch_debug and self.config.rest_enabled:
            self._guarded(
                "REST server + pytorch distributed debug server (paired ports)",
                self.services.start_rest_and_torch_debug,
            )
        else:
            if torch_debug:
                self._guarded(
                    "pytorch distributed debug server", self.services.start_torch_debug
                )
            if self._is_rank_zero and self.config.rest_enabled:
                self._guarded("REST server", self.services.start_rest)
        if self._is_rank_zero and self.config.fuse_enabled:
            self._guarded("FUSE mount", lambda: self.services.start_fuse(base_dir))

        self.state.update(status="running")

    def on_fit_start(self, pl_module: Any) -> None:
        """Publishes the first full-fidelity snapshot once the Trainer is attached."""
        self._model_class_name = type(pl_module).__name__
        self._model_hparams = dict(getattr(pl_module, "hparams", {}))
        self._model_summary = basic_summary_text(pl_module)
        self._model_summary_json = basic_summary_json(pl_module)
        self._tunable_hparam_specs = collect_tunable_hparams(type(pl_module))
        self._publish(pl_module, status="held" if self.gate.is_held() else "running")

    def teardown(self, pl_module: Any, stage: str) -> None:
        """Stops Midrun's services. Idempotent; releases any dangling hold first."""
        del pl_module, stage
        self.current_step.clear()
        self._stop_services(final_status="finished")

    def stop_after_failure(self) -> None:
        """Stops Midrun's services after an uncaught exception; the final status is `failed`."""
        self.current_step.clear()
        self._stop_services(final_status="failed")

    def _stop_services(self, final_status: str) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
        self.state.update(status="stopping")
        # A command queued for this stage must not run in the next one.
        self.command_queue.end_queued(
            "expired", error="the stage stopped before the command ran"
        )
        self.gate.release()
        self._guarded("REST shutdown", self._stop_rest)
        if self.cuda_memory_history.enabled:
            # Recording is process-global: it would outlive the stage. REST is
            # down, so nothing turns it on again.
            self.cuda_memory_history = set_cuda_memory_history(False, max_entries=None)
        self._guarded("FUSE shutdown", self._stop_fuse)
        self._guarded("pytorch distributed debug shutdown", self.torch_debug.stop)
        if (run := self.profiler_run) is not None:
            self.profiler_run = None
            self._guarded("profiler shutdown", lambda: run.save(self.artifact_store))
        self._guarded("control group shutdown", self.rank_sync.close)
        self.state.update(status=final_status)

    def _stop_rest(self) -> None:
        self.services.stop_rest()

    def _stop_fuse(self) -> None:
        self.services.stop_fuse()

    # ---- lifecycle hooks --------------------------------------------------------

    def open_logging_session(self, trainer: Any, pl_module: Any) -> LoggingSession:
        """Opens this stage `_run`'s logging session; see `StageSessions`."""
        return self.sessions.open_logging_session(trainer, pl_module)

    def close_logging_session(self) -> None:
        """Closes the logging session, if one is open; see `StageSessions`."""
        self.sessions.close_logging_session()

    def ensure_hook_session(self, trainer: Any, pl_module: Any) -> None:
        """Bootstraps this run's hooks tree for one stage `_run`; see `StageSessions`."""
        self.sessions.ensure_hook_session(trainer, pl_module)

    def dispatch_hook(
        self,
        trainer: Any,
        hook: str,
        *,
        stage_override: str | None = None,
        arguments: dict[str, Any] | None = None,
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
    ) -> HandleResult | None:
        """Dispatches one Lightning callback occurrence; see `StageSessions`."""
        return self.sessions.dispatch_hook(
            trainer,
            hook,
            stage_override=stage_override,
            arguments=arguments,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def close_hook_session(self) -> None:
        """Closes the hooks session, if one is open; see `StageSessions`."""
        self.sessions.close_hook_session()

    # ---- safe points ----------------------------------------------------------

    def process_safe_point(
        self, pl_module: Any, safe_point: SafePoint, batch_idx: int | None = None
    ) -> None:
        """Publishes state, executes due commands, and blocks here if a hold is active.

        Called only from the Lightning training thread, at one of the safe points
        listed in `midrun.runtime.events.SafePoint`. In a distributed run, every
        rank calls it at the same safe points, because it is a collective.
        """
        trainer = pl_module.trainer
        held = self._run_due_commands(pl_module, trainer, safe_point, allowed=None)
        if safe_point is SafePoint.BEFORE_OPTIMIZER_STEP:
            self.live_captures.serve(pl_module)
        self._publish(
            pl_module, status="held" if held else "running", batch_idx=batch_idx
        )
        self._refresh_tunable_hparams(pl_module, trainer)
        if held:
            self._hold_loop(pl_module, trainer)

    def _refresh_tunable_hparams(self, pl_module: Any, trainer: Any) -> None:
        """Rebuilds `tunable_hparams` from each registered knob's current value.

        Reassigns the whole dict rather than mutating it in place, matching
        `cuda_memory_history`/`profiler_run`'s thread-safety idiom. A single
        unreadable knob is logged once and skipped rather than blanking the rest.
        """
        fresh: dict[str, tuple[HparamValue, torch.Tensor | None]] = {}
        for name, spec in self._tunable_hparam_specs.items():
            try:
                parent, leaf = resolve_hparam_path(pl_module, trainer, spec.path)
                value = get_field(parent, leaf)
                described = describe_hparam(value, read_tensor=False)
                fresh[name] = (
                    replace(described, min=spec.min, max=spec.max, label=spec.label),
                    value if isinstance(value, torch.Tensor) else None,
                )
            except Exception as exc:  # noqa: BLE001 -- one broken knob must not blank the dict
                if name not in self._unreadable_hparams_reported:
                    self._unreadable_hparams_reported.add(name)
                    logger.warning(
                        "midrun tunable hparam {!r} unreadable: {}", name, exc
                    )
        self._tunable_hparams = fresh

    def _hold_loop(self, pl_module: Any, trainer: Any) -> None:
        """Blocks every rank while rank 0's gate is held.

        Rank 0 sends its hold state about ten times a second, so the other ranks wait
        in the control group's broadcast and not in a training collective, which
        would reach the process-group timeout.
        """
        while self._run_due_commands(
            pl_module, trainer, safe_point=None, allowed=_ALLOWED_DURING_HOLD
        ):
            self._publish(pl_module, status="held")
            self._refresh_tunable_hparams(pl_module, trainer)
            if self._is_rank_zero:
                self.gate.wait_for_control_event(timeout=0.1)
        self._publish(pl_module, status="running")

    def _run_due_commands(
        self,
        pl_module: Any,
        trainer: Any,
        safe_point: SafePoint | None,
        allowed: frozenset[str] | None,
    ) -> bool:
        """Runs the due commands on every rank and returns whether rank 0 is held.

        Only rank 0 has the command queue and the gate, so it sends its due commands
        and its hold state to the other ranks. They run only `_ALL_RANK_KINDS`. This
        is a collective: every rank must call it at the same safe point.
        """
        due = self._due_commands(
            safe_point, trainer.current_epoch, trainer.global_step, allowed
        )
        commands, held = self.rank_sync.broadcast((due, self.gate.is_held()))
        if not commands:
            return held
        for command in commands:
            if self._is_rank_zero or command.kind in _ALL_RANK_KINDS:
                self._execute(pl_module, trainer, command, safe_point)
        # A command can enter or release a hold, so rank 0 sends its state again.
        return self.rank_sync.broadcast(self.gate.is_held())

    def _due_commands(
        self,
        safe_point: SafePoint | None,
        epoch: int,
        global_step: int,
        allowed: frozenset[str] | None,
    ) -> tuple[Command, ...]:
        return tuple(self.command_queue.due(safe_point, epoch, global_step, allowed))

    def _execute(
        self,
        pl_module: Any,
        trainer: Any,
        command: Command,
        safe_point: SafePoint | None,
    ) -> None:
        # Skips a hold cancelled after `_due_commands` returned it. Only holds can be
        # cancelled, and only rank 0 runs them, so no rank skips what the others run.
        if not self.command_queue.mark_executing(command):
            return
        # The queue validates the kind at submit, so every queued kind has a handler.
        handler = COMMAND_HANDLERS[command.kind]
        try:
            result = handler(self, pl_module, trainer, command, safe_point)
        except Exception as exc:  # noqa: BLE001 -- command boundary: untrusted REST input must not crash training
            logger.warning(
                "midrun command {} ({}) failed: {}", command.id, command.kind, exc
            )
            self.command_queue.mark_result(command, succeeded=False, error=str(exc))
            return
        except BaseException as exc:
            # An interrupt or exit ends training, but not as a command still executing.
            error = f"interrupted by {type(exc).__name__}"
            self.command_queue.mark_result(command, succeeded=False, error=error)
            raise
        self.command_queue.mark_result(command, succeeded=True, result=result)

    # ---- per-batch hooks ----------------------------------------------------------

    def begin_train_batch(self, pl_module: Any, batch: Any, batch_idx: int) -> None:
        """Records batch-start timing/state. Called from `on_train_batch_start`.

        Enters a cooperative hold directly (bypassing the command queue, like
        `handle_hold` does from inside a command) if the dataloader finite guard is
        enabled and finds a non-finite tensor in `batch` on any rank.
        """
        trainer = pl_module.trainer
        now = time.perf_counter_ns()
        self.pipeline.batch_start(
            now, trainer.current_epoch, trainer.global_step, batch_idx
        )
        self.current_step.begin(
            batch, batch_idx, trainer.current_epoch, trainer.global_step, now
        )
        if self.finite_guard.enabled:
            report = self.finite_guard.check(batch)
            # Every rank holds when one rank's batch is bad, so a collective call.
            if self.rank_sync.any_rank(report is not None):
                self._handle_bad_batch(pl_module, trainer, batch_idx, report)

    def before_backward(self, loss: Any) -> None:
        """Records the loss and pre-backward timestamp. Called from `on_before_backward`."""
        self.current_step.loss = loss
        self.pipeline.before_backward(time.perf_counter_ns())

    def after_backward(self) -> None:
        """Records the post-backward timestamp. Called from `on_after_backward`."""
        self.pipeline.after_backward(time.perf_counter_ns())

    def finish_train_batch(self, pl_module: Any) -> None:
        """Finalizes this batch's pipeline-pressure sample and clears `current_step`.

        Called from `on_train_batch_end`, after safe-point processing, in a `finally`
        block so the retained batch/loss/exception never survive into the next batch.
        """
        if self.config.pipeline_pressure_enabled:
            self.pipeline.batch_end(time.perf_counter_ns())
        self._advance_profiler(pl_module)
        self.current_step.clear()

    def _advance_profiler(self, pl_module: Any) -> None:
        """Steps the active `ProfilerRun`, if any, finalizing it once due.

        Runs inside the hot per-batch hook path, not a command handler, so this is a
        dedicated always-swallowing boundary rather than `_guarded` -- `_guarded`
        re-raises under `inspection_strict`, which would let an inspector failure
        crash live training.
        """
        run = self.profiler_run
        if run is None:
            return
        try:
            done = run.step()
        except Exception as exc:  # noqa: BLE001 -- inspector must never destabilize training
            logger.warning("midrun: profiler step failed: {}", exc)
            self.profiler_run = None
            return
        if not done:
            return
        try:
            run.save(self.artifact_store)
        except Exception as exc:  # noqa: BLE001 -- inspector must never destabilize training
            logger.warning("midrun: profiler finalize failed: {}", exc)
        self.profiler_run = None
        if run.then_hold:
            trainer = pl_module.trainer
            self.gate.enter(
                next_id("profiler"),
                "train_batch_end",
                trainer.current_epoch,
                trainer.global_step,
            )
            self._publish(pl_module, status="held")
            # Only rank 0 profiles, so with more ranks the hold starts at the next
            # safe point, where every rank learns of it.
            if self.rank_sync.world_size == 1:
                self._hold_loop(pl_module, trainer)

    def _handle_bad_batch(
        self,
        pl_module: Any,
        trainer: Any,
        batch_idx: int,
        report: FiniteCheckReport | None,
    ) -> None:
        """Records this rank's bad batch, if any, then holds every rank in any case."""
        if report is not None:
            record_bad_batch(
                self.artifacts,
                report,
                rank=self._local_rank_info.global_rank if self._local_rank_info else 0,
                epoch=trainer.current_epoch,
                global_step=trainer.global_step,
                batch_idx=batch_idx,
            )
        if self._is_rank_zero:
            self.gate.enter(
                next_id("anomaly"),
                "train_batch_start",
                trainer.current_epoch,
                trainer.global_step,
            )
        self._publish(pl_module, status="held", batch_idx=batch_idx)
        self._hold_loop(pl_module, trainer)

    # ---- exception breakpoint ------------------------------------------------------

    def handle_backward_exception(
        self, pl_module: Any, loss: Any, exc: Exception
    ) -> None:
        """Freezes the failing batch/loss and holds for inspection before re-raising.

        Called from `MidrunMixin.backward`'s `except` clause; no-ops if
        `break_on_backward_exception` is disabled. Diagnostic only: gradient state at
        this point is not a valid complete gradient set, and distributed collectives
        may already be inconsistent across ranks -- see `ExceptionBreakpointInfo`.

        Only rank 0 holds: the other ranks have no REST server, so nothing could
        release their hold, and they re-raise at once.
        """
        if not self.config.break_on_backward_exception or not self._is_rank_zero:
            return
        trainer = pl_module.trainer
        self.current_step.loss = loss
        self.current_step.exception = exc
        rank = self._local_rank_info.global_rank if self._local_rank_info else 0
        self._exception_hold = ExceptionBreakpointInfo(
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            epoch=trainer.current_epoch,
            global_step=trainer.global_step,
            batch_idx=self.current_step.batch_idx,
            rank=rank,
            started_at=time.time(),
        )
        self._publish(pl_module, status="exception_held")
        try:
            self._exception_hold_loop(pl_module, trainer)
        finally:
            # An interrupt also ends the hold. The caller re-raises: drop the live
            # references, the traceback frames among them.
            self._exception_hold = None
            self.state.update(exception=None)
            self.current_step.clear()
        self._publish(pl_module, status="running")

    def _exception_hold_loop(self, pl_module: Any, trainer: Any) -> None:
        # Only rank 0 holds here, so this loop runs its commands without a collective.
        while self._exception_hold is not None:
            for command in self._due_commands(
                None,
                trainer.current_epoch,
                trainer.global_step,
                allowed=_ALLOWED_DURING_EXCEPTION,
            ):
                self._execute(pl_module, trainer, command, safe_point=None)
            self._publish(pl_module, status="exception_held")
            if self._exception_hold is None:
                break
            self.gate.wait_for_control_event(timeout=0.1)

    def _detect_backward_interception(self, pl_module: Any) -> None:
        """Probes whether the active strategy is known to route backward through
        `LightningModule.backward()`, via an explicit allowlist (not runtime
        bypass-detection, which is unreliable).
        """
        trainer = getattr(pl_module, "trainer", None)
        strategy = getattr(trainer, "strategy", None) if trainer is not None else None
        if strategy is None:
            return
        strategy_name = type(strategy).__name__
        if strategy_name in _KNOWN_SAFE_BACKWARD_STRATEGIES:
            return
        reason = (
            f"strategy {strategy_name!r} is not in the known-safe allowlist for "
            "LightningModule.backward() interception"
        )
        self.state.update(
            backward_interception_supported=False, backward_interception_reason=reason
        )
        if self.config.break_on_backward_exception:
            logger.warning(
                "midrun: break_on_backward_exception requested but {}", reason
            )

    # ---- state publication ------------------------------------------------------

    def _publish(
        self, pl_module: Any, status: str, batch_idx: int | None = None
    ) -> None:
        trainer = pl_module.trainer
        metrics: dict[str, ScalarMetric] = {}
        for key, value in trainer.callback_metrics.items():
            try:
                metrics[key] = ScalarMetric(
                    value=float(value),
                    epoch=trainer.current_epoch,
                    step=trainer.global_step,
                    updated_at=time.time(),
                )
            except (TypeError, ValueError):
                continue
        learning_rates = tuple(
            float(group["lr"])
            for optimizer in trainer.optimizers
            for group in optimizer.param_groups
        )
        stage = trainer.state.stage.value if trainer.state.stage is not None else None
        if batch_idx is None and self.current_step.batch is not None:
            # A publish inside a batch, such as from a hold, shows that batch.
            batch_idx = self.current_step.batch_idx
        pending = len(self.command_queue.list_all(status="queued"))
        self.state.update(
            status=status,
            stage=stage,
            phase=phase_of_stage(stage),
            epoch=trainer.current_epoch,
            global_step=trainer.global_step,
            batch_idx=batch_idx,
            ranks=self._current_ranks(),
            metrics=metrics,
            learning_rates=learning_rates,
            pending_commands=pending,
            hold_state="held" if self.gate.is_held() else "none",
            exception=self._exception_hold,
        )

    def _current_rank_info(self) -> RankInfo:
        """Refreshes `worker_pids` on the rank info gathered once at setup.

        DataLoader workers are spawned lazily and non-persistent ones churn per
        epoch, so this field alone is recomputed on every publish rather than
        gathered once like the rest of `RankInfo`.
        """
        import multiprocessing

        assert self._local_rank_info is not None, (
            "_gather_ranks must run before the first _publish"
        )
        worker_pids = tuple(
            sorted(
                child.pid
                for child in multiprocessing.active_children()
                if child.pid is not None
            )
        )
        return replace(self._local_rank_info, worker_pids=worker_pids)

    def _current_ranks(self) -> tuple[RankInfo, ...]:
        """Every rank's info, with this rank's `worker_pids` refreshed.

        The other ranks' entries are the ones gathered at setup, so their
        `worker_pids` are not live.
        """
        local = self._current_rank_info()
        return tuple(
            local if info.global_rank == local.global_rank else info
            for info in self.state.read().ranks
        )

    def _gather_ranks(self, pl_module: Any) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            global_rank = dist.get_rank()
        else:
            world_size = 1
            global_rank = 0
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = str(getattr(pl_module, "device", "cpu"))
        rank_info = RankInfo(
            global_rank=global_rank,
            local_rank=local_rank,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            device=device,
        )
        self._local_rank_info = rank_info
        ranks = tuple(self.rank_sync.all_gather(rank_info))
        self.state.update(world_size=world_size, ranks=ranks)

    # ---- service management ------------------------------------------------------

    def _guarded(self, description: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            if self.config.inspection_strict:
                raise
            logger.warning("midrun: {} failed: {}", description, exc)

    def _resolve_artifact_dir(self, base_dir: Path) -> None:
        artifact_dir = self.config.artifact_dir or base_dir / "artifacts"
        self.artifacts = ArtifactStore(artifact_dir)
        self.state.update(artifacts_path=str(artifact_dir))
