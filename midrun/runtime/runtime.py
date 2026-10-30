"""Owns Midrun's runtime services and dispatches Lightning-hook-driven state
publication and command execution.

The Lightning training thread is the only thread that mutates or deeply inspects
Trainer/model state. REST and FUSE only read published `RuntimeSnapshot`s or enqueue
`Command`s; this module is what actually executes them, from inside Lightning hooks.
"""

import os
import secrets
import shutil
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn, SupportsIndex

from loguru import logger

from midrun.config import MidrunConfig
from midrun.fuse import mountpoint_check, process_guard
from midrun.hooks.manager.bootstrap_hooks_tree import bootstrap_hooks_tree
from midrun.hooks.session import HandleResult, HookSession
from midrun.lightning_backend import LightningBackend
from midrun.logging.logging_session import LoggingSession
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.bind_free_socket import bind_free_socket
from midrun.runtime.bind_socket import bind_socket
from midrun.runtime.command_args.validate_command_args import validate_command_args
from midrun.runtime.commands.command import Command
from midrun.runtime.commands.command_queue import CommandQueue
from midrun.runtime.commands.next_id import next_id
from midrun.runtime.cuda_memory.cuda_memory_history_state import (
    CudaMemoryHistoryState,
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
from midrun.runtime.new_hooks_session_id import new_hooks_session_id
from midrun.runtime.phase_of_stage import phase_of_stage
from midrun.runtime.pipeline_pressure import PipelinePressureTracker
from midrun.runtime.profiler import ProfilerRun
from midrun.runtime.rank_sync import RankSync
from midrun.runtime.record_bad_batch import record_bad_batch
from midrun.runtime.rest_base_url import rest_base_url
from midrun.runtime.start_logging_session import start_logging_session
from midrun.runtime.start_rest_server import start_rest_server
from midrun.runtime.state.exception_breakpoint_info import ExceptionBreakpointInfo
from midrun.runtime.state.initial_snapshot import initial_snapshot
from midrun.runtime.state.rank_info import RankInfo
from midrun.runtime.state.scalar_metric import ScalarMetric
from midrun.runtime.state.state_store import StateStore
from midrun.runtime.torchinfo_inspector.basic_summary_json import basic_summary_json
from midrun.runtime.torchinfo_inspector.basic_summary_text import basic_summary_text
from midrun.runtime.trainer_stage_name import trainer_stage_name
from midrun.torch_debug.manager import TorchDebugManager

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
            Entries declared via the dict form also carry `min`/`max`/`label`.
        rank_sync: Control-plane collectives with the other ranks of a distributed
            run.
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

        self._fuse_ops: Any = None
        self._fuse_thread: threading.Thread | None = None
        self._fuse_mountpoint: Path | None = None
        self._fuse_watcher: Any = None

        self._rest_server: Any = None
        self._rest_thread: threading.Thread | None = None
        self._rest_port: int | None = None

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
        self.tunable_hparams: dict[str, HparamValue] = {}
        self._tunable_hparam_specs: dict[str, TunableHparamSpec] = {}
        self._unreadable_hparams_reported = set()

        self._hook_session: HookSession | None = None
        self._logging_session: LoggingSession | None = None

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
        if self._rest_port is None:
            return None
        return rest_base_url(self.config.rest_host, self._rest_port)

    @property
    def fuse_mountpoint(self) -> Path | None:
        """The active FUSE mountpoint, once started."""
        return self._fuse_mountpoint

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
                self._start_rest_and_torch_debug,
            )
        else:
            if torch_debug:
                self._guarded(
                    "pytorch distributed debug server", self._start_torch_debug
                )
            if self._is_rank_zero and self.config.rest_enabled:
                self._guarded("REST server", self._start_rest)
        if self._is_rank_zero and self.config.fuse_enabled:
            self._guarded("FUSE mount", lambda: self._start_fuse(base_dir))

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
        self._stop_services(final_status="finished")

    def stop_after_failure(self) -> None:
        """Stops Midrun's services after an uncaught exception; the final status is `failed`."""
        self._stop_services(final_status="failed")

    def _stop_services(self, final_status: str) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
        self.state.update(status="stopping")
        self.gate.release()
        self._guarded("REST shutdown", self._stop_rest)
        self._guarded("FUSE shutdown", self._stop_fuse)
        self._guarded("pytorch distributed debug shutdown", self.torch_debug.stop)
        if (run := self.profiler_run) is not None:
            self.profiler_run = None
            self._guarded("profiler shutdown", lambda: run.save(self.artifact_store))
        self.state.update(status=final_status)

    # ---- lifecycle hooks --------------------------------------------------------

    def open_logging_session(self, trainer: Any, pl_module: Any) -> LoggingSession:
        """Opens this stage `_run`'s logging session, unless it is open, and returns it.

        Idempotent. The first Midrun callback `setup` of the `_run` opens it: the hooks
        adapter's, which needs its logger, else the service guard's. The mixin's
        `teardown` or the guard's `on_exception` closes it (`close_logging_session`).
        """
        if self._logging_session is None:
            # Set before the bridge or the file can fail, so that the close finds it.
            self._logging_session = LoggingSession(new_hooks_session_id())
            start_logging_session(
                self._logging_session,
                self.config,
                default_base_dir(pl_module, self.run_id),
                trainer.global_rank,
            )
        return self._logging_session

    def close_logging_session(self) -> None:
        """Closes the logging session, if one is open. Idempotent."""
        if self._logging_session is not None:
            self._logging_session.close()
            self._logging_session = None

    def ensure_hook_session(self, trainer: Any, pl_module: Any) -> None:
        """Bootstraps this run's hooks tree for one stage `_run`.

        Idempotent -- a no-op if a session from this same stage is already open.
        Called only from the hooks adapter's own `setup(trainer, pl_module, stage)`,
        which Lightning always calls before any other callback method for that
        `_run` and always pairs with a matching `teardown` (barring an uncaught
        exception, handled separately by `close_hook_session`). A hook that
        fires with no stage active at all (e.g. a standalone `Trainer.save_checkpoint`
        call outside `fit`/`validate`/`test`/`predict`) finds no session and is a
        silent no-op in `dispatch_hook` -- there is no run for it to belong to.
        Reads rank/world size directly from `trainer`: `_local_rank_info` is not
        populated yet at this point (the adapter's `setup` runs before the composed
        model's own `setup`, which is what triggers `_gather_ranks`).
        A resumed run dispatches with the session id that the marker records; the
        new id names only this stage's logging session and a freshly published tree.
        """
        if self._hook_session is not None:
            return
        logging_session = self.open_logging_session(trainer, pl_module)
        session_id = logging_session.session_id
        base_dir = default_base_dir(pl_module, self.run_id)
        log = logging_session.bind()
        shell = self.config.hooks_shell
        if shell is None:
            resolved = shutil.which("bash")
            if resolved is None:
                raise RuntimeError(
                    "midrun: hooks_enabled requires bash on PATH or an explicit hooks_shell"
                )
            shell = Path(resolved)
        setpriv = shutil.which("setpriv")
        tree = bootstrap_hooks_tree(
            base_dir, self.config.hooks_source_dir, session_id=session_id, log=log
        )
        self._hook_session = HookSession(
            tree.live_dir,
            session_id=tree.session_id,
            rank=trainer.global_rank,
            world_size=trainer.world_size,
            rank_policy=self.config.hooks_rank_policy,
            shell=shell,
            setpriv=None if setpriv is None else Path(setpriv),
            timeout_s=self.config.hooks_timeout_s,
            max_params_bytes=self.config.hooks_max_params_bytes,
            max_metadata_items=self.config.hooks_max_metadata_items,
            max_metadata_depth=self.config.hooks_max_metadata_depth,
            max_export_bytes=self.config.hooks_max_export_bytes,
            max_output_bytes=self.config.hooks_max_output_bytes,
            log=log,
        )

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
        """Dispatches one Lightning callback occurrence through the live `HookSession`.

        A no-op if `ensure_hook_session` never installed a session (hooks disabled).
        """
        if self._hook_session is None:
            return None
        stage = (
            stage_override
            if stage_override is not None
            else trainer_stage_name(trainer)
        )
        return self._hook_session.handle(
            hook,
            stage=stage,
            epoch=trainer.current_epoch,
            global_step=trainer.global_step,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            arguments=arguments,
        )

    def close_hook_session(self) -> None:
        """Closes the hooks session, if one is open.

        Idempotent. Called from the hooks adapter's own `teardown` on the normal
        path, and from its `on_exception` on the failure path, since Lightning skips
        the callback's own `teardown` after an uncaught exception.
        """
        if self._hook_session is not None:
            self._hook_session.close()
            self._hook_session = None

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
        fresh: dict[str, HparamValue] = {}
        for name, spec in self._tunable_hparam_specs.items():
            try:
                parent, leaf = resolve_hparam_path(pl_module, trainer, spec.path)
                described = describe_hparam(get_field(parent, leaf))
                fresh[name] = replace(
                    described, min=spec.min, max=spec.max, label=spec.label
                )
            except Exception as exc:  # noqa: BLE001 -- one broken knob must not blank the dict
                if name not in self._unreadable_hparams_reported:
                    self._unreadable_hparams_reported.add(name)
                    logger.warning(
                        "midrun tunable hparam {!r} unreadable: {}", name, exc
                    )
        self.tunable_hparams = fresh

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
        handler = COMMAND_HANDLERS.get(command.kind)
        if handler is None:
            self.command_queue.mark_result(
                command, succeeded=False, error=f"unknown command kind {command.kind!r}"
            )
            return
        try:
            result = handler(self, pl_module, trainer, command, safe_point)
        except Exception as exc:  # noqa: BLE001 -- command boundary: untrusted REST input must not crash training
            logger.warning(
                "midrun command {} ({}) failed: {}", command.id, command.kind, exc
            )
            self.command_queue.mark_result(command, succeeded=False, error=str(exc))
            return
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
        self._exception_hold_loop(pl_module, trainer)
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
        pending = sum(
            1 for command in self.command_queue.list_all() if command.status == "queued"
        )
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

    def _start_torch_debug(self) -> None:
        self.torch_debug.start_after_distributed_init()
        self.state.update(torch_debug_url=self.torch_debug.ready_url(timeout=10))

    def _start_rest_and_torch_debug(self) -> None:
        """Starts REST and the torch_debug frontend as one paired unit: both ports are
        offset from their configured base port by the same multiple of `_PORT_OFFSET_STEP`
        (e.g. offset 200 pairs REST 8290 with torch_debug 26199) -- a human-legible
        pairing, and a way to keep the two services from independently drifting to
        unrelated ports when several runs share a host and start at nearly the same time.

        Per offset attempt: probe the torch_debug candidate (it can't be held -- see
        `TorchDebugManager.start_after_distributed_init`), then bind-and-hold the REST
        candidate at the same offset (race-free once held), then re-probe the
        torch_debug candidate immediately before starting it, narrowing (not
        eliminating -- an upstream constraint) its check-then-bind race to that final
        gap. A collision at either check abandons the offset and retries the next one.
        """
        max_attempts = self.config.paired_port_search_attempts
        for attempt in range(max_attempts):
            offset = attempt * _PORT_OFFSET_STEP
            debug_candidate = self.config.torch_debug_port + offset
            if not self.torch_debug.port_is_free(debug_candidate):
                continue
            try:
                sock = bind_socket(
                    self.config.rest_host, self.config.rest_port + offset
                )
            except OSError:
                continue
            if not self.torch_debug.port_is_free(debug_candidate):
                sock.close()
                continue
            try:
                self.torch_debug.start_after_distributed_init(port=debug_candidate)
                debug_url = self.torch_debug.ready_url(timeout=10)
            except Exception:
                # REST still starts, so the run stays controllable without the debug server.
                self._start_rest_on_socket(sock, self.config.rest_port + offset)
                raise
            self.state.update(torch_debug_url=debug_url)
            self._start_rest_on_socket(sock, self.config.rest_port + offset)
            return
        raise RuntimeError(
            f"no free paired REST/torch_debug port offset found in {max_attempts} "
            f"attempts (step {_PORT_OFFSET_STEP})"
        )

    def _resolve_artifact_dir(self, base_dir: Path) -> None:
        artifact_dir = self.config.artifact_dir or base_dir / "artifacts"
        self.artifacts = ArtifactStore(artifact_dir)
        self.state.update(artifacts_path=str(artifact_dir))

    def _start_fuse(self, base_dir: Path) -> None:
        import mfusepy

        from midrun.fuse.filesystem import MidrunFS

        mountpoint = mountpoint_check.prepare_mountpoint(
            self.config.fuse_mountpoint or base_dir / "procfs"
        )
        ops = MidrunFS(self)

        def _run() -> None:
            mfusepy.FUSE(
                ops,
                str(mountpoint),
                foreground=True,
                direct_io=True,
                ro=True,
                attr_timeout=0,
                fsname=mountpoint_check.FUSE_FSNAME,
            )

        thread = threading.Thread(target=_run, name="midrun-fuse", daemon=True)
        self._fuse_watcher = process_guard.mount_guarded(
            mountpoint, thread, ops.wait_ready
        )
        self._fuse_ops = ops
        self._fuse_thread = thread
        self._fuse_mountpoint = mountpoint

    def _stop_fuse(self) -> None:
        if self._fuse_thread is None:
            return
        import subprocess

        subprocess.run(["fusermount3", "-u", str(self._fuse_mountpoint)], check=False)  # noqa: S603, S607 -- own mount
        self._fuse_thread.join(timeout=5)
        if self._fuse_watcher is not None:
            # A byte on stdin tells the watcher to check once and exit.
            self._fuse_watcher.communicate(b"\n", timeout=5)
        self._fuse_ops = None
        self._fuse_thread = None
        self._fuse_mountpoint = None
        self._fuse_watcher = None

    def _start_rest(self) -> None:
        sock, port = bind_free_socket(
            self.config.rest_host, self.config.rest_port, self.config.rest_port_search
        )
        self._start_rest_on_socket(sock, port)

    def _start_rest_on_socket(self, sock: socket.socket, port: int) -> None:
        from midrun.rest.app import build_app

        server, thread = start_rest_server(
            build_app(self), sock, self.config.rest_host, port, self.run_id
        )
        self._rest_server = server
        self._rest_thread = thread
        self._rest_port = port
        self.state.update(rest_host=self.config.rest_host, rest_port=port)

    def _stop_rest(self) -> None:
        if self._rest_server is None:
            return
        self._rest_server.should_exit = True
        if self._rest_thread is not None:
            self._rest_thread.join(timeout=5)
            if self._rest_thread.is_alive():
                logger.warning("midrun: the REST server thread did not stop in 5 s")
        self._rest_server = None
        self._rest_thread = None
        self._rest_port = None


# Shared offset step for `MidrunRuntime._start_rest_and_torch_debug`'s paired port
# selection: REST and torch_debug candidates at attempt N are both `N * _PORT_OFFSET_STEP`
# above their configured base port (e.g. offset 200 pairs REST 8290 with debug 26199).
_PORT_OFFSET_STEP = 100
