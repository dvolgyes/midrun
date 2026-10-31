"""MidrunMixin: the sole Midrun integration surface for a Lightning `LightningModule`.

Compose it before either supported Lightning namespace's `LightningModule`:

    import lightning.pytorch as PL
    from midrun import MidrunMixin

    class MyModel(MidrunMixin, PL.LightningModule):
        ...

or the same with `import pytorch_lightning as PL`. The two Lightning namespaces must
not be mixed inside one model/Trainer runtime; `__init_subclass__` rejects a class
whose MRO spans both, and `setup()` rejects a Trainer from the other namespace.

If a subclass overrides one of the reserved hooks below (`setup`, `on_fit_start`,
`teardown`, `on_train_batch_start`, `on_before_backward`, `backward`,
`on_after_backward`, `on_train_batch_end`, `on_train_epoch_end`,
`on_validation_epoch_end`, `on_before_optimizer_step`, `on_fit_end`,
`configure_callbacks`), it must call `super()`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.detect_lightning_backend_from_mro import detect_lightning_backend_from_mro
from midrun.hooks.lightning import build_hooks_callback_class
from midrun.lightning_backend import LightningBackend
from midrun.runtime.events import SafePoint
from midrun.runtime.runtime import MidrunRuntime

# Type checkers see LightningModule as the base, so that the `super()` calls to its
# hooks type-check. At run time the base is `object`, so that the mixin composes
# with the LightningModule of either Lightning namespace.
if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch import LightningModule as _MixinBase
else:
    _MixinBase = object


class MidrunMixin(_MixinBase):
    """Starts and owns a `MidrunRuntime` for the Lightning model it is composed with.

    Attributes:
        __midrun_lightning_backend__: The Lightning backend detected from the
            subclass's MRO at class-definition time (`lightning.pytorch` or
            `pytorch_lightning`); set once per subclass by `__init_subclass__`.
        __midrun_hooks_callback_class__: This subclass's concrete lifecycle-hooks
            `Callback` adapter class, built once per subclass by `__init_subclass__`.
        __midrun_service_guard_class__: This subclass's concrete `Callback` class
            that stops Midrun's services after an uncaught exception, built once
            per subclass by `__init_subclass__`.
    """

    __midrun_lightning_backend__: LightningBackend
    __midrun_hooks_callback_class__: type
    __midrun_service_guard_class__: type

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Detects the Lightning backend and builds this subclass's hooks callback class."""
        super().__init_subclass__(**kwargs)
        cls.__midrun_lightning_backend__ = detect_lightning_backend_from_mro(
            cls.__mro__
        )
        cls.__midrun_hooks_callback_class__ = build_hooks_callback_class(
            cls.__midrun_lightning_backend__
        )
        cls.__midrun_service_guard_class__ = type(
            "MidrunServiceGuard",
            (_ServiceGuardMethods, cls.__midrun_lightning_backend__.callback_class),
            {},
        )

    def __init__(
        self,
        *args: Any,
        midrun_enabled: bool = True,
        fuse_enabled: bool = True,
        fuse_mountpoint: Path | str | None = None,
        rest_enabled: bool = True,
        rest_host: str = "127.0.0.1",
        rest_port: int = 8090,
        rest_port_search: int = 100,
        rest_allowed_hosts: Sequence[str] = (),
        torch_debug_enabled: bool = True,
        torch_debug_port: int = 25999,
        torch_debug_port_search: int = 100,
        paired_port_search_attempts: int = 20,
        artifact_dir: Path | str | None = None,
        inspection_strict: bool = False,
        pipeline_pressure_enabled: bool = True,
        pipeline_pressure_window: int = 256,
        break_on_backward_exception: bool = True,
        dataloader_finite_check_enabled: bool = False,
        optimizer_surgery_enabled: bool = True,
        hooks_enabled: bool = True,
        hooks_source_dir: Path | str | None = None,
        hooks_timeout_s: float | None = None,
        hooks_rank_policy: str = "rank_zero",
        hooks_shell: Path | str | None = None,
        hooks_max_params_bytes: int = 256 * 1024,
        hooks_max_metadata_items: int = 1000,
        hooks_max_metadata_depth: int = 8,
        hooks_max_export_bytes: int = 1024**3,
        hooks_max_output_bytes: int = 64 * 1024,
        intercept_lightning_logging: bool = True,
        run_log_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        config = MidrunConfig(
            enabled=midrun_enabled,
            fuse_enabled=fuse_enabled,
            fuse_mountpoint=None if fuse_mountpoint is None else Path(fuse_mountpoint),
            rest_enabled=rest_enabled,
            rest_host=rest_host,
            rest_port=rest_port,
            rest_port_search=rest_port_search,
            rest_allowed_hosts=tuple(rest_allowed_hosts),
            torch_debug_enabled=torch_debug_enabled,
            torch_debug_port=torch_debug_port,
            torch_debug_port_search=torch_debug_port_search,
            paired_port_search_attempts=paired_port_search_attempts,
            artifact_dir=None if artifact_dir is None else Path(artifact_dir),
            inspection_strict=inspection_strict,
            pipeline_pressure_enabled=pipeline_pressure_enabled,
            pipeline_pressure_window=pipeline_pressure_window,
            break_on_backward_exception=break_on_backward_exception,
            dataloader_finite_check_enabled=dataloader_finite_check_enabled,
            optimizer_surgery_enabled=optimizer_surgery_enabled,
            hooks_enabled=hooks_enabled,
            hooks_source_dir=None
            if hooks_source_dir is None
            else Path(hooks_source_dir),
            hooks_timeout_s=hooks_timeout_s,
            hooks_rank_policy=hooks_rank_policy,
            hooks_shell=None if hooks_shell is None else Path(hooks_shell),
            hooks_max_params_bytes=hooks_max_params_bytes,
            hooks_max_metadata_items=hooks_max_metadata_items,
            hooks_max_metadata_depth=hooks_max_metadata_depth,
            hooks_max_export_bytes=hooks_max_export_bytes,
            hooks_max_output_bytes=hooks_max_output_bytes,
            intercept_lightning_logging=intercept_lightning_logging,
            run_log_enabled=run_log_enabled,
        )
        self._midrun = build_runtime(config, type(self).__midrun_lightning_backend__)
        self._midrun_hooks_adapter: Any | None = None
        self._midrun_service_guard: Any | None = None
        super().__init__(*args, **kwargs)

    @property
    def midrun(self) -> MidrunRuntime:
        """The `MidrunRuntime` that this model owns."""
        return self._midrun

    def setup(self, stage: str) -> None:
        """Lightning's `setup` hook; starts Midrun's services for this stage."""
        super().setup(stage)
        if self._midrun.config.enabled:
            self._validate_trainer_backend()
            self._midrun.setup_environment(self, stage)

    def on_fit_start(self) -> None:
        """Lightning's `on_fit_start` hook; forwards to `MidrunRuntime`."""
        super().on_fit_start()
        if self._midrun.config.enabled:
            self._midrun.on_fit_start(self)

    def teardown(self, stage: str) -> None:
        """Lightning's `teardown` hook; stops Midrun's services for this stage.

        Lightning calls this after the callbacks' `teardown`, so the logging session
        closes here, after the services stop, and the run log keeps their warnings.
        """
        try:
            if self._midrun.config.enabled:
                self._midrun.teardown(self, stage)
        finally:
            try:
                self._midrun.close_logging_session()
            finally:
                super().teardown(stage)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        """Lightning's `on_train_batch_start` hook; forwards to `MidrunRuntime`."""
        result: int | None = super().on_train_batch_start(batch, batch_idx)
        if self._midrun.config.enabled:
            self._midrun.begin_train_batch(self, batch, batch_idx)
        return result

    def on_before_backward(self, loss: Any) -> None:
        """Lightning's `on_before_backward` hook; forwards to `MidrunRuntime`."""
        super().on_before_backward(loss)
        if self._midrun.config.enabled:
            self._midrun.before_backward(loss)

    def backward(self, loss: Any, *args: Any, **kwargs: Any) -> None:
        """Lightning's `backward` hook; reports an exception here to `MidrunRuntime`."""
        try:
            super().backward(loss, *args, **kwargs)
        except Exception as exc:
            if self._midrun.config.enabled:
                self._midrun.handle_backward_exception(self, loss, exc)
            raise

    def on_after_backward(self) -> None:
        """Lightning's `on_after_backward` hook; forwards to `MidrunRuntime`."""
        super().on_after_backward()
        if self._midrun.config.enabled:
            self._midrun.after_backward()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Lightning's `on_train_batch_end` hook; forwards to `MidrunRuntime`."""
        try:
            super().on_train_batch_end(outputs, batch, batch_idx)
            if self._midrun.config.enabled:
                self._midrun.process_safe_point(
                    self, SafePoint.TRAIN_BATCH_END, batch_idx=batch_idx
                )
        finally:
            if self._midrun.config.enabled:
                self._midrun.finish_train_batch(self)

    def on_train_epoch_end(self) -> None:
        """Lightning's `on_train_epoch_end` hook; forwards to `MidrunRuntime`."""
        super().on_train_epoch_end()
        if self._midrun.config.enabled:
            self._midrun.process_safe_point(self, SafePoint.TRAIN_EPOCH_END)

    def on_validation_epoch_end(self) -> None:
        """Lightning's `on_validation_epoch_end` hook; forwards to `MidrunRuntime`."""
        super().on_validation_epoch_end()
        if self._midrun.config.enabled:
            self._midrun.process_safe_point(self, SafePoint.VALIDATION_EPOCH_END)

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Lightning's `on_before_optimizer_step` hook; forwards to `MidrunRuntime`."""
        super().on_before_optimizer_step(optimizer)
        if self._midrun.config.enabled:
            self._midrun.process_safe_point(self, SafePoint.BEFORE_OPTIMIZER_STEP)

    def on_fit_end(self) -> None:
        """Lightning's `on_fit_end` hook; forwards to `MidrunRuntime`."""
        super().on_fit_end()
        if self._midrun.config.enabled:
            self._midrun.process_safe_point(self, SafePoint.FIT_END)

    def configure_callbacks(self) -> Sequence[Any]:
        """Appends Midrun's lifecycle-hooks adapter and service guard to the model's callbacks.

        The service guard comes last, so the `on_exception` shell hook runs while
        the services are still up.
        """
        # super().configure_callbacks() is provided by the composed LightningModule.
        callbacks = _normalize_callbacks(super().configure_callbacks())
        if not self._midrun.config.enabled:
            return callbacks
        if self._midrun.config.hooks_enabled:
            if self._midrun_hooks_adapter is None:
                self._midrun_hooks_adapter = type(
                    self
                ).__midrun_hooks_callback_class__()
            callbacks.append(self._midrun_hooks_adapter)
        if self._midrun_service_guard is None:
            self._midrun_service_guard = type(self).__midrun_service_guard_class__(
                self._midrun
            )
        callbacks.append(self._midrun_service_guard)
        return callbacks

    def _validate_trainer_backend(self) -> None:
        backend = type(self).__midrun_lightning_backend__
        # self.trainer is provided by the composed LightningModule; backend.Trainer is a
        # real class at runtime, but pylint can't see through the dataclass field's `type`
        # annotation.
        trainer = self.trainer
        # pylint: disable-next=isinstance-second-argument-not-valid-type
        if not isinstance(trainer, backend.Trainer):
            raise RuntimeError(
                f"Midrun Lightning backend mismatch: model uses {backend.name}, but Trainer is "
                f"{type(trainer).__module__}.{type(trainer).__name__}. "
                "Do not mix lightning.pytorch and pytorch_lightning."
            )


def _normalize_callbacks(value: Any) -> list[Any]:
    """Coerces a `configure_callbacks()` return value to a fresh, appendable list."""
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


class _ServiceGuardMethods:
    """Mixed in ahead of a backend's `Callback` base by `MidrunMixin.__init_subclass__`.

    After an uncaught exception (including `KeyboardInterrupt`), Lightning skips
    `LightningModule.teardown` and calls only the callbacks' `on_exception`. This
    callback stops Midrun's services there, also when the hooks adapter is not
    attached, so no REST server, FUSE mount, or debug server outlives a failed run.

    This callback also opens the logging session of each stage `_run`, and closes it
    after a failure, so that the run log and the Lightning logging bridge do not
    depend on `hooks_enabled`. After a normal run, `MidrunMixin.teardown` closes it.

    Attributes:
        runtime: The model's `MidrunRuntime`.
    """

    runtime: MidrunRuntime

    def __init__(self, runtime: MidrunRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    def setup(
        self, trainer: "Trainer", pl_module: "LightningModule", stage: str
    ) -> None:
        """Opens the logging session, unless the hooks adapter's `setup` opened it."""
        del stage
        self.runtime.open_logging_session(trainer, pl_module)

    def on_exception(
        self, trainer: object, pl_module: object, exception: BaseException
    ) -> None:
        """Stops Midrun's services, records the status `failed`, closes the logging session."""
        del trainer, pl_module, exception
        try:
            self.runtime.stop_after_failure()
        finally:
            self.runtime.close_logging_session()
