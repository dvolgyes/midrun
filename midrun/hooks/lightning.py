"""Lightning `Callback` adapter: explicit signatures for the 37 callbacks the catalogue tracks.

Deliberately not built by runtime reflection over the installed `Callback` class --
`tests/hooks/lightning_contract_test.py` compares this module's coverage against
both namespaces' installed `Callback` API and fails on drift, but production code
here never discovers a method that way.
"""

from functools import cache
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from midrun.lightning_backend import LightningBackend

if TYPE_CHECKING:
    from lightning.pytorch import Trainer

    from midrun.mixin import MidrunMixin


class _HookMethods:
    """Mixed in ahead of a backend's `Callback` base by `build_hooks_callback_class`.

    Each method forwards only its own named arguments (never `trainer`/`pl_module`)
    to `MidrunRuntime.dispatch_hook`, which re-derives stage/epoch/step/rank
    identity fresh from `trainer` for every call. `setup`/`teardown` additionally
    own the hooks session's lifetime: Lightning always calls `setup` before any
    other callback method for a stage `_run`, and always pairs it with a matching
    `teardown` afterward (barring an uncaught exception, handled by `on_exception`
    below) -- so opening in `setup` and closing in `teardown` cannot leak or
    outlive the `_run` it belongs to, even for a stage entered without `fit`/
    `validate`/`test`/`predict` ever completing normally.

    A session that cannot open (no bash, a foreign `hooks/` tree, a bad marker, a
    copy error) raises under `inspection_strict`; otherwise it logs a warning, and
    the stage trains with no session, so every dispatch is a no-op.
    """

    def setup(self, trainer: "Trainer", pl_module: "MidrunMixin", stage: str) -> None:
        """Opens the stage's hooks session before dispatching setup."""
        if not self._owns(pl_module):
            return
        runtime = pl_module.midrun
        try:
            runtime.ensure_hook_session(trainer, pl_module)
        except (RuntimeError, OSError) as exc:
            if runtime.config.inspection_strict:
                raise
            logger.warning(
                "midrun: hooks are off for this stage: the hooks session failed: {}",
                exc,
            )
        self._dispatch(trainer, pl_module, "setup", stage_override=stage)

    def teardown(
        self, trainer: "Trainer", pl_module: "MidrunMixin", stage: str
    ) -> None:
        """Dispatches teardown before closing the stage's hooks session."""
        if not self._owns(pl_module):
            return
        try:
            self._dispatch(trainer, pl_module, "teardown", stage_override=stage)
        finally:
            pl_module.midrun.close_hook_session()

    def on_fit_start(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the start of fitting."""
        self._dispatch(trainer, pl_module, "on_fit_start")

    def on_fit_end(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the end of fitting."""
        self._dispatch(trainer, pl_module, "on_fit_end")

    def on_sanity_check_start(
        self, trainer: "Trainer", pl_module: "MidrunMixin"
    ) -> None:
        """Dispatches the start of Lightning's sanity check."""
        self._dispatch(trainer, pl_module, "on_sanity_check_start")

    def on_sanity_check_end(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the end of Lightning's sanity check."""
        self._dispatch(trainer, pl_module, "on_sanity_check_end")

    def on_train_start(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the start of training."""
        self._dispatch(trainer, pl_module, "on_train_start")

    def on_train_end(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the end of training."""
        self._dispatch(trainer, pl_module, "on_train_end")

    def on_train_epoch_start(
        self, trainer: "Trainer", pl_module: "MidrunMixin"
    ) -> None:
        """Dispatches the start of a training epoch."""
        self._dispatch(trainer, pl_module, "on_train_epoch_start")

    def on_train_epoch_end(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the end of a training epoch."""
        self._dispatch(trainer, pl_module, "on_train_epoch_end")

    def on_train_batch_start(
        self,
        trainer: "Trainer",
        pl_module: "MidrunMixin",
        batch: object,
        batch_idx: int,
    ) -> None:
        """Dispatches the incoming training batch."""
        self._dispatch(
            trainer,
            pl_module,
            "on_train_batch_start",
            arguments={"batch": batch},
            batch_idx=batch_idx,
        )

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "MidrunMixin",
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Dispatches the training batch and its step outputs."""
        self._dispatch(
            trainer,
            pl_module,
            "on_train_batch_end",
            arguments={"outputs": outputs, "batch": batch},
            batch_idx=batch_idx,
        )

    def on_validation_start(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the start of validation."""
        self._dispatch(trainer, pl_module, "on_validation_start")

    def on_validation_end(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the end of validation."""
        self._dispatch(trainer, pl_module, "on_validation_end")

    def on_validation_epoch_start(
        self, trainer: "Trainer", pl_module: "MidrunMixin"
    ) -> None:
        """Dispatches the start of a validation epoch."""
        self._dispatch(trainer, pl_module, "on_validation_epoch_start")

    def on_validation_epoch_end(
        self, trainer: "Trainer", pl_module: "MidrunMixin"
    ) -> None:
        """Dispatches the end of a validation epoch."""
        self._dispatch(trainer, pl_module, "on_validation_epoch_end")

    def on_validation_batch_start(
        self,
        trainer: "Trainer",
        pl_module: "MidrunMixin",
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Dispatches the incoming validation batch with its loader index."""
        self._dispatch(
            trainer,
            pl_module,
            "on_validation_batch_start",
            arguments={"batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "MidrunMixin",
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Dispatches the validation batch and its step outputs."""
        self._dispatch(
            trainer,
            pl_module,
            "on_validation_batch_end",
            arguments={"outputs": outputs, "batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_test_start(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the start of testing."""
        self._dispatch(trainer, pl_module, "on_test_start")

    def on_test_end(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the end of testing."""
        self._dispatch(trainer, pl_module, "on_test_end")

    def on_test_epoch_start(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the start of a test epoch."""
        self._dispatch(trainer, pl_module, "on_test_epoch_start")

    def on_test_epoch_end(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the end of a test epoch."""
        self._dispatch(trainer, pl_module, "on_test_epoch_end")

    def on_test_batch_start(
        self,
        trainer: "Trainer",
        pl_module: "MidrunMixin",
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Dispatches the incoming test batch with its loader index."""
        self._dispatch(
            trainer,
            pl_module,
            "on_test_batch_start",
            arguments={"batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_test_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "MidrunMixin",
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Dispatches the test batch and its step outputs."""
        self._dispatch(
            trainer,
            pl_module,
            "on_test_batch_end",
            arguments={"outputs": outputs, "batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_predict_start(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the start of prediction."""
        self._dispatch(trainer, pl_module, "on_predict_start")

    def on_predict_end(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches the end of prediction."""
        self._dispatch(trainer, pl_module, "on_predict_end")

    def on_predict_epoch_start(
        self, trainer: "Trainer", pl_module: "MidrunMixin"
    ) -> None:
        """Dispatches the start of a prediction epoch."""
        self._dispatch(trainer, pl_module, "on_predict_epoch_start")

    def on_predict_epoch_end(
        self, trainer: "Trainer", pl_module: "MidrunMixin"
    ) -> None:
        """Dispatches the end of a prediction epoch."""
        self._dispatch(trainer, pl_module, "on_predict_epoch_end")

    def on_predict_batch_start(
        self,
        trainer: "Trainer",
        pl_module: "MidrunMixin",
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Dispatches the incoming prediction batch with its loader index."""
        self._dispatch(
            trainer,
            pl_module,
            "on_predict_batch_start",
            arguments={"batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_predict_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "MidrunMixin",
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Dispatches the prediction batch and its outputs."""
        self._dispatch(
            trainer,
            pl_module,
            "on_predict_batch_end",
            arguments={"outputs": outputs, "batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_before_backward(
        self, trainer: "Trainer", pl_module: "MidrunMixin", loss: torch.Tensor
    ) -> None:
        """Dispatches the loss before backward computation."""
        self._dispatch(
            trainer, pl_module, "on_before_backward", arguments={"loss": loss}
        )

    def on_after_backward(self, trainer: "Trainer", pl_module: "MidrunMixin") -> None:
        """Dispatches after backward computation."""
        self._dispatch(trainer, pl_module, "on_after_backward")

    def on_before_optimizer_step(
        self,
        trainer: "Trainer",
        pl_module: "MidrunMixin",
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Dispatches the optimizer before its step."""
        self._dispatch(
            trainer,
            pl_module,
            "on_before_optimizer_step",
            arguments={"optimizer": optimizer},
        )

    def on_before_zero_grad(
        self,
        trainer: "Trainer",
        pl_module: "MidrunMixin",
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Dispatches the optimizer before its gradients are cleared."""
        self._dispatch(
            trainer,
            pl_module,
            "on_before_zero_grad",
            arguments={"optimizer": optimizer},
        )

    def on_save_checkpoint(
        self, trainer: "Trainer", pl_module: "MidrunMixin", checkpoint: dict[str, Any]
    ) -> None:
        """Dispatches the checkpoint being saved."""
        self._dispatch(
            trainer,
            pl_module,
            "on_save_checkpoint",
            arguments={"checkpoint": checkpoint},
        )

    def on_load_checkpoint(
        self, trainer: "Trainer", pl_module: "MidrunMixin", checkpoint: dict[str, Any]
    ) -> None:
        """Dispatches the checkpoint being loaded."""
        self._dispatch(
            trainer,
            pl_module,
            "on_load_checkpoint",
            arguments={"checkpoint": checkpoint},
        )

    def on_exception(
        self, trainer: "Trainer", pl_module: "MidrunMixin", exception: BaseException
    ) -> None:
        """Dispatches an exception and closes the active hooks session."""
        if not self._owns(pl_module):
            return
        try:
            self._dispatch(
                trainer,
                pl_module,
                "on_exception",
                arguments={"exception": exception},
            )
        finally:
            pl_module.midrun.close_hook_session()

    def _owns(self, pl_module: object) -> bool:
        return getattr(pl_module, "_midrun_hooks_adapter", None) is self

    def _dispatch(
        self,
        trainer: "Trainer",
        pl_module: "MidrunMixin",
        hook: str,
        *,
        stage_override: str | None = None,
        arguments: dict[str, Any] | None = None,
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
    ) -> None:
        if not self._owns(pl_module):
            return
        runtime = pl_module.midrun
        runtime.dispatch_hook(
            trainer,
            hook,
            stage_override=stage_override,
            arguments=arguments,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )


@cache
def build_hooks_callback_class(backend: LightningBackend) -> type:
    """Builds this backend's concrete Midrun hooks `Callback` adapter class.

    Lightning replaces callback instances by type when the Trainer changes models.
    """
    return type("MidrunHooksCallback", (_HookMethods, backend.callback_class), {})
