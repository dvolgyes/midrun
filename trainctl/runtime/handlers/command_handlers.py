"""Maps each command kind to its handler."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trainctl.runtime.commands.command import Command
from trainctl.runtime.events import SafePoint
from trainctl.runtime.handlers.handle_checkpoint import handle_checkpoint
from trainctl.runtime.handlers.handle_hold import handle_hold
from trainctl.runtime.handlers.handle_profile import handle_profile
from trainctl.runtime.handlers.handle_release_exception import handle_release_exception
from trainctl.runtime.handlers.handle_reset_momentum import handle_reset_momentum
from trainctl.runtime.handlers.handle_save_current_batch import (
    handle_save_current_batch,
)
from trainctl.runtime.handlers.handle_save_current_loss import handle_save_current_loss
from trainctl.runtime.handlers.handle_save_optimizer_summary import (
    handle_save_optimizer_summary,
)
from trainctl.runtime.handlers.handle_save_partial_gradients import (
    handle_save_partial_gradients,
)
from trainctl.runtime.handlers.handle_set_dataloader_finite_check import (
    handle_set_dataloader_finite_check,
)
from trainctl.runtime.handlers.handle_set_hparam import handle_set_hparam
from trainctl.runtime.handlers.handle_set_learning_rate import handle_set_learning_rate
from trainctl.runtime.handlers.handle_set_lightning_profiler import (
    handle_set_lightning_profiler,
)
from trainctl.runtime.handlers.handle_snapshot import handle_snapshot
from trainctl.runtime.handlers.handle_stop import handle_stop
from trainctl.runtime.handlers.handle_torchinfo_capture import handle_torchinfo_capture
from trainctl.runtime.handlers.handle_torchlens_capture import handle_torchlens_capture
from trainctl.runtime.handlers.handle_torchview_capture import handle_torchview_capture
from trainctl.runtime.handlers.handle_torchviz_capture import handle_torchviz_capture

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


COMMAND_HANDLERS: dict[
    str,
    Callable[["TrainctlRuntime", Any, Any, Command, SafePoint | None], dict[str, Any]],
] = {
    "hold": handle_hold,
    "stop": handle_stop,
    "checkpoint": handle_checkpoint,
    "snapshot": handle_snapshot,
    "set_learning_rate": handle_set_learning_rate,
    "reset_momentum": handle_reset_momentum,
    "set_dataloader_finite_check": handle_set_dataloader_finite_check,
    "set_hparam": handle_set_hparam,
    "save_current_batch": handle_save_current_batch,
    "save_current_loss": handle_save_current_loss,
    "save_partial_gradients": handle_save_partial_gradients,
    "save_optimizer_summary": handle_save_optimizer_summary,
    "release_exception": handle_release_exception,
    "torchinfo_capture": handle_torchinfo_capture,
    "profile": handle_profile,
    "torchview_capture": handle_torchview_capture,
    "torchviz_capture": handle_torchviz_capture,
    "torchlens_capture": handle_torchlens_capture,
    "set_lightning_profiler": handle_set_lightning_profiler,
}
