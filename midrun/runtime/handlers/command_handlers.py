"""Maps each command kind to its handler."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from midrun.runtime.commands.command import Command
from midrun.runtime.events import SafePoint
from midrun.runtime.handlers.handle_checkpoint import handle_checkpoint
from midrun.runtime.handlers.handle_hold import handle_hold
from midrun.runtime.handlers.handle_profile import handle_profile
from midrun.runtime.handlers.handle_release_exception import handle_release_exception
from midrun.runtime.handlers.handle_reset_momentum import handle_reset_momentum
from midrun.runtime.handlers.handle_save_current_batch import (
    handle_save_current_batch,
)
from midrun.runtime.handlers.handle_save_current_loss import handle_save_current_loss
from midrun.runtime.handlers.handle_save_optimizer_summary import (
    handle_save_optimizer_summary,
)
from midrun.runtime.handlers.handle_save_partial_gradients import (
    handle_save_partial_gradients,
)
from midrun.runtime.handlers.handle_set_dataloader_finite_check import (
    handle_set_dataloader_finite_check,
)
from midrun.runtime.handlers.handle_set_hparam import handle_set_hparam
from midrun.runtime.handlers.handle_set_learning_rate import handle_set_learning_rate
from midrun.runtime.handlers.handle_set_lightning_profiler import (
    handle_set_lightning_profiler,
)
from midrun.runtime.handlers.handle_snapshot import handle_snapshot
from midrun.runtime.handlers.handle_stop import handle_stop
from midrun.runtime.handlers.handle_torchinfo_capture import handle_torchinfo_capture
from midrun.runtime.handlers.handle_torchlens_capture import handle_torchlens_capture
from midrun.runtime.handlers.handle_torchview_capture import handle_torchview_capture
from midrun.runtime.handlers.handle_torchviz_capture import handle_torchviz_capture

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


COMMAND_HANDLERS: dict[
    str,
    Callable[["MidrunRuntime", Any, Any, Command, SafePoint | None], dict[str, Any]],
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
