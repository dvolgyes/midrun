"""The `CommandSpec` of each command kind."""

from collections.abc import Mapping
from types import MappingProxyType

from midrun.runtime.command_args.checkpoint_args import CheckpointArgs
from midrun.runtime.command_args.hold_args import HoldArgs
from midrun.runtime.command_args.no_args import NoArgs
from midrun.runtime.command_args.profile_args import ProfileArgs
from midrun.runtime.command_args.reset_momentum_args import ResetMomentumArgs
from midrun.runtime.command_args.set_dataloader_finite_check_args import (
    SetDataloaderFiniteCheckArgs,
)
from midrun.runtime.command_args.set_hparam_args import SetHparamArgs
from midrun.runtime.command_args.set_learning_rate_args import SetLearningRateArgs
from midrun.runtime.command_args.set_lightning_profiler_args import (
    SetLightningProfilerArgs,
)
from midrun.runtime.command_args.snapshot_args import SnapshotArgs
from midrun.runtime.command_args.torchviz_capture_args import TorchvizCaptureArgs
from midrun.runtime.command_specs.command_spec import CommandSpec
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

COMMAND_SPECS: Mapping[str, CommandSpec] = MappingProxyType(
    {
        "hold": CommandSpec(
            args=HoldArgs,
            handler=handle_hold,
            all_ranks=False,
            during_hold=False,
            during_exception=False,
        ),
        "stop": CommandSpec(
            args=NoArgs,
            handler=handle_stop,
            all_ranks=True,
            during_hold=True,
            during_exception=False,
        ),
        "checkpoint": CommandSpec(
            args=CheckpointArgs,
            handler=handle_checkpoint,
            all_ranks=True,
            during_hold=True,
            during_exception=False,
        ),
        "snapshot": CommandSpec(
            args=SnapshotArgs,
            handler=handle_snapshot,
            all_ranks=False,
            during_hold=True,
            during_exception=False,
        ),
        "set_learning_rate": CommandSpec(
            args=SetLearningRateArgs,
            handler=handle_set_learning_rate,
            all_ranks=True,
            during_hold=True,
            during_exception=False,
        ),
        "reset_momentum": CommandSpec(
            args=ResetMomentumArgs,
            handler=handle_reset_momentum,
            all_ranks=True,
            during_hold=True,
            during_exception=False,
        ),
        "set_dataloader_finite_check": CommandSpec(
            args=SetDataloaderFiniteCheckArgs,
            handler=handle_set_dataloader_finite_check,
            all_ranks=True,
            during_hold=True,
            during_exception=False,
        ),
        "set_hparam": CommandSpec(
            args=SetHparamArgs,
            handler=handle_set_hparam,
            all_ranks=True,
            during_hold=True,
            during_exception=False,
        ),
        "save_current_batch": CommandSpec(
            args=NoArgs,
            handler=handle_save_current_batch,
            all_ranks=False,
            during_hold=False,
            during_exception=True,
        ),
        "save_current_loss": CommandSpec(
            args=NoArgs,
            handler=handle_save_current_loss,
            all_ranks=False,
            during_hold=False,
            during_exception=True,
        ),
        "save_partial_gradients": CommandSpec(
            args=NoArgs,
            handler=handle_save_partial_gradients,
            all_ranks=False,
            during_hold=False,
            during_exception=True,
        ),
        "save_optimizer_summary": CommandSpec(
            args=NoArgs,
            handler=handle_save_optimizer_summary,
            all_ranks=False,
            during_hold=False,
            during_exception=True,
        ),
        "release_exception": CommandSpec(
            args=NoArgs,
            handler=handle_release_exception,
            all_ranks=False,
            during_hold=False,
            during_exception=True,
        ),
        "torchinfo_capture": CommandSpec(
            args=NoArgs,
            handler=handle_torchinfo_capture,
            all_ranks=False,
            during_hold=True,
            during_exception=False,
        ),
        "profile": CommandSpec(
            args=ProfileArgs,
            handler=handle_profile,
            all_ranks=False,
            during_hold=True,
            during_exception=False,
        ),
        "torchview_capture": CommandSpec(
            args=NoArgs,
            handler=handle_torchview_capture,
            all_ranks=False,
            during_hold=True,
            during_exception=False,
        ),
        "torchviz_capture": CommandSpec(
            args=TorchvizCaptureArgs,
            handler=handle_torchviz_capture,
            all_ranks=False,
            during_hold=True,
            during_exception=True,
        ),
        "torchlens_capture": CommandSpec(
            args=NoArgs,
            handler=handle_torchlens_capture,
            all_ranks=False,
            during_hold=True,
            during_exception=False,
        ),
        "set_lightning_profiler": CommandSpec(
            args=SetLightningProfilerArgs,
            handler=handle_set_lightning_profiler,
            all_ranks=False,
            during_hold=True,
            during_exception=False,
        ),
    }
)
