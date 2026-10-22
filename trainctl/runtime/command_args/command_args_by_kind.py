"""The argument model of each command kind."""

from collections.abc import Mapping
from types import MappingProxyType

from trainctl.runtime.command_args.checkpoint_args import CheckpointArgs
from trainctl.runtime.command_args.command_args_model import CommandArgsModel
from trainctl.runtime.command_args.hold_args import HoldArgs
from trainctl.runtime.command_args.no_args import NoArgs
from trainctl.runtime.command_args.profile_args import ProfileArgs
from trainctl.runtime.command_args.reset_momentum_args import ResetMomentumArgs
from trainctl.runtime.command_args.set_dataloader_finite_check_args import (
    SetDataloaderFiniteCheckArgs,
)
from trainctl.runtime.command_args.set_hparam_args import SetHparamArgs
from trainctl.runtime.command_args.set_learning_rate_args import SetLearningRateArgs
from trainctl.runtime.command_args.set_lightning_profiler_args import (
    SetLightningProfilerArgs,
)
from trainctl.runtime.command_args.snapshot_args import SnapshotArgs
from trainctl.runtime.command_args.torchviz_capture_args import TorchvizCaptureArgs

COMMAND_ARGS: Mapping[str, type[CommandArgsModel]] = MappingProxyType(
    {
        "hold": HoldArgs,
        "stop": NoArgs,
        "checkpoint": CheckpointArgs,
        "snapshot": SnapshotArgs,
        "set_learning_rate": SetLearningRateArgs,
        "reset_momentum": ResetMomentumArgs,
        "set_dataloader_finite_check": SetDataloaderFiniteCheckArgs,
        "set_hparam": SetHparamArgs,
        "save_current_batch": NoArgs,
        "save_current_loss": NoArgs,
        "save_partial_gradients": NoArgs,
        "save_optimizer_summary": NoArgs,
        "release_exception": NoArgs,
        "torchinfo_capture": NoArgs,
        "profile": ProfileArgs,
        "torchview_capture": NoArgs,
        "torchviz_capture": TorchvizCaptureArgs,
        "torchlens_capture": NoArgs,
        "set_lightning_profiler": SetLightningProfilerArgs,
    }
)
