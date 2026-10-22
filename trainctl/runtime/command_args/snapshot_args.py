"""The argument model of the snapshot command."""

from typing import Literal

from trainctl.runtime.command_args.command_args_model import CommandArgsModel


class SnapshotArgs(CommandArgsModel):
    """Args of `snapshot`."""

    kind: Literal[
        "runtime", "model_summary", "optimizer_summary", "scheduler_summary"
    ] = "runtime"
