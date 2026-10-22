"""The argument model of the torchviz_capture command."""

from trainctl.runtime.command_args.command_args_model import CommandArgsModel


class TorchvizCaptureArgs(CommandArgsModel):
    """Args of `torchviz_capture`."""

    # None: on in a live capture, off in an exception hold.
    show_attrs: bool | None = None
    show_saved: bool | None = None
