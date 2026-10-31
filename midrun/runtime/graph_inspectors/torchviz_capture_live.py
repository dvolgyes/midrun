"""Renders the autograd graph of a fresh forward pass with torchviz."""

import shutil
from collections.abc import Mapping

import torch

from midrun.runtime.graph_inspectors.torchviz_capture import torchviz_capture


def torchviz_capture_live(
    model: torch.nn.Module,
    forward_input: object,
    show_attrs: bool = True,
    show_saved: bool = True,
) -> dict[str, str | bytes]:
    """Runs a fresh forward pass and renders its (still-intact) autograd graph.

    Unlike calling `torchviz_capture` on an already-backwarded loss -- which fails
    with "Trying to backward through the graph a second time" once Lightning's own
    backward (`retain_graph=False`) has freed its saved tensors -- this computes a
    new output whose graph has never been backwarded. The caller sets the mode:
    `inspector_forward(..., needs_grad=True)` gives eval mode with grad on, so the
    pass does not perturb BatchNorm running stats mid-training.

    A list or tuple `forward_input` is passed as positional arguments and a mapping
    as keyword arguments, as torchinfo and torchview do.

    Raises:
        ImportError: torchviz or the `dot` binary is unavailable.
    """
    if shutil.which("dot") is None:
        raise ImportError("torchviz_capture requires the Graphviz 'dot' binary")
    if isinstance(forward_input, list | tuple):
        output = model(*forward_input)
    elif isinstance(forward_input, Mapping):
        output = model(**forward_input)
    else:
        output = model(forward_input)
    return torchviz_capture(model, output, show_attrs, show_saved)
