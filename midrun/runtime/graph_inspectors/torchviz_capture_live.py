"""Renders the autograd graph of a fresh forward pass with torchviz."""

import shutil
from typing import Any

import torch

from midrun.runtime.graph_inspectors.torchviz_capture import torchviz_capture


def torchviz_capture_live(
    model: Any, forward_input: Any, show_attrs: bool = True, show_saved: bool = True
) -> dict[str, str | bytes]:
    """Runs a fresh forward pass and renders its (still-intact) autograd graph.

    Unlike calling `torchviz_capture` on an already-backwarded loss -- which fails
    with "Trying to backward through the graph a second time" once Lightning's own
    backward (`retain_graph=False`) has freed its saved tensors -- this computes a
    new output whose graph has never been backwarded. Runs in eval mode (restored
    afterward) so it doesn't perturb BatchNorm running stats mid-training, matching
    `torchinfo_rich_summary`'s own guarantee.

    Raises:
        ImportError: torchviz or the `dot` binary is unavailable.
    """
    if shutil.which("dot") is None:
        raise ImportError("torchviz_capture requires the Graphviz 'dot' binary")
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            output = model(forward_input)
    finally:
        model.train(was_training)
    return torchviz_capture(model, output, show_attrs, show_saved)
