"""Renders the architecture graph of a model with torchview."""

import shutil
from collections.abc import Mapping, Sequence

import torch

from midrun.runtime.graph_inspectors.render_graph import render_graph
from midrun.runtime.graph_inspectors.torchview_device import torchview_device


def torchview_capture(
    model: torch.nn.Module,
    batch: torch.Tensor | Sequence[object] | Mapping[str, object],
) -> dict[str, str | bytes]:
    """Renders the model's architecture graph traced against `batch`.

    Raises:
        ImportError: torchview or the `dot` binary is unavailable.
    """
    import torchview

    if shutil.which("dot") is None:
        raise ImportError("torchview_capture requires the Graphviz 'dot' binary")
    graph = torchview.draw_graph(
        model, input_data=batch, device=torchview_device(model)
    )
    return render_graph(graph.visual_graph)
