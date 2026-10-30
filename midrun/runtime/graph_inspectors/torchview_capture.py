"""Renders the architecture graph of a model with torchview."""

import shutil
from typing import Any

from midrun.runtime.graph_inspectors.render_graph import render_graph


def torchview_capture(model: Any, batch: Any) -> dict[str, str | bytes]:
    """Renders the model's architecture graph traced against `batch`.

    Raises:
        ImportError: torchview or the `dot` binary is unavailable.
    """
    import torchview

    if shutil.which("dot") is None:
        raise ImportError("torchview_capture requires the Graphviz 'dot' binary")
    # torchview defaults to cuda whenever it's available and calls model.to(device)
    # on the *actual* model (nn.Module.to is in-place) -- pin it to the model's real
    # device so a live capture can't silently relocate the model mid-training.
    graph = torchview.draw_graph(
        model, input_data=batch, device=next(model.parameters()).device
    )
    return render_graph(graph.visual_graph)
