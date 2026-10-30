"""Renders an autograd graph with torchviz."""

import shutil
from typing import Any

from midrun.runtime.graph_inspectors.render_graph import render_graph


def torchviz_capture(
    model: Any, output: Any, show_attrs: bool = True, show_saved: bool = True
) -> dict[str, str | bytes]:
    """Renders the autograd graph rooted at `output`.

    `output` must not already have been backwarded with `retain_graph=False` (the
    default) -- its saved tensors would already be freed. Use `torchviz_capture_live`
    for a live diagnostic capture during ordinary training, where the most recent
    `current_step.loss` is always already-backwarded by Lightning itself; this
    function stays useful as-is for the backward-exception breakpoint, where
    `current_step.loss` is the loss whose `backward()` call raised and is exactly
    the graph state worth inspecting.

    Raises:
        ImportError: torchviz or the `dot` binary is unavailable.
    """
    import torchviz

    if shutil.which("dot") is None:
        raise ImportError("torchviz_capture requires the Graphviz 'dot' binary")
    dot = torchviz.make_dot(
        output,
        params=dict(model.named_parameters()),
        show_attrs=show_attrs,
        show_saved=show_saved,
    )
    return render_graph(dot)
