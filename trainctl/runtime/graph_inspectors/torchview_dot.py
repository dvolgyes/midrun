"""The DOT source of the architecture graph of a model, from torchview."""

from typing import Any


def torchview_dot(model: Any, batch: Any) -> str:
    """Returns the DOT source for the model's architecture graph traced against `batch`.

    Never renders, so unlike `torchview_capture` this needs only the `torchview`
    package, not the `dot` binary.

    Raises:
        ImportError: torchview is not installed.
    """
    import torchview

    source: str = torchview.draw_graph(
        model, input_data=batch, device=next(model.parameters()).device
    ).visual_graph.source
    return source
