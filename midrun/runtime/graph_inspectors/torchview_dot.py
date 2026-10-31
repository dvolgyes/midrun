"""The DOT source of the architecture graph of a model, from torchview."""

from collections.abc import Mapping, Sequence

import torch

from midrun.runtime.graph_inspectors.torchview_device import torchview_device


def torchview_dot(
    model: torch.nn.Module,
    batch: torch.Tensor | Sequence[object] | Mapping[str, object],
) -> str:
    """Returns the DOT source for the model's architecture graph traced against `batch`.

    Never renders, so unlike `torchview_capture` this needs only the `torchview`
    package, not the `dot` binary.

    Raises:
        ImportError: torchview is not installed.
    """
    import torchview

    source: str = torchview.draw_graph(
        model, input_data=batch, device=torchview_device(model)
    ).visual_graph.source
    return source
