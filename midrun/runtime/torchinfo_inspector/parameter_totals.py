"""Inventory the unique parameters of a model for torchinfo's totals."""

from collections.abc import Iterable

from torch import Tensor
from torch.nn.parameter import is_lazy


def parameter_totals(parameters: Iterable[Tensor]) -> tuple[int, int, int]:
    """Return count, trainable count, and bytes for initialized model parameters."""
    total = trainable = total_bytes = 0
    for parameter in parameters:
        if is_lazy(parameter):
            continue
        count = parameter.numel()
        total += count
        total_bytes += count * parameter.element_size()
        if parameter.requires_grad:
            trainable += count
    return total, trainable, total_bytes
