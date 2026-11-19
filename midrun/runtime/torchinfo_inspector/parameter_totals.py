"""Inventory the unique parameters of a model for torchinfo's totals."""

from collections.abc import Iterable
from typing import NamedTuple

from torch import Tensor
from torch.nn.parameter import is_lazy


class ParameterTotals(NamedTuple):
    """Size of the known part of a model.

    :ivar total: Number of initialized parameter elements.
    :ivar trainable: Elements that require gradients.
    :ivar total_bytes: Storage size of the counted elements.
    :ivar partial: At least one parameter has no shape yet.
    """

    total: int
    trainable: int
    total_bytes: int
    partial: bool


def parameter_totals(parameters: Iterable[Tensor]) -> ParameterTotals:
    """Count initialized elements from a deduplicated parameter iterator."""
    total = trainable = total_bytes = 0
    partial = False
    for parameter in parameters:
        if is_lazy(parameter):
            partial = True
            continue
        count = parameter.numel()
        total += count
        total_bytes += count * parameter.element_size()
        if parameter.requires_grad:
            trainable += count
    return ParameterTotals(total, trainable, total_bytes, partial)
