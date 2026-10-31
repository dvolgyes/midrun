"""Finds the floating-point and complex tensors in a training batch."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchTensorRef:
    """One floating/complex tensor found while traversing a batch.

    Attributes:
        path: Human-readable location within the batch, e.g. `batch[1]["targets"]`.
        tensor: The live tensor. Never copied by the traversal itself.
    """

    path: str
    tensor: torch.Tensor


def iter_batch_tensors(batch: object, path: str = "batch") -> list[BatchTensorRef]:
    """Traverses `batch`, yielding every reachable floating/complex tensor.

    Supports `Tensor`, `Mapping`, `list`, `tuple` (including namedtuples). Integer and
    boolean tensors are skipped -- non-finite values aren't meaningful for them.
    Unknown container types are skipped rather than traversed.
    """
    refs: list[BatchTensorRef] = []
    if isinstance(batch, torch.Tensor):
        if batch.is_floating_point() or batch.is_complex():
            refs.append(BatchTensorRef(path=path, tensor=batch))
        return refs
    if isinstance(batch, Mapping):
        for key, value in batch.items():
            refs.extend(iter_batch_tensors(value, f"{path}[{key!r}]"))
        return refs
    if isinstance(batch, tuple | list):
        for index, value in enumerate(batch):
            refs.extend(iter_batch_tensors(value, f"{path}[{index}]"))
        return refs
    return refs
