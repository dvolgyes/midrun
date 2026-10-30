"""Opt-in check of every floating/complex tensor in a training batch for NaN/Inf.

Answers "did the batch delivered to the model already contain non-finite values?" --
not a replacement for PyTorch/Lightning autograd anomaly detection, which diagnoses a
failing *backward*. This checks *inputs*, before the training step runs, so a bad
batch can be handled with the normal cooperative hold rather than the exception-hold
path in `midrun.runtime.runtime.handle_backward_exception`.

Disabled by default: checking every batch can force a host/device synchronization and
is not free, even in the simple per-tensor form implemented here.
"""

from dataclasses import dataclass
from typing import Any

import torch

from midrun.runtime.iter_batch_tensors import iter_batch_tensors


@dataclass(frozen=True)
class BadTensorInfo:
    """A tensor that failed the finite check.

    Attributes:
        path: Same location convention as `BatchTensorRef.path`.
        shape: Tensor shape.
        dtype: Tensor dtype, as a string.
        device: Tensor device, as a string.
        nan_count: Number of NaN elements.
        posinf_count: Number of `+inf` elements.
        neginf_count: Number of `-inf` elements.
    """

    path: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    nan_count: int
    posinf_count: int
    neginf_count: int


@dataclass(frozen=True)
class FiniteCheckReport:
    """The result of a failed `DataloaderFiniteGuard.check`.

    Attributes:
        bad_tensors: Every tensor in the batch that failed the finite check.
    """

    bad_tensors: tuple[BadTensorInfo, ...]


class DataloaderFiniteGuard:
    """Checks training batches for non-finite tensors, when enabled.

    Attributes:
        enabled: Whether `check` actually traverses batches. Mutated only from the
            training thread, via the `set_dataloader_finite_check` command.
        checked_batches: Number of batches checked since this guard was created.
        failures: Number of batches that failed the check.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.checked_batches = 0
        self.failures = 0

    def check(self, batch: Any) -> FiniteCheckReport | None:
        """Returns a `FiniteCheckReport` if `batch` contains non-finite values, else `None`.

        No-ops (returns `None` without traversing `batch`) when `enabled` is False.
        """
        if not self.enabled:
            return None
        self.checked_batches += 1
        bad: list[BadTensorInfo] = []
        for ref in iter_batch_tensors(batch):
            if bool(torch.isfinite(ref.tensor).all()):
                continue
            # isposinf/isneginf reject complex input, so a complex element counts
            # when either of its components is infinite.
            if ref.tensor.is_complex():
                components = torch.view_as_real(ref.tensor)
                posinf = torch.isposinf(components).any(dim=-1)
                neginf = torch.isneginf(components).any(dim=-1)
            else:
                posinf = torch.isposinf(ref.tensor)
                neginf = torch.isneginf(ref.tensor)
            bad.append(
                BadTensorInfo(
                    path=ref.path,
                    shape=tuple(ref.tensor.shape),
                    dtype=str(ref.tensor.dtype),
                    device=str(ref.tensor.device),
                    nan_count=int(torch.isnan(ref.tensor).sum()),
                    posinf_count=int(posinf.sum()),
                    neginf_count=int(neginf.sum()),
                )
            )
        if not bad:
            return None
        self.failures += 1
        return FiniteCheckReport(bad_tensors=tuple(bad))
