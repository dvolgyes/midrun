"""One published CUDA scalar, read from its pinned host copy."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HostScalar:
    """A CUDA scalar that a safe point copied to pinned host memory without a wait.

    The copy holds no reference to a device tensor, so the published state holds no
    device memory and no autograd graph. `float()` waits for `ready`: only a reader
    calls it, so only the reader waits for the device, never the training loop.

    Attributes:
        host: The pinned host copy of the values that one safe point stacked.
        index: The position of this value in `host`.
        ready: The event recorded after the copy on the device's stream.
    """

    host: Tensor
    index: int
    ready: torch.cuda.Event

    def __float__(self) -> float:
        """Waits for the copy and returns the value."""
        self.ready.synchronize()
        return float(self.host[self.index])
