"""The latest value of one logged scalar metric."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ScalarMetric:
    """A single scalar metric value pushed from a Lightning hook.

    Attributes:
        value: The metric value: a float, or a detached one-element tensor that
            stays on its device until a reader copies it (see `host_snapshot`).
        epoch: The epoch the value was produced at.
        step: The global step the value was produced at.
        updated_at: Unix timestamp the value was published.
    """

    value: float | Tensor
    epoch: int
    step: int
    updated_at: float
