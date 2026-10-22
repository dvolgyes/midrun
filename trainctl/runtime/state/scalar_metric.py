"""The latest value of one logged scalar metric."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarMetric:
    """A single scalar metric value pushed from a Lightning hook.

    Attributes:
        value: The metric value.
        epoch: The epoch the value was produced at.
        step: The global step the value was produced at.
        updated_at: Unix timestamp the value was published.
    """

    value: float
    epoch: int
    step: int
    updated_at: float
