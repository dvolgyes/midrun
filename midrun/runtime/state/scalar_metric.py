"""The latest value of one logged scalar metric."""

from dataclasses import dataclass

from midrun.runtime.state.host_scalar import HostScalar


@dataclass(frozen=True)
class ScalarMetric:
    """A single scalar metric value pushed from a Lightning hook.

    Attributes:
        value: The metric value: a float, or a `HostScalar` for a CUDA value
            that a reader waits for (see `host_snapshot`).
        epoch: The epoch the value was produced at.
        step: The global step the value was produced at.
        updated_at: Unix timestamp the value was published.
    """

    value: float | HostScalar
    epoch: int
    step: int
    updated_at: float
