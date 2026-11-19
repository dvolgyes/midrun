"""Reads the host copies of a snapshot's learning rates."""

from midrun.runtime.state.host_scalar import HostScalar


def host_learning_rates(
    learning_rates: tuple[float | HostScalar, ...],
) -> tuple[float, ...]:
    """Returns the requested learning rates as plain floats."""
    return tuple(float(lr) for lr in learning_rates)
