"""Reads the host copies of a runtime snapshot's values for a reader."""

from dataclasses import replace

from midrun.runtime.state.host_learning_rates import host_learning_rates
from midrun.runtime.state.host_metrics import host_metrics
from midrun.runtime.state.runtime_snapshot import RuntimeSnapshot


def host_snapshot(snapshot: RuntimeSnapshot) -> RuntimeSnapshot:
    """Returns `snapshot` with each metric value and learning rate as a plain float.

    A safe point publishes a CUDA value as a `HostScalar`, so the training loop
    does not wait for the device. A reader calls this before it serializes the
    snapshot; the wait for the host copy blocks only the reader.

    Args:
        snapshot: The published snapshot.

    Returns:
        A snapshot whose `metrics` values and `learning_rates` are floats.
    """
    return replace(
        snapshot,
        metrics=host_metrics(snapshot.metrics),
        learning_rates=host_learning_rates(snapshot.learning_rates),
    )
