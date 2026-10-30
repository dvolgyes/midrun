"""Copies the tensor values of a runtime snapshot to the host for a reader."""

from dataclasses import replace

from torch import Tensor

from midrun.runtime.state.runtime_snapshot import RuntimeSnapshot


def host_snapshot(snapshot: RuntimeSnapshot) -> RuntimeSnapshot:
    """Returns `snapshot` with each metric value and learning rate as a plain float.

    A safe point publishes tensor values without a copy to the host, so the
    training loop does not wait for the device. A reader calls this before it
    serializes the snapshot; the copy of a device tensor waits only the reader.

    Args:
        snapshot: The published snapshot.

    Returns:
        A snapshot whose `metrics` values and `learning_rates` are floats.
    """
    metrics = {
        key: replace(metric, value=float(metric.value))
        if isinstance(metric.value, Tensor)
        else metric
        for key, metric in snapshot.metrics.items()
    }
    learning_rates = tuple(float(lr) for lr in snapshot.learning_rates)
    return replace(snapshot, metrics=metrics, learning_rates=learning_rates)
