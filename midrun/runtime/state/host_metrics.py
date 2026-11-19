"""Reads the host copies of a snapshot's metric values."""

from collections.abc import Mapping
from dataclasses import replace

from midrun.runtime.state.host_scalar import HostScalar
from midrun.runtime.state.scalar_metric import ScalarMetric


def host_metrics(metrics: Mapping[str, ScalarMetric]) -> dict[str, ScalarMetric]:
    """Returns the requested metrics with CUDA scalar values resolved on the reader."""
    return {
        key: replace(metric, value=float(metric.value))
        if isinstance(metric.value, HostScalar)
        else metric
        for key, metric in metrics.items()
    }
