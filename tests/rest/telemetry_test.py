"""Behavioral tests for the telemetry readers."""

from typing import cast
from unittest.mock import Mock

import torch
from fastapi.testclient import TestClient

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.state.host_scalar import HostScalar
from midrun.runtime.state.scalar_metric import ScalarMetric


def _client() -> tuple[TestClient, MidrunRuntime]:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    return TestClient(build_app(runtime), base_url="http://localhost"), runtime


def test_telemetry_pipeline_reports_none_before_first_batch() -> None:
    client, _runtime = _client()
    response = client.get("/telemetry/pipeline")
    assert response.status_code == 200
    body = response.json()
    assert body["latest"] is None
    assert body["summary"]["samples"] == 0


def test_telemetry_pipeline_reports_latest_and_summary_after_a_batch() -> None:
    client, runtime = _client()
    runtime.pipeline.batch_start(0, epoch=0, batch_idx=0)
    runtime.pipeline.batch_end(1_000, global_step=1)

    response = client.get("/telemetry/pipeline")

    assert response.status_code == 200
    body = response.json()
    assert body["latest"]["batch_total_ns"] == 1_000
    assert body["summary"]["samples"] == 1


def test_partial_snapshot_readers_wait_only_for_returned_scalars() -> None:
    client, runtime = _client()
    metric_event = Mock()
    rate_event = Mock()
    host = torch.arange(128, dtype=torch.float32)
    runtime.state.update(
        metrics={
            str(i): ScalarMetric(
                HostScalar(host, i, cast(torch.cuda.Event, metric_event)), 0, 0, 0.0
            )
            for i in range(128)
        },
        learning_rates=(0.25,),
    )
    assert client.get("/optimizers").json() == {"learning_rates": [0.25]}
    metric_event.synchronize.assert_not_called()

    runtime.state.update(
        learning_rates=(HostScalar(host, 1, cast(torch.cuda.Event, rate_event)),)
    )
    metrics = client.get("/metrics").json()
    assert {key: data["value"] for key, data in metrics.items()} == {
        str(i): float(i) for i in range(128)
    }
    rate_event.synchronize.assert_not_called()
