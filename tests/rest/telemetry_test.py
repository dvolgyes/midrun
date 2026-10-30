"""Behavioral tests for GET /telemetry/pipeline."""

from fastapi.testclient import TestClient

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.runtime import MidrunRuntime


def _client() -> tuple[TestClient, MidrunRuntime]:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    return TestClient(build_app(runtime)), runtime


def test_telemetry_pipeline_reports_none_before_first_batch() -> None:
    client, _runtime = _client()
    response = client.get("/telemetry/pipeline")
    assert response.status_code == 200
    body = response.json()
    assert body["latest"] is None
    assert body["summary"]["samples"] == 0


def test_telemetry_pipeline_reports_latest_and_summary_after_a_batch() -> None:
    client, runtime = _client()
    runtime.pipeline.batch_start(0, epoch=0, global_step=0, batch_idx=0)
    runtime.pipeline.batch_end(1_000)

    response = client.get("/telemetry/pipeline")

    assert response.status_code == 200
    body = response.json()
    assert body["latest"]["batch_total_ns"] == 1_000
    assert body["summary"]["samples"] == 1
