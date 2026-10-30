"""Behavioral tests for GET /inspectors."""

from fastapi.testclient import TestClient

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.runtime import MidrunRuntime


def _client() -> TestClient:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    return TestClient(build_app(runtime), base_url="http://localhost")


def test_inspectors_reports_availability_for_all_four() -> None:
    response = _client().get("/inspectors")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"torchinfo", "torchview", "torchviz", "torchlens"}
    for entry in body.values():
        assert isinstance(entry["available"], bool)
