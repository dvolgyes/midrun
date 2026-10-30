"""Behavioral tests for the artifact routes when the store or one of its directories is missing.

A guarded start failure of the artifact directory leaves `runtime.artifacts` as
`None` while REST still runs, and an operator can delete a store directory while
the run continues.
"""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.runtime import MidrunRuntime


def _runtime() -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    return build_runtime(config, load_backend("lightning.pytorch"))


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/artifacts", None),
        ("GET", "/artifacts/snap_000001", None),
        ("POST", "/debug/captures", {"kind": "x", "files": {"a.txt": "a"}}),
        ("GET", "/debug/cuda-memory/snapshot", None),
    ],
)
def test_artifact_routes_answer_503_without_a_store(
    method: str, path: str, body: dict[str, object] | None
) -> None:
    client = TestClient(build_app(_runtime()), base_url="http://localhost")

    response = client.request(method, path, json=body)

    assert response.status_code == 503
    assert "artifact store" in response.json()["detail"]


@pytest.mark.parametrize("subdir", ["checkpoints", "snapshots", "debug"])
def test_artifact_listing_shows_a_deleted_directory_as_empty(
    tmp_path: Path, subdir: str
) -> None:
    runtime = _runtime()
    runtime.artifacts = ArtifactStore(tmp_path)
    runtime.artifacts.write_snapshot("runtime", {"foo": "bar"})
    client = TestClient(build_app(runtime), base_url="http://localhost")
    shutil.rmtree(tmp_path / subdir)

    response = client.get("/artifacts")

    assert response.status_code == 200
    assert response.json()[subdir] == []
