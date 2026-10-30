"""Behavioral tests for POST /debug/captures and its artifact round-trip."""

from pathlib import Path

from fastapi.testclient import TestClient

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore


def _client(tmp_path: Path) -> TestClient:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return TestClient(build_app(runtime), base_url="http://localhost")


def test_post_debug_captures_round_trip(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/debug/captures",
        json={
            "kind": "pyspy-stack",
            "files": {"result.json": '{"threads": []}'},
            "metadata": {"pid": 123},
        },
    )
    assert response.status_code == 200
    manifest = response.json()
    assert manifest["kind"] == "pyspy-stack"
    assert manifest["metadata"] == {"pid": 123}

    listed = client.get("/artifacts").json()
    assert manifest["id"] in listed["debug"]

    fetched = client.get(f"/artifacts/{manifest['id']}").json()
    assert fetched["id"] == manifest["id"]
    assert fetched["kind"] == "pyspy-stack"


def test_post_debug_captures_defaults_files_and_metadata_to_empty(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    response = client.post("/debug/captures", json={"kind": "gdb-snapshot"})
    assert response.status_code == 200
    manifest = response.json()
    assert manifest["files"] == []
    assert manifest["metadata"] == {}


def test_post_debug_captures_rejects_path_traversal(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/debug/captures", json={"kind": "bad", "files": {"../x.txt": "y"}}
    )
    assert response.status_code == 422


def test_get_unknown_artifact_id_is_404(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/artifacts/dbg_999999")
    assert response.status_code == 404


def test_post_debug_captures_cannot_overwrite_the_manifest(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/debug/captures", json={"kind": "bad", "files": {"manifest.json": "{}"}}
    )
    assert response.status_code == 422
    assert client.get("/artifacts").json()["debug"] == []


def test_post_checkpoints_rejects_traversal_label(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post("/checkpoints", json={"label": "../../../escape"})
    assert response.status_code == 422
    assert client.get("/commands").json() == []
