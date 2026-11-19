"""Behavioral tests for the /artifact-files StaticFiles mount and GET /artifacts/{id}'s files_url.

Checkpoints/snapshots/debug captures are real files under `runtime.artifacts.root`;
the mount serves them directly rather than through a hand-rolled download route.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.runtime import MidrunRuntime


def _client(tmp_path: Path) -> tuple[TestClient, MidrunRuntime]:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return TestClient(build_app(runtime), base_url="http://localhost"), runtime


def test_get_artifact_includes_files_url(tmp_path: Path) -> None:
    client, runtime = _client(tmp_path)
    manifest = runtime.artifacts.write_debug_capture("demo", {"summary.txt": "hello"})

    response = client.get(f"/artifacts/{manifest['id']}")

    assert response.status_code == 200
    assert response.json()["files_url"] == f"/artifact-files/debug/{manifest['id']}"


def test_artifact_files_mount_serves_debug_capture_bytes(tmp_path: Path) -> None:
    client, runtime = _client(tmp_path)
    manifest = runtime.artifacts.write_debug_capture(
        "demo", {"summary.txt": "hello world"}
    )

    response = client.get(f"/artifact-files/debug/{manifest['id']}/summary.txt")

    assert response.status_code == 200
    assert response.text == "hello world"


def test_artifact_files_mount_serves_snapshot_payload(tmp_path: Path) -> None:
    client, runtime = _client(tmp_path)
    manifest = runtime.artifacts.write_snapshot("runtime", {"foo": "bar"})

    response = client.get(f"/artifact-files/snapshots/{manifest['id']}/runtime.json")

    assert response.status_code == 200
    assert response.json() == {"foo": "bar"}


def test_artifact_files_mount_404_for_unknown_file(tmp_path: Path) -> None:
    client, _runtime = _client(tmp_path)

    response = client.get("/artifact-files/debug/does-not-exist/summary.txt")

    assert response.status_code == 404


def test_artifact_files_mount_never_renders_an_uploaded_page(tmp_path: Path) -> None:
    client, _runtime = _client(tmp_path)
    capture = client.post(
        "/debug/captures",
        json={"kind": "x", "files": {"evil.html": "<script>alert(1)</script>"}},
    ).json()

    response = client.get(f"/artifact-files/debug/{capture['id']}/evil.html")

    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"] == "sandbox"
    assert response.headers["content-disposition"].startswith("attachment")


def test_artifact_files_mount_keeps_svg_type_for_the_dashboard_preview(
    tmp_path: Path,
) -> None:
    client, runtime = _client(tmp_path)
    manifest = runtime.artifacts.write_debug_capture(
        "graph", {"graph.svg": "<svg xmlns='http://www.w3.org/2000/svg'/>"}
    )

    response = client.get(f"/artifact-files/debug/{manifest['id']}/graph.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.headers["content-security-policy"] == "sandbox"


def test_get_checkpoints_lists_each_checkpoint_with_its_download_url(
    tmp_path: Path,
) -> None:
    client, runtime = _client(tmp_path)
    assert client.get("/checkpoints").json() == {"checkpoints": []}
    path = runtime.artifacts.reserve_checkpoint_path("before-lr")
    (reserved,) = client.get("/checkpoints").json()["checkpoints"]
    assert reserved["size_bytes"] == 0
    assert client.get(reserved["url"]).content == b""
    path.write_bytes(b"12345")

    (entry,) = client.get("/checkpoints").json()["checkpoints"]

    assert entry["name"] == path.name
    assert entry["size_bytes"] == 5
    assert entry["modified_at"] == path.stat().st_mtime
    assert client.get(entry["url"]).content == b"12345"


@pytest.mark.parametrize("category", ["debug", "snapshots"])
@pytest.mark.parametrize("write_fails", [False, True])
def test_directory_publication_hides_incomplete_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, category: str, write_fails: bool
) -> None:
    root = tmp_path / "artifacts"
    client, runtime = _client(root)
    store = runtime.artifact_store
    filename = "data.bin" if category == "debug" else "metrics.json"
    artifact_id = "capture_000001"
    monkeypatch.setattr("midrun.runtime.artifacts.next_id", lambda _: artifact_id)
    entered, resume = Event(), Event()
    write_bytes = Path.write_bytes

    def pause_write(path: Path, data: bytes) -> int:
        if path.name == filename:
            write_bytes(path, data[:1])
            entered.set()
            assert resume.wait(10)
            if write_fails:
                raise OSError("disk write failed")
        return write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", pause_write)
    url = f"/artifact-files/{category}/{artifact_id}/{filename}"
    with ThreadPoolExecutor(max_workers=1) as pool:
        if category == "debug":
            pending = pool.submit(
                store.write_debug_capture, "sample", {filename: b"abc"}
            )
        else:
            pending = pool.submit(store.write_snapshot, "metrics", {"step": 2})
        try:
            assert entered.wait(10)
            (partial,) = root.rglob(filename)
            assert partial.read_bytes() != b""
            assert (
                client.get(f"/artifact-files/{partial.relative_to(root)}").status_code
                == 404
            )
            assert client.get("/artifacts").json()[category] == []
            assert client.get(f"/artifacts/{artifact_id}").status_code == 404
            assert client.get(url).status_code == 404
        finally:
            resume.set()
        if write_fails:
            with pytest.raises(OSError, match="disk write failed"):
                pending.result()
            assert client.get("/artifacts").json()[category] == []
            assert client.get(f"/artifacts/{artifact_id}").status_code == 404
            assert client.get(url).status_code == 404
        else:
            manifest = pending.result()
            assert client.get("/artifacts").json()[category] == [manifest["id"]]
            assert client.get(f"/artifacts/{artifact_id}").json()["id"] == artifact_id
            response = client.get(url)
            if category == "debug":
                assert response.content == b"abc"
            else:
                assert response.json() == {"step": 2}
    assert not partial.parent.exists()
    assert {entry.name for entry in root.iterdir()} == {
        "debug",
        "snapshots",
        "checkpoints",
    }
