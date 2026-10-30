"""Behavioral tests for ArtifactStore.write_debug_capture (multi-file + metadata)."""

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from trainctl.runtime.artifacts import ArtifactStore


def test_write_debug_capture_writes_files_and_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.write_debug_capture(
        "pyspy-stack", {"result.json": '{"a": 1}'}, {"pid": 123}
    )

    dbg_dir = tmp_path / "debug" / manifest["id"]
    assert (dbg_dir / "result.json").read_text() == '{"a": 1}'

    written_manifest = json.loads((dbg_dir / "manifest.json").read_text())
    assert written_manifest["kind"] == "pyspy-stack"
    assert written_manifest["metadata"] == {"pid": 123}
    assert written_manifest["files"] == ["result.json"]


def test_write_debug_capture_writes_multiple_named_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.write_debug_capture(
        "gdb-snapshot", {"result.json": "{}", "raw.txt": "hello"}
    )

    dbg_dir = tmp_path / "debug" / manifest["id"]
    assert (dbg_dir / "raw.txt").read_text() == "hello"
    assert manifest["files"] == ["raw.txt", "result.json"]
    assert manifest["metadata"] == {}


@pytest.mark.parametrize("bad_filename", ["../escape.txt", "sub/dir.txt", ".", ".."])
def test_write_debug_capture_rejects_path_traversal(
    tmp_path: Path, bad_filename: str
) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="invalid capture filename"):
        store.write_debug_capture("bad", {bad_filename: "x"})


def test_write_debug_capture_ids_dont_collide_under_concurrency(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ids: list[str] = []
    lock = threading.Lock()

    def _write() -> None:
        manifest = store.write_debug_capture("kind", {"f.txt": "x"})
        with lock:
            ids.append(manifest["id"])

    threads = [threading.Thread(target=_write) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ids) == 20
    assert len(set(ids)) == 20


@pytest.mark.parametrize(
    "bad_label", ["../../outside", "a/b", "..", "", "x\0y", ".hidden", "x" * 65]
)
def test_checkpoint_path_rejects_unsafe_labels(tmp_path: Path, bad_label: str) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="invalid checkpoint label"):
        store.checkpoint_path(bad_label)


def test_checkpoint_path_stays_in_checkpoints_dir(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    path = store.checkpoint_path("before-lr_change.v2")
    assert path.parent == tmp_path / "checkpoints"
    assert path.name.endswith("-before-lr_change.v2.ckpt")


_WRITE_IN_A_NEW_PROCESS = """
import json
import sys
from pathlib import Path
from trainctl.runtime.artifacts import ArtifactStore

store = ArtifactStore(Path(sys.argv[1]))
if sys.argv[2] == "snapshot":
    manifest = store.write_snapshot("runtime", {"writer": sys.argv[3]})
else:
    manifest = store.write_debug_capture("kind", {"writer.txt": sys.argv[3]})
print(manifest["path"])
"""


@pytest.mark.parametrize(
    ("artifact", "payload"),
    [("snapshot", "runtime.json"), ("capture", "writer.txt")],
)
def test_processes_sharing_an_artifact_dir_keep_each_others_artifacts(
    tmp_path: Path, artifact: str, payload: str
) -> None:
    """Each process numbers its ids from 1; a later run or DDP rank must not overwrite."""
    paths = [
        subprocess.run(  # noqa: S603 -- fixed interpreter and script
            [
                sys.executable,
                "-c",
                _WRITE_IN_A_NEW_PROCESS,
                str(tmp_path),
                artifact,
                writer,
            ],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        for writer in ("first", "second")
    ]
    assert paths[0] != paths[1]
    contents = [(Path(path) / payload).read_text() for path in paths]
    assert "first" in contents[0]
    assert "second" in contents[1]


@pytest.mark.parametrize("bad_filename", ["", "x\0y", "manifest.json", ".hidden"])
def test_write_debug_capture_rejects_bad_name_before_writing(
    tmp_path: Path, bad_filename: str
) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="invalid capture filename"):
        store.write_debug_capture("bad", {"ok.txt": "x", bad_filename: "y"})
    assert list((tmp_path / "debug").iterdir()) == []
