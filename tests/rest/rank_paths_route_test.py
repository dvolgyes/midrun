"""Behavioral tests for the meta/ranks subtree through GET /files and the shared virtual-tree functions.

FUSE and REST resolve rank paths with the same parser, so a directory answers the
same with or without a trailing slash, and a rank or worker that does not exist is
neither a directory nor a file.
"""

import pytest
from fastapi.testclient import TestClient

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.state.rank_info import RankInfo
from midrun.runtime.virtual_fs.is_virtual_dir import is_virtual_dir
from midrun.runtime.virtual_fs.list_virtual_dir import list_virtual_dir
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content


def _runtime() -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = build_runtime(config, load_backend("lightning.pytorch"))
    rank = RankInfo(
        global_rank=0,
        local_rank=0,
        pid=999,
        hostname="h",
        device="cpu",
        worker_pids=(111,),
    )
    runtime.state.update(world_size=1, ranks=(rank,))
    return runtime


@pytest.mark.parametrize(
    ("path", "entry"),
    [
        ("meta/ranks", "000"),
        ("meta/ranks/000", "workers"),
        ("meta/ranks/000/workers", "000"),
        ("meta/ranks/000/workers/000", "pid"),
        ("model", "class"),
    ],
)
def test_directory_lists_the_same_with_and_without_a_trailing_slash(
    path: str, entry: str
) -> None:
    client = TestClient(build_app(_runtime()), base_url="http://localhost")

    plain = client.get(f"/files/{path}")
    slashed = client.get(f"/files/{path}/")

    assert plain.status_code == slashed.status_code == 200
    assert slashed.json() == plain.json()
    assert entry in plain.json()["entries"]


@pytest.mark.parametrize("path", ["meta/ranks/000/pid/", "model/class/"])
def test_file_with_a_trailing_slash_is_not_found(path: str) -> None:
    client = TestClient(build_app(_runtime()), base_url="http://localhost")

    assert client.get(f"/files/{path}").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/meta/ranks/001",
        "/meta/ranks/999",
        "/meta/ranks/001/workers",
        "/meta/ranks/000/workers/001",
        "/meta/ranks/000/nonexistent",
        "/meta/ranks/000/workers/000/nonexistent",
    ],
)
def test_missing_rank_or_worker_is_neither_a_directory_nor_a_file(path: str) -> None:
    runtime = _runtime()

    assert not is_virtual_dir(runtime, path)
    assert list_virtual_dir(runtime, path) == []
    assert virtual_file_content(runtime, path) is None
