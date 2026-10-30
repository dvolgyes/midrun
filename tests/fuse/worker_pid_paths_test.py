"""Behavioral tests for the meta/ranks/<idx>/workers/<widx>/pid FUSE dynamic subtree.

Exercises `MidrunFS` directly against a `MidrunRuntime` with hand-published
rank state, rather than mounting real FUSE -- the routing logic under test lives
entirely in `virtual_file_content`/`is_virtual_dir`/`MidrunFS.readdir`, not in the
kernel mount.
"""

import pytest

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.fuse.filesystem import MidrunFS
from midrun.load_backend import load_backend
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.state.rank_info import RankInfo
from midrun.runtime.virtual_fs.is_virtual_dir import is_virtual_dir
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content


def _runtime_with_rank(worker_pids: tuple[int, ...] = (111, 222)) -> MidrunRuntime:
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
        worker_pids=worker_pids,
    )
    runtime.state.update(world_size=1, ranks=(rank,))
    return runtime


def test_workers_dir_listed_under_rank() -> None:
    fs = MidrunFS(_runtime_with_rank())
    assert fs.readdir("/meta/ranks/000", 0) == [
        ".",
        "..",
        "pid",
        "hostname",
        "global-rank",
        "local-rank",
        "device",
        "workers",
    ]


def test_worker_indices_listed() -> None:
    fs = MidrunFS(_runtime_with_rank((111, 222)))
    assert fs.readdir("/meta/ranks/000/workers", 0) == [".", "..", "000", "001"]


def test_worker_indices_empty_when_no_workers() -> None:
    fs = MidrunFS(_runtime_with_rank(()))
    assert fs.readdir("/meta/ranks/000/workers", 0) == [".", ".."]


def test_worker_fields_listed_under_worker_index() -> None:
    fs = MidrunFS(_runtime_with_rank((111,)))
    assert fs.readdir("/meta/ranks/000/workers/000", 0) == [".", "..", "pid"]


def test_worker_pid_file_content() -> None:
    runtime = _runtime_with_rank((111, 222))
    assert virtual_file_content(runtime, "/meta/ranks/000/workers/000/pid") == b"111\n"
    assert virtual_file_content(runtime, "/meta/ranks/000/workers/001/pid") == b"222\n"


def test_worker_pid_file_read_and_getattr_round_trip() -> None:
    fs = MidrunFS(_runtime_with_rank((111,)))
    attrs = fs.getattr("/meta/ranks/000/workers/000/pid")
    assert attrs["st_size"] == len(b"111\n")

    fh = fs.open("/meta/ranks/000/workers/000/pid", 0)
    try:
        assert fs.read("/meta/ranks/000/workers/000/pid", 4096, 0, fh) == b"111\n"
    finally:
        fs.release("/meta/ranks/000/workers/000/pid", fh)


def test_worker_index_out_of_range_is_not_found() -> None:
    runtime = _runtime_with_rank((111,))
    assert virtual_file_content(runtime, "/meta/ranks/000/workers/001/pid") is None
    assert not is_virtual_dir(runtime, "/meta/ranks/000/workers/001")


def test_rank_index_out_of_range_is_not_found() -> None:
    runtime = _runtime_with_rank((111,))
    assert virtual_file_content(runtime, "/meta/ranks/007/workers/000/pid") is None
    assert not is_virtual_dir(runtime, "/meta/ranks/007/workers")


@pytest.mark.parametrize("alias", ["0", "-0", "+0", "00", "0000", "0_0", " 000"])
def test_index_spellings_that_are_not_listed_are_not_found(alias: str) -> None:
    runtime = _runtime_with_rank((111,))
    assert virtual_file_content(runtime, f"/meta/ranks/{alias}/pid") is None
    assert not is_virtual_dir(runtime, f"/meta/ranks/{alias}")
    assert virtual_file_content(runtime, f"/meta/ranks/000/workers/{alias}/pid") is None
    assert not is_virtual_dir(runtime, f"/meta/ranks/000/workers/{alias}")
