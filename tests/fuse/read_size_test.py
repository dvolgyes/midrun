"""Behavioral tests: a read of a mounted file returns all the bytes that its open made.

The kernel limits splice and sendfile reads (system `cat`, `shutil.copyfile`) to the
size that `getattr` reported. That size must be the length of the content that the
open handle serves, also when the content changes between two calls.
"""

import itertools
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.runtime import virtual_fs
from trainctl.runtime.runtime import TrainctlRuntime


class _Module:
    device = "cpu"


@pytest.fixture
def mountpoint(tmp_path: Path) -> Iterator[Path]:
    runtime = TrainctlRuntime(
        TrainctlConfig(
            rest_enabled=False,
            torch_debug_enabled=False,
            artifact_dir=tmp_path / "artifacts",
            fuse_mountpoint=tmp_path / "procfs",
        ),
        load_backend("lightning.pytorch"),
    )
    runtime.setup_environment(_Module(), "fit")
    yield tmp_path / "procfs"
    runtime.teardown(_Module(), "fit")


# A generated file, and a live file that reports no size before it is opened.
@pytest.fixture(params=["/state/status", "/model/graph.dot"])
def growing_file(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Makes the file one byte longer each time it is generated."""
    lengths = itertools.count(1)
    generate = virtual_fs.generate

    def _growing(runtime: TrainctlRuntime, path: str) -> bytes | None:
        if path == request.param:
            return b"x" * next(lengths) + b"\n"
        return generate(runtime, path)

    monkeypatch.setattr(virtual_fs, "generate", _growing)
    return request.param.lstrip("/")


def _complete(content: bytes) -> bool:
    return content.endswith(b"x\n")


def test_copyfile_reads_a_changing_file_completely(
    mountpoint: Path, tmp_path: Path, growing_file: str
) -> None:
    copy = tmp_path / "copy"
    for _ in range(20):
        shutil.copyfile(mountpoint / growing_file, copy)
        assert _complete(copy.read_bytes()), copy.read_bytes()


def test_cat_after_stat_reads_a_changing_file_completely(
    mountpoint: Path, growing_file: str
) -> None:
    path = mountpoint / growing_file
    for _ in range(20):
        _ = path.stat()
        content = subprocess.run(  # noqa: S603 -- fixed command, test-owned path
            ["cat", str(path)],  # noqa: S607 -- the system cat is what is tested
            capture_output=True,
            check=True,
        ).stdout
        assert _complete(content), content
