"""Behavioral tests: a failing file generator gives the same errno to stat and open.

Before, a generator exception inside `getattr` escaped to mfusepy, which returned
EINVAL; only `open` turned it into EIO (or ETIMEDOUT for a timeout).
"""

import errno
from collections.abc import Iterator
from pathlib import Path

import pytest

from trainctl.config import TrainctlConfig
from trainctl.fuse import filesystem
from trainctl.load_backend import load_backend
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


@pytest.mark.parametrize(
    ("failure", "expected_errno"),
    [(RuntimeError("generator bug"), errno.EIO), (TimeoutError(), errno.ETIMEDOUT)],
    ids=["error", "timeout"],
)
def test_stat_of_a_failing_generated_file_reports_its_errno(
    mountpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_errno: int,
) -> None:
    content_of = filesystem.virtual_file_content

    def _failing(runtime: TrainctlRuntime, path: str) -> bytes | None:
        if path == "/state/status":
            raise failure
        return content_of(runtime, path)

    monkeypatch.setattr(filesystem, "virtual_file_content", _failing)
    with pytest.raises(OSError) as raised:
        _ = (mountpoint / "state" / "status").stat()
    assert raised.value.errno == expected_errno
