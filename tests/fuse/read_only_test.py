"""Behavioral tests: the mount is read-only, and writes fail with EROFS.

Before, the mount was read-write: a write-mode open succeeded, and only the write
failed, with ENOSYS.
"""

import errno
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import mfusepy
import pytest

from midrun.config import MidrunConfig
from midrun.fuse.filesystem import MidrunFS
from midrun.load_backend import load_backend
from midrun.runtime.runtime import MidrunRuntime


class _Module:
    device = "cpu"


def _runtime(tmp_path: Path, **overrides: object) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False,
        torch_debug_enabled=False,
        artifact_dir=tmp_path / "artifacts",
        **overrides,
    )
    return MidrunRuntime(config, load_backend("lightning.pytorch"))


@pytest.fixture
def mountpoint(tmp_path: Path) -> Iterator[Path]:
    runtime = _runtime(tmp_path, fuse_mountpoint=tmp_path / "procfs")
    runtime.setup_environment(_Module(), "fit")
    yield tmp_path / "procfs"
    runtime.teardown(_Module(), "fit")


@pytest.mark.parametrize(
    "change",
    [
        lambda root: (root / "state" / "status").write_text("x"),
        lambda root: os.close(os.open(root / "state" / "status", os.O_RDWR)),
        lambda root: (root / "new.txt").touch(),
        lambda root: (root / "new-dir").mkdir(),
        lambda root: (root / "state" / "status").unlink(),
        lambda root: (root / "state" / "status").chmod(0o666),
    ],
    ids=["write", "open-rdwr", "create", "mkdir", "unlink", "chmod"],
)
def test_changes_to_the_mount_fail_with_erofs(
    mountpoint: Path, change: Callable[[Path], object]
) -> None:
    with pytest.raises(OSError, match="Read-only file system") as raised:
        change(mountpoint)
    assert raised.value.errno == errno.EROFS


@pytest.mark.parametrize("flags", [os.O_WRONLY, os.O_RDWR])
def test_open_for_writing_is_refused(tmp_path: Path, flags: int) -> None:
    fs = MidrunFS(_runtime(tmp_path, fuse_enabled=False))

    with pytest.raises(mfusepy.FuseOSError) as raised:
        fs.open("/state/status", flags)

    assert raised.value.errno == errno.EROFS
