"""Behavioral tests: Midrun checks its FUSE mountpoint before it mounts.

A stale Midrun mount from a killed run is replaced. A live mount, a mount of
another filesystem, or a non-empty directory is refused with a clear error, and a
failed mount is reported at once, not after the readiness timeout.
"""

import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.runtime import MidrunRuntime

# Mounts an empty filesystem with the given fsname and serves it until killed.
_FOREIGN_MOUNT = """
import sys
import mfusepy
mfusepy.FUSE(mfusepy.Operations(), sys.argv[1], foreground=True, fsname=sys.argv[2])
"""


class _Module:
    device = "cpu"


def _mounted(path: Path) -> bool:
    # os.path.ismount() is False for a dead FUSE mount (stat fails with ENOTCONN).
    mountinfo = Path("/proc/self/mountinfo").read_text().splitlines()
    return any(line.split()[4] == str(path) for line in mountinfo)


@pytest.fixture
def mountpoint(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "procfs"
    path.mkdir()
    yield path
    while _mounted(path):
        subprocess.run(["fusermount3", "-uz", str(path)], check=False)  # noqa: S603, S607 -- test-owned mount


def _mount_in_child(path: Path, fsname: str) -> subprocess.Popen[str]:
    proc = subprocess.Popen(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", _FOREIGN_MOUNT, str(path), fsname],
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while not _mounted(path):
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            _, stderr = proc.communicate(timeout=10)
            pytest.fail(f"child mount did not start: {stderr}")
        time.sleep(0.02)
    # The mount shows before fusermount3 hands it to the child, and a kill in that
    # window makes fusermount3 remove it. A served stat means the child has it.
    path.stat()
    return proc


def _runtime(path: Path, tmp_path: Path) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False,
        torch_debug_enabled=False,
        inspection_strict=True,
        artifact_dir=tmp_path / "artifacts",
        fuse_mountpoint=path,
    )
    return build_runtime(config, load_backend("lightning.pytorch"))


def test_stale_midrun_mount_is_replaced(mountpoint: Path, tmp_path: Path) -> None:
    child = _mount_in_child(mountpoint, "midrun")
    child.send_signal(signal.SIGKILL)
    child.wait(timeout=10)
    assert _mounted(mountpoint)

    runtime = _runtime(mountpoint, tmp_path)
    runtime.setup_environment(_Module(), "fit")
    try:
        assert (mountpoint / "state" / "status").read_text().strip() == "running"
    finally:
        runtime.teardown(_Module(), "fit")
    assert not _mounted(mountpoint)


@pytest.mark.parametrize(
    ("fsname", "message"),
    [("midrun", "another midrun run"), ("other", "not a midrun mount")],
)
def test_live_mount_is_refused(
    mountpoint: Path, tmp_path: Path, fsname: str, message: str
) -> None:
    child = _mount_in_child(mountpoint, fsname)
    try:
        runtime = _runtime(mountpoint, tmp_path)
        with pytest.raises(RuntimeError, match=message):
            runtime.setup_environment(_Module(), "fit")
        assert child.poll() is None
        assert _mounted(mountpoint)
    finally:
        child.kill()
        child.communicate(timeout=10)


def test_non_empty_mountpoint_is_refused(mountpoint: Path, tmp_path: Path) -> None:
    (mountpoint / "notes.txt").write_text("keep me")

    runtime = _runtime(mountpoint, tmp_path)
    with pytest.raises(RuntimeError, match="not empty"):
        runtime.setup_environment(_Module(), "fit")

    assert not _mounted(mountpoint)
    assert (mountpoint / "notes.txt").read_text() == "keep me"


def test_failed_mount_is_reported_at_once(mountpoint: Path, tmp_path: Path) -> None:
    mountpoint.chmod(0o500)  # fusermount3 refuses a mountpoint without write access
    try:
        runtime = _runtime(mountpoint, tmp_path)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="FUSE mount .* failed"):
            runtime.setup_environment(_Module(), "fit")
        assert time.monotonic() - started < 5
    finally:
        mountpoint.chmod(0o700)
