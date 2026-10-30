"""Behavioral tests: a mount that misses the readiness timeout is not left behind.

Before, `mount_guarded` raised on the timeout but left the FUSE thread running, so
its mount stayed (or appeared later) with no watcher or teardown to remove it.
"""

import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import mfusepy
import pytest

from midrun.fuse import process_guard


def _mounted(path: Path) -> bool:
    mountinfo = Path("/proc/self/mountinfo").read_text().splitlines()
    return any(line.split()[4] == str(path) for line in mountinfo)


@pytest.fixture
def mountpoint(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "procfs"
    path.mkdir()
    yield path
    while _mounted(path):
        subprocess.run(["fusermount3", "-uz", str(path)], check=False)  # noqa: S603, S607 -- test-owned mount


def _never_ready(timeout: float) -> bool:
    """Reports a mount that is live but never signals readiness in time."""
    threading.Event().wait(timeout)
    return False


def test_ready_timeout_unmounts_and_stops_the_fuse_thread(
    mountpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_guard, "_READY_TIMEOUT_S", 0.5)
    thread = threading.Thread(
        target=lambda: mfusepy.FUSE(
            mfusepy.Operations(), str(mountpoint), foreground=True, fsname="midrun"
        ),
        daemon=True,
    )
    with pytest.raises(RuntimeError, match="did not become ready in time"):
        _ = process_guard.mount_guarded(mountpoint, thread, _never_ready)
    assert not _mounted(mountpoint)
    assert not thread.is_alive()
