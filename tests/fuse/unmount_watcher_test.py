"""Behavioral tests: teardown stops the unmount watcher also when a forked child lives.

A fork-started DataLoader worker inherits the write end of the watcher's stdin
pipe. Before, the watcher waited for EOF on that pipe, which never came while the
worker lived, so teardown waited for the watcher and then failed with a timeout.
"""

import os
import time
import warnings
from pathlib import Path

from trainctl.config import TrainctlConfig
from trainctl.load_backend import load_backend
from trainctl.runtime.runtime import TrainctlRuntime


class _Module:
    device = "cpu"


def _fork_holding_child() -> tuple[int, int]:
    """Forks a child that lives until the returned pipe end is written or closed."""
    release_read, release_write = os.pipe()
    with warnings.catch_warnings():
        # The FUSE thread runs; the child only blocks in read() and exits.
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        os.close(release_write)
        _ = os.read(release_read, 1)
        os._exit(0)
    os.close(release_read)
    return pid, release_write


def test_teardown_with_a_forked_child_alive_is_prompt(tmp_path: Path) -> None:
    runtime = TrainctlRuntime(
        TrainctlConfig(
            rest_enabled=False,
            torch_debug_enabled=False,
            inspection_strict=True,
            artifact_dir=tmp_path / "artifacts",
            fuse_mountpoint=tmp_path / "procfs",
        ),
        load_backend("lightning.pytorch"),
    )
    runtime.setup_environment(_Module(), "fit")
    child, release = _fork_holding_child()
    try:
        started = time.monotonic()
        runtime.teardown(_Module(), "fit")
        assert time.monotonic() - started < 4
    finally:
        os.close(release)
        _ = os.waitpid(child, 0)
