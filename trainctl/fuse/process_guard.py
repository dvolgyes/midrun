"""Keeps a Trainctl FUSE mount from changing the process's signal handling or outliving it."""

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from trainctl.fuse import unmount_watcher
from trainctl.fuse.mountpoint_check import FUSE_FSNAME

_READY_TIMEOUT_S = 10.0
_READY_POLL_S = 0.05

# libfuse's fuse_main installs C handlers for these where the disposition is SIG_DFL
# (Python still reports SIG_DFL); its SIGTERM handler unmounts and lets the process run on.
_FUSE_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)


def mount_guarded(
    mountpoint: Path,
    fuse_thread: threading.Thread,
    wait_ready: Callable[[float], bool],
) -> subprocess.Popen[bytes]:
    """Starts the FUSE loop, then undoes libfuse's signal handlers and watches for process death.

    Args:
        mountpoint: The resolved directory that `fuse_thread` mounts.
        fuse_thread: The not yet started thread that runs the FUSE loop.
        wait_ready: Returns whether the mount became ready within the given seconds.

    Returns:
        The unmount watcher process. Closing its stdin tells it to check once and exit.

    Raises:
        RuntimeError: The mount failed or did not become ready in time.
    """
    saved = {sig: signal.getsignal(sig) for sig in _FUSE_SIGNALS}
    try:
        fuse_thread.start()
        _wait_for_mount(mountpoint, fuse_thread, wait_ready)
    finally:
        _restore_signal_handlers(saved)
    # The watcher gets its own session, so a terminal Ctrl+C that ends the training
    # process does not end the watcher before it can unmount.
    return subprocess.Popen(  # noqa: S603 -- fixed interpreter and script
        [
            sys.executable,
            "-I",
            unmount_watcher.__file__,
            str(os.getpid()),
            str(mountpoint),
            FUSE_FSNAME,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_for_mount(
    path: Path, fuse_thread: threading.Thread, wait_ready: Callable[[float], bool]
) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while not wait_ready(_READY_POLL_S):
        if not fuse_thread.is_alive():
            raise RuntimeError(
                f"FUSE mount at {path} failed; libfuse printed the reason on stderr"
            )
        if time.monotonic() > deadline:
            raise RuntimeError(f"FUSE mount at {path} did not become ready in time")


def _restore_signal_handlers(saved: dict[signal.Signals, Any]) -> None:
    if threading.current_thread() is not threading.main_thread():
        logger.warning(
            "trainctl: FUSE mounted outside the main thread; libfuse keeps its "
            "SIGHUP/SIGTERM handlers, which unmount without ending the process"
        )
        return
    for sig, handler in saved.items():
        signal.signal(sig, handler)
