"""Keeps a Trainctl FUSE mount from changing the process's signal handling or outliving it."""

import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from trainctl.fuse import unmount_watcher

FUSE_FSNAME = "trainctl"

# libfuse's fuse_main installs C handlers for these where the disposition is SIG_DFL
# (Python still reports SIG_DFL); its SIGTERM handler unmounts and lets the process run on.
_FUSE_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)


def mount_guarded(
    mountpoint: Path | str,
    fuse_thread: threading.Thread,
    wait_ready: Callable[[float], bool],
) -> subprocess.Popen[bytes]:
    """Starts the FUSE loop, then undoes libfuse's signal handlers and watches for process death.

    Args:
        mountpoint: The directory that `fuse_thread` mounts.
        fuse_thread: The not yet started thread that runs the FUSE loop.
        wait_ready: Returns whether the mount became ready within the given seconds.

    Returns:
        The unmount watcher process. Closing its stdin tells it to check once and exit.

    Raises:
        RuntimeError: The mount did not become ready in time.
    """
    saved = {sig: signal.getsignal(sig) for sig in _FUSE_SIGNALS}
    try:
        fuse_thread.start()
        if not wait_ready(10):
            raise RuntimeError("FUSE mount did not become ready in time")
    finally:
        _restore_signal_handlers(saved)
    return _start_unmount_watcher(mountpoint)


def _restore_signal_handlers(saved: dict[signal.Signals, Any]) -> None:
    if threading.current_thread() is not threading.main_thread():
        logger.warning(
            "trainctl: FUSE mounted outside the main thread; libfuse keeps its "
            "SIGHUP/SIGTERM handlers, which unmount without ending the process"
        )
        return
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _start_unmount_watcher(mountpoint: Path | str) -> subprocess.Popen[bytes]:
    # The watcher gets its own session, so a terminal Ctrl+C that ends the training
    # process does not end the watcher before it can unmount.
    return subprocess.Popen(  # noqa: S603 -- fixed interpreter and script
        [
            sys.executable,
            "-I",
            unmount_watcher.__file__,
            str(os.getpid()),
            str(Path(mountpoint).resolve()),
            FUSE_FSNAME,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )
