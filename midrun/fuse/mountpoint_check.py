"""Checks a directory before Midrun mounts its FUSE filesystem there."""

import errno
import os
import subprocess
from pathlib import Path

from loguru import logger

from midrun.fuse import unmount_watcher

FUSE_FSNAME = "midrun"

# The kernel's answers for a FUSE mount that no process serves. ECONNREFUSED was seen
# when the process died right after mounting, ENOTCONN otherwise.
_DEAD_MOUNT_ERRNOS = frozenset({errno.ENOTCONN, errno.ECONNREFUSED})


def prepare_mountpoint(mountpoint: Path | str) -> Path:
    """Makes `mountpoint` ready for a new Midrun mount and returns its resolved path.

    A stale Midrun mount (its process died) is removed. A missing directory is
    made.

    Raises:
        RuntimeError: `mountpoint` holds a live mount or files.
    """
    path = Path(mountpoint).resolve()
    _remove_stale_mount(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise RuntimeError(
            f"FUSE mountpoint {path} is not empty; set fuse_mountpoint to an empty directory"
        )
    return path


def _remove_stale_mount(path: Path) -> None:
    mount = unmount_watcher.mount_at(str(path))
    if mount is None:
        return
    if mount != ("fuse", FUSE_FSNAME):
        raise RuntimeError(
            f"{path} is already a mount point, not a midrun mount; "
            "set fuse_mountpoint to another directory"
        )
    try:
        # statfs is not cached by the kernel, unlike stat.
        os.statvfs(path)
    except OSError as exc:
        if exc.errno not in _DEAD_MOUNT_ERRNOS:
            raise
        logger.warning(
            "midrun: removing the stale FUSE mount of an ended run at {}", path
        )
        subprocess.run(["fusermount3", "-uz", str(path)], check=True)  # noqa: S603, S607 -- stale midrun mount
        return
    raise RuntimeError(
        f"{path} is already mounted by another midrun run; "
        "set fuse_mountpoint to another directory"
    )
