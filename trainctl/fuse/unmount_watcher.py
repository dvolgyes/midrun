"""Removes a Trainctl FUSE mount after the training process ends, also after SIGKILL.

`TrainctlRuntime._start_fuse` runs this file by path (not as a package module, so
the watcher does not import torch or Lightning) with the training process's pid,
the mountpoint, and the FUSE filesystem name. It stops when its stdin closes (a
normal teardown, or the death of every process holding the pipe) or when its
parent pid changes (the training process died). Then, if a FUSE mount with that
filesystem name is still at the mountpoint, it lazily unmounts it.

fusermount's own `auto_unmount` option is not used: on a tested host (libfuse 2.9.9
and 3.18.2 with fusermount3 3.18.2) it left the dead mount in place after SIGKILL.

Usage: python -I unmount_watcher.py PARENT_PID MOUNTPOINT FSNAME
"""

import os
import select
import subprocess
import sys
from pathlib import Path

_POLL_S = 0.5

# /proc/self/mountinfo prints these characters of a path as octal escapes.
_MOUNTINFO_ESCAPES = (("\\", "\\134"), (" ", "\\040"), ("\t", "\\011"), ("\n", "\\012"))


def _is_our_mount(mountpoint: str, fsname: str) -> bool:
    expected = mountpoint
    for char, escape in _MOUNTINFO_ESCAPES:
        expected = expected.replace(char, escape)
    mountinfo = Path("/proc/self/mountinfo").read_text(
        encoding="utf-8", errors="replace"
    )
    for line in mountinfo.splitlines():
        fields, _, tail = line.partition(" - ")
        fs_type, source = [*tail.split(), "", ""][:2]
        if (
            fields.split()[4] == expected
            and fs_type.split(".")[0] == "fuse"
            and source == fsname
        ):
            return True
    return False


def _wait_for_parent_exit(parent_pid: int) -> None:
    stdin = sys.stdin.fileno()
    while os.getppid() == parent_pid:
        readable, _, _ = select.select([stdin], [], [], _POLL_S)
        if readable and not os.read(stdin, 4096):
            return


def main() -> int:
    """Waits for the training process to end, then removes its stale mount."""
    parent_pid, mountpoint, fsname = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    _wait_for_parent_exit(parent_pid)
    if _is_our_mount(mountpoint, fsname):
        subprocess.run(["fusermount3", "-uz", mountpoint], check=False)  # noqa: S603, S607 -- own mount
    return 0


if __name__ == "__main__":
    sys.exit(main())
