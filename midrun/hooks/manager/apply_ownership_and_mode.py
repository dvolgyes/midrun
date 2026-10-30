"""Gives a copied path the ownership and mode of its source."""

import errno
import os
from pathlib import Path


def apply_ownership_and_mode(path: Path, uid: int, gid: int, mode: int) -> bool:
    """Sets the owner, group and mode of `path`; calls chown only when they differ.

    A user that is not root cannot give a file another owner, or a group that the
    user is not a member of (EPERM). In a user namespace, such as a rootless
    container, an owner or group that the namespace does not map cannot be set
    either (EINVAL). Then `path` keeps its owner and group, and gets `mode`.

    Returns:
        Whether `path` has the owner `uid` and the group `gid`.
    """
    current = path.stat()
    preserved = True
    if current.st_uid != uid or current.st_gid != gid:
        try:
            os.chown(path, uid, gid)
        except OSError as exc:
            if not isinstance(exc, PermissionError) and exc.errno != errno.EINVAL:
                raise
            preserved = False
    path.chmod(mode)
    return preserved
