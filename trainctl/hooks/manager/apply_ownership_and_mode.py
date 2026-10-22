"""Gives a copied path the ownership and mode of its source."""

import os
from pathlib import Path


def apply_ownership_and_mode(path: Path, uid: int, gid: int, mode: int) -> None:
    """Sets the owner, group and mode of `path`; calls chown only when they differ.

    Raises:
        PermissionError: chown failed. The message names `path`, `uid` and `gid`.
    """
    current = path.stat()
    if current.st_uid != uid or current.st_gid != gid:
        try:
            os.chown(path, uid, gid)
        except PermissionError as exc:
            raise PermissionError(
                f"cannot set ownership of {path} to uid={uid}, gid={gid}: {exc}"
            ) from exc
    path.chmod(mode)
