"""Removes an unpublished staging tree, including copies of read-only directories."""

import shutil
import stat
from pathlib import Path


def remove_staging_tree(staging_dir: Path) -> None:
    """Removes `staging_dir` and everything in it, as far as possible.

    A directory copied from a read-only baseline keeps its read-only mode, which
    would block removing its entries. So each directory is first made owner-only
    `rwx`, top-down, so each directory is writable before its turn. Removal errors
    are ignored: an unpublished staging tree is never used.
    """
    for directory, _dirnames, _filenames in staging_dir.walk():
        directory.chmod(stat.S_IRWXU)
    shutil.rmtree(staging_dir, ignore_errors=True)
