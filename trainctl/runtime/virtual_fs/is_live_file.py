"""Checks whether a virtual path is a live file."""

from trainctl.runtime.virtual_fs.live_files import LIVE_FILES


def is_live_file(path: str) -> bool:
    """Returns whether `path` is one of the lazy, safe-point-blocking live files."""
    return path in LIVE_FILES
