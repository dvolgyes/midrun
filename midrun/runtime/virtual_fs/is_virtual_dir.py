"""Checks whether a virtual path is a directory."""

from typing import TYPE_CHECKING

from midrun.runtime.virtual_fs.live_files import LIVE_FILES
from midrun.runtime.virtual_fs.parse_rank_path import parse_rank_path
from midrun.runtime.virtual_fs.static_files import STATIC_FILES

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def is_virtual_dir(runtime: "MidrunRuntime", path: str) -> bool:
    """Returns whether `path` names a virtual directory (not a file) in this tree."""
    if path == "/":
        return True
    prefix = path.rstrip("/") + "/"
    if any(candidate.startswith(prefix) for candidate in (*STATIC_FILES, *LIVE_FILES)):
        return True
    parsed = parse_rank_path(path, runtime.state.read().ranks)
    return parsed is not None and parsed.is_dir
