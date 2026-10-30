"""Checks whether a virtual path is a directory."""

from typing import TYPE_CHECKING

from midrun.runtime.virtual_fs.listed_index import listed_index
from midrun.runtime.virtual_fs.live_files import LIVE_FILES
from midrun.runtime.virtual_fs.resolve_rank import resolve_rank
from midrun.runtime.virtual_fs.static_files import STATIC_FILES

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def is_virtual_dir(runtime: "MidrunRuntime", path: str) -> bool:
    """Returns whether `path` names a virtual directory (not a file) in this tree."""
    if path == "/":
        return True
    if path == "/meta/ranks":
        return True
    prefix = path.rstrip("/") + "/"
    if any(candidate.startswith(prefix) for candidate in (*STATIC_FILES, *LIVE_FILES)):
        return True
    ranks = runtime.state.read().ranks
    parts = path.split("/")
    if len(parts) == 4 and parts[1] == "meta" and parts[2] == "ranks":
        return listed_index(parts, 3, len(ranks)) is not None
    if (
        len(parts) == 5
        and parts[1] == "meta"
        and parts[2] == "ranks"
        and parts[4] == "workers"
    ):
        return listed_index(parts, 3, len(ranks)) is not None
    if (
        len(parts) == 6
        and parts[1] == "meta"
        and parts[2] == "ranks"
        and parts[4] == "workers"
    ):
        rank = resolve_rank(parts, ranks)
        if rank is None:
            return False
        return listed_index(parts, 5, len(rank.worker_pids)) is not None
    return False
