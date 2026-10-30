"""Lists the children of a virtual directory."""

from typing import TYPE_CHECKING

from midrun.runtime.virtual_fs.live_files import LIVE_FILES
from midrun.runtime.virtual_fs.parse_rank_path import parse_rank_path
from midrun.runtime.virtual_fs.rank_path_fields import RANK_FIELDS, WORKER_FIELDS
from midrun.runtime.virtual_fs.static_files import STATIC_FILES

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def list_virtual_dir(runtime: "MidrunRuntime", path: str) -> list[str]:
    """Returns `path`'s child names, sorted, without `.`/`..` (a FUSE-only convention)."""
    ranks = runtime.state.read().ranks
    parsed = parse_rank_path(path, ranks)
    if parsed is not None:
        match parsed.kind:
            case "ranks":
                return [f"{i:03d}" for i in range(len(ranks))]
            case "rank":
                return [*RANK_FIELDS, "workers"]
            case "workers" if parsed.rank is not None:
                return [f"{i:03d}" for i in range(len(parsed.rank.worker_pids))]
            case "worker":
                return [*WORKER_FIELDS]
            case _:
                return []

    prefix = path if path.endswith("/") else path + "/"
    names: set[str] = set()
    for candidate in (*STATIC_FILES, *LIVE_FILES, "/meta/ranks"):
        if not candidate.startswith(prefix):
            continue
        remainder = candidate[len(prefix) :]
        name = remainder.split("/", 1)[0]
        if name:
            names.add(name)
    return sorted(names)
