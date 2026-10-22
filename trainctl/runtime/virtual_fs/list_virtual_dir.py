"""Lists the children of a virtual directory."""

from typing import TYPE_CHECKING

from trainctl.runtime.virtual_fs.live_files import LIVE_FILES
from trainctl.runtime.virtual_fs.rank_path_fields import RANK_FIELDS, WORKER_FIELDS
from trainctl.runtime.virtual_fs.resolve_rank import resolve_rank
from trainctl.runtime.virtual_fs.static_files import STATIC_FILES

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def list_virtual_dir(runtime: "TrainctlRuntime", path: str) -> list[str]:
    """Returns `path`'s child names, sorted, without `.`/`..` (a FUSE-only convention)."""
    if path == "/meta/ranks":
        count = len(runtime.state.read().ranks)
        return [f"{i:03d}" for i in range(count)]

    parts = path.split("/")
    ranks = runtime.state.read().ranks
    if len(parts) == 4 and parts[1] == "meta" and parts[2] == "ranks":
        return [*RANK_FIELDS, "workers"]
    if (
        len(parts) == 5
        and parts[1] == "meta"
        and parts[2] == "ranks"
        and parts[4] == "workers"
    ):
        rank = resolve_rank(parts, ranks)
        count = len(rank.worker_pids) if rank is not None else 0
        return [f"{i:03d}" for i in range(count)]
    if (
        len(parts) == 6
        and parts[1] == "meta"
        and parts[2] == "ranks"
        and parts[4] == "workers"
    ):
        return [*WORKER_FIELDS]

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
