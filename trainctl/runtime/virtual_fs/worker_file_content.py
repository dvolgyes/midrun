"""The content of a file in a meta/ranks/<idx>/workers/<widx>/ directory."""

from typing import TYPE_CHECKING

from trainctl.runtime.virtual_fs.encode_text import encode_text
from trainctl.runtime.virtual_fs.listed_index import listed_index
from trainctl.runtime.virtual_fs.rank_path_fields import WORKER_FIELDS
from trainctl.runtime.virtual_fs.resolve_rank import resolve_rank

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def worker_file_content(runtime: "TrainctlRuntime", path: str) -> bytes | None:
    """Returns the content of a `/meta/ranks/<idx>/workers/<widx>/<field>` file, or `None`."""
    parts = path.split("/")
    if (
        len(parts) != 7
        or parts[1] != "meta"
        or parts[2] != "ranks"
        or parts[4] != "workers"
        or parts[6] not in WORKER_FIELDS
    ):
        return None
    rank = resolve_rank(parts, runtime.state.read().ranks)
    if rank is None:
        return None
    widx = listed_index(parts, 5, len(rank.worker_pids))
    if widx is None:
        return None
    return encode_text(str(rank.worker_pids[widx]))
