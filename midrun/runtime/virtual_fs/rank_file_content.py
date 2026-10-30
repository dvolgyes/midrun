"""The content of a file in a meta/ranks/<idx>/ directory."""

from typing import TYPE_CHECKING

from midrun.runtime.virtual_fs.encode_text import encode_text
from midrun.runtime.virtual_fs.rank_path_fields import RANK_FIELDS
from midrun.runtime.virtual_fs.resolve_rank import resolve_rank

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime
    from midrun.runtime.state.rank_info import RankInfo


def _rank_field_values(rank: "RankInfo") -> dict[str, str]:
    return {
        "pid": str(rank.pid),
        "hostname": rank.hostname,
        "global-rank": str(rank.global_rank),
        "local-rank": str(rank.local_rank),
        "device": rank.device,
    }


def rank_file_content(runtime: "MidrunRuntime", path: str) -> bytes | None:
    """Returns the content of a `/meta/ranks/<idx>/<field>` file, or `None` for another path."""
    parts = path.split("/")
    if (
        len(parts) != 5
        or parts[1] != "meta"
        or parts[2] != "ranks"
        or parts[4] not in RANK_FIELDS
    ):
        return None
    rank = resolve_rank(parts, runtime.state.read().ranks)
    if rank is None:
        return None
    return encode_text(_rank_field_values(rank)[parts[4]])
