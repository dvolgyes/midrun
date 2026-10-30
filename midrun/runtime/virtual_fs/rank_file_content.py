"""The content of a file in a meta/ranks/<idx>/ directory."""

from typing import TYPE_CHECKING

from midrun.runtime.virtual_fs.encode_text import encode_text
from midrun.runtime.virtual_fs.parse_rank_path import parse_rank_path

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
    parsed = parse_rank_path(path, runtime.state.read().ranks)
    if (
        parsed is None
        or parsed.kind != "rank_field"
        or parsed.rank is None
        or parsed.field is None
    ):
        return None
    return encode_text(_rank_field_values(parsed.rank)[parsed.field])
