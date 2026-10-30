"""The content of a file in a meta/ranks/<idx>/workers/<widx>/ directory."""

from typing import TYPE_CHECKING

from midrun.runtime.virtual_fs.encode_text import encode_text
from midrun.runtime.virtual_fs.parse_rank_path import parse_rank_path

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def worker_file_content(runtime: "MidrunRuntime", path: str) -> bytes | None:
    """Returns the content of a `/meta/ranks/<idx>/workers/<widx>/<field>` file, or `None`."""
    parsed = parse_rank_path(path, runtime.state.read().ranks)
    if (
        parsed is None
        or parsed.kind != "worker_field"
        or parsed.rank is None
        or parsed.worker_index is None
    ):
        return None
    return encode_text(str(parsed.rank.worker_pids[parsed.worker_index]))
