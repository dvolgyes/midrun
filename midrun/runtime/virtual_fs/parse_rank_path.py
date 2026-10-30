"""Parses a path in the meta/ranks subtree; the one grammar FUSE and REST share."""

from midrun.runtime.state.rank_info import RankInfo
from midrun.runtime.virtual_fs.listed_index import listed_index
from midrun.runtime.virtual_fs.rank_path import RankPath
from midrun.runtime.virtual_fs.rank_path_fields import RANK_FIELDS, WORKER_FIELDS


def parse_rank_path(path: str, ranks: tuple[RankInfo, ...]) -> RankPath | None:
    """Returns what `path` names under `/meta/ranks`, or `None`.

    One trailing slash is accepted on a directory path. The result is `None` for a
    path outside the subtree, for a rank, worker or field that does not exist, and
    for a file path with a trailing slash.
    """
    trailing_slash = path.endswith("/")
    parts = path.removesuffix("/").split("/")
    if parts[:3] != ["", "meta", "ranks"]:
        return None
    if len(parts) == 3:
        return RankPath("ranks")
    rank_index = listed_index(parts, 3, len(ranks))
    if rank_index is None:
        return None
    rank = ranks[rank_index]
    parsed: RankPath | None = None
    match parts[4:]:
        case []:
            parsed = RankPath("rank", rank)
        case ["workers"]:
            parsed = RankPath("workers", rank)
        case [field] if field in RANK_FIELDS:
            parsed = RankPath("rank_field", rank, field=field)
        case ["workers", _, *rest]:
            worker_index = listed_index(parts, 5, len(rank.worker_pids))
            if worker_index is None:
                return None
            if not rest:
                parsed = RankPath("worker", rank, worker_index)
            elif len(rest) == 1 and rest[0] in WORKER_FIELDS:
                parsed = RankPath("worker_field", rank, worker_index, rest[0])
    if parsed is None or (trailing_slash and not parsed.is_dir):
        return None
    return parsed
