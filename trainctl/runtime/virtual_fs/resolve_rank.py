"""Finds the rank that a meta/ranks path names."""

from typing import TYPE_CHECKING

from trainctl.runtime.virtual_fs.listed_index import listed_index

if TYPE_CHECKING:
    from trainctl.runtime.state.rank_info import RankInfo


def resolve_rank(parts: list[str], ranks: tuple["RankInfo", ...]) -> "RankInfo | None":
    """Returns the rank that `parts[3]` names, or `None` if it names no rank."""
    index = listed_index(parts, 3, len(ranks))
    return ranks[index] if index is not None else None
