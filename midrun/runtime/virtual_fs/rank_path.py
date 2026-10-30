"""A parsed path in the meta/ranks subtree."""

from dataclasses import dataclass
from typing import Literal

from midrun.runtime.state.rank_info import RankInfo

RankPathKind = Literal[
    "ranks", "rank", "rank_field", "workers", "worker", "worker_field"
]


@dataclass(frozen=True)
class RankPath:
    """A `meta/ranks` path that names an existing directory or file.

    :ivar kind: `ranks` (`/meta/ranks`), `rank` (`.../NNN`), `rank_field`
        (`.../NNN/<field>`), `workers` (`.../NNN/workers`), `worker`
        (`.../workers/WWW`) or `worker_field` (`.../workers/WWW/<field>`).
    :ivar rank: The rank that the path names; `None` for `ranks`.
    :ivar worker_index: The worker that the path names; `None` above the worker level.
    :ivar field: The file name for a `rank_field` or `worker_field` path.
    """

    kind: RankPathKind
    rank: RankInfo | None = None
    worker_index: int | None = None
    field: str | None = None

    @property
    def is_dir(self) -> bool:
        """Whether the path names a directory, not a file."""
        return self.kind not in ("rank_field", "worker_field")
