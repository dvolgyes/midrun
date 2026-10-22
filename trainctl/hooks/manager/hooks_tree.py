"""A run's live hooks tree and its session id."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HooksTree:
    """A run's live hooks tree and the session id that its marker records.

    Attributes:
        live_dir: The live `hooks/` directory (`base_dir / "hooks"`).
        session_id: The id of the session that published the tree. A resumed run
            keeps it, so its hooks see the same `session_id` as the first one.
    """

    live_dir: Path
    session_id: str
