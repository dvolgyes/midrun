"""The bookkeeping of one bounded argument traversal."""

from dataclasses import dataclass, field


@dataclass
class TraversalState:
    """Mutable bookkeeping threaded through one `describe_arguments` call."""

    max_items: int
    max_depth: int
    seen: set[int] = field(default_factory=set)
    truncated: bool = False
    reasons: set[str] = field(default_factory=set)
