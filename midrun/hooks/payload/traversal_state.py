"""The bookkeeping of one bounded argument traversal."""

from dataclasses import dataclass, field

# `max_items` and `max_depth` alone allow max_items**max_depth nodes, which shared
# references reach without that much memory, so the whole traversal has a budget.
_NODES_PER_ITEM = 10


@dataclass
class TraversalState:
    """Mutable bookkeeping threaded through one `describe_arguments` call.

    `nodes_left` starts at 10 x `max_items` and counts the values visited; at zero
    the traversal stops and records the `max_nodes` reason.
    """

    max_items: int
    max_depth: int
    seen: set[int] = field(default_factory=set)
    truncated: bool = False
    reasons: set[str] = field(default_factory=set)
    nodes_left: int = field(init=False)

    def __post_init__(self) -> None:
        """Sets the node budget from `max_items`."""
        self.nodes_left = _NODES_PER_ITEM * self.max_items
