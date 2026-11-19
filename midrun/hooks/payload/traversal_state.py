"""The bookkeeping of one bounded argument traversal."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

# `max_items` and `max_depth` alone allow max_items**max_depth nodes, which shared
# references reach without that much memory, so the whole traversal has a budget.
_NODES_PER_ITEM = 10


@dataclass
class TraversalState:
    """Mutable bookkeeping threaded through one `describe_arguments` call.

    `nodes_left` starts at 10 x `max_items` and counts the values visited; at zero
    the traversal stops and records the `max_nodes` reason.

    :ivar max_items: Maximum entries visited per container.
    :ivar max_depth: Maximum recursive container depth.
    :ivar seen: Container identities on the active traversal path.
    :ivar truncated: Whether a configured bound was reached.
    :ivar reasons: Names of bounds reached during traversal.
    :ivar nodes_left: Remaining visits in the traversal-wide budget.
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

    def bounded_items[K, V](
        self, items: Iterable[tuple[K, V]], total: int
    ) -> Iterator[tuple[K, V]]:
        """Spend one node before visiting each item; never fetch an omitted item."""
        iterator = iter(items)
        for index in range(total):
            if index >= self.max_items or self.nodes_left <= 0:
                self.truncated = True
                self.reasons.add(
                    "max_items" if index >= self.max_items else "max_nodes"
                )
                break
            item = next(iterator, None)
            if item is None:
                break
            self.nodes_left -= 1
            yield item
