"""Rank identity and dispatch eligibility for a hook session."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class HookRank:
    """Describe which rank owns the session and whether it can launch hooks.

    :ivar rank: Global rank of the training process.
    :ivar world_size: Number of ranks in the training job.
    :ivar policy: Configured rank selection policy.
    """

    rank: int
    world_size: int
    policy: str

    @property
    def eligible(self) -> bool:
        """Whether this rank is enabled for dispatch."""
        return self.policy != "rank_zero" or self.rank == 0
