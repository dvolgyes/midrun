"""One declared tunable-hparam entry with its bounds and label."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TunableHparamSpec:
    """One collected+normalized `trainctl_tunable_hparams` entry.

    Attributes:
        path: Dotted resolution path -- identical to this spec's key in
            `collect_tunable_hparams`'s return dict.
        min: Optional inclusive lower bound `validate_bounds` enforces. `None` is
            unbounded below; always `None` for a bare `list[str]`-form entry.
        max: Optional inclusive upper bound, same semantics.
        label: Optional human-readable display name; `None` falls back to `path`.
    """

    path: str
    min: int | float | None = None
    max: int | float | None = None
    label: str | None = None
