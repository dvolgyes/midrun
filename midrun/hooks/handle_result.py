"""Combined light and heavy execution results of one callback occurrence."""

from dataclasses import dataclass

from midrun.hooks.runner import ExecutionResult


@dataclass(frozen=True)
class HandleResult:
    """Both modalities' outcomes for one dispatched callback occurrence.

    :ivar light: Light hook outcome, or None when disabled.
    :ivar heavy: Heavy hook outcome, or None when disabled.
    """

    light: ExecutionResult | None
    heavy: ExecutionResult | None
