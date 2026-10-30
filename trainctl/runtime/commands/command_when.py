"""The normalized scheduling condition of a command."""

from dataclasses import dataclass

from trainctl.runtime.events import SafePoint


@dataclass(frozen=True)
class CommandWhen:
    """Normalized scheduling condition for a command.

    Attributes:
        safe_point: The safe point this condition is scored against, or None to
            match any safe point (always None for `immediate` and `next_any`).
        occurrence: `immediate` (due as soon as any safe point, including inside an
            active hold, is reached), `next_any` (same as immediate but outside a hold),
            `next` (next occurrence of `safe_point`), `epoch` (next `safe_point` at or
            after `epoch`), or `step` (next `safe_point` at or after `step`).
        epoch: Target epoch for `occurrence == "epoch"`.
        step: Target global step for `occurrence == "step"`.
    """

    safe_point: SafePoint | None
    occurrence: str
    epoch: int | None = None
    step: int | None = None

    def is_due(
        self, safe_point: SafePoint | None, epoch: int, global_step: int
    ) -> bool:
        """Decides whether this condition is satisfied at the given point in training."""
        if self.occurrence == "immediate":
            return True
        # A hold checks due-ness without a safe point.
        if safe_point is None:
            return False
        if self.occurrence == "next_any":
            return True
        if self.occurrence == "next":
            return safe_point == self.safe_point
        at_safe_point = self.safe_point is None or safe_point == self.safe_point
        if self.occurrence == "epoch":
            return at_safe_point and self.epoch is not None and epoch >= self.epoch
        if self.occurrence == "step":
            return at_safe_point and self.step is not None and global_step >= self.step
        raise ValueError(f"unknown occurrence {self.occurrence!r}")
