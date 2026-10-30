"""Parses the REST `when` value of a command."""

from midrun.runtime.commands.command_when import CommandWhen
from midrun.runtime.events import SafePoint


def parse_when(raw: object) -> CommandWhen:
    """Parses a REST-supplied `when` value into a `CommandWhen`.

    Accepts the shapes documented in the Midrun REST API: the strings `"now"` and
    `"next_safe_point"`, or a mapping with at most one of `epoch` and `global_step`
    (non-negative integers) and an optional `safe_point`. Without `epoch` or
    `global_step`, the mapping needs `safe_point`, and the only accepted
    `occurrence` is `"next"`. Validation is complete here, so that a queued command
    can never fail its due-ness check on the training thread.

    Args:
        raw: The decoded JSON value of the `when`/`execution.when` field.

    Returns:
        The normalized scheduling condition.

    Raises:
        ValueError: `raw` does not match a supported shape.
    """
    if raw in ("now", None):
        return CommandWhen(safe_point=None, occurrence="immediate")
    if raw == "next_safe_point":
        return CommandWhen(safe_point=None, occurrence="next_any")
    if not isinstance(raw, dict):
        raise ValueError(f"invalid 'when' spec: {raw!r}")
    unknown = set(raw) - _WHEN_KEYS
    if unknown:
        raise ValueError(f"unknown 'when' keys: {sorted(unknown)}")
    safe_point = _parse_safe_point(raw.get("safe_point"))
    targets = [key for key in ("epoch", "global_step") if key in raw]
    if len(targets) > 1:
        raise ValueError("'when' accepts only one of 'epoch' and 'global_step'")
    if targets and "occurrence" in raw:
        raise ValueError(f"'when' accepts 'occurrence' only without {targets[0]!r}")
    if "epoch" in raw:
        epoch = _parse_non_negative_int("epoch", raw["epoch"])
        return CommandWhen(safe_point=safe_point, occurrence="epoch", epoch=epoch)
    if "global_step" in raw:
        step = _parse_non_negative_int("global_step", raw["global_step"])
        return CommandWhen(safe_point=safe_point, occurrence="step", step=step)
    if safe_point is None:
        raise ValueError("'when' needs 'safe_point', 'epoch' or 'global_step'")
    occurrence = raw.get("occurrence", "next")
    if occurrence != "next":
        raise ValueError(
            f"unsupported 'occurrence' {occurrence!r}; only 'next' is valid"
        )
    return CommandWhen(safe_point=safe_point, occurrence="next")


_WHEN_KEYS = frozenset({"safe_point", "occurrence", "epoch", "global_step"})


def _parse_safe_point(raw: object) -> SafePoint | None:
    if raw is None:
        return None
    valid = [point.value for point in SafePoint]
    if not isinstance(raw, str) or raw not in valid:
        raise ValueError(f"invalid 'safe_point' {raw!r}; expected one of {valid}")
    return SafePoint(raw)


def _parse_non_negative_int(key: str, raw: object) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"'when.{key}' must be a non-negative integer, got {raw!r}")
    return raw
