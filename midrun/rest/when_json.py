"""The JSON form of a command's `when` in REST responses."""

from midrun.runtime.commands.command_when import CommandWhen


def when_json(when: CommandWhen) -> str | dict[str, str | int]:
    """Returns `when` in the shape that `parse_when` accepts, so a client can resubmit it."""
    if when.occurrence == "immediate":
        return "now"
    if when.occurrence == "next_any":
        return "next_safe_point"
    shown: dict[str, str | int] = {}
    if when.safe_point is not None:
        shown["safe_point"] = when.safe_point.value
    if when.occurrence == "epoch" and when.epoch is not None:
        shown["epoch"] = when.epoch
    elif when.occurrence == "step" and when.step is not None:
        shown["global_step"] = when.step
    elif when.occurrence != "next":
        raise ValueError(f"cannot show 'when' with occurrence {when.occurrence!r}")
    return shown
