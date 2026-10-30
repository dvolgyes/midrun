"""Parses the index in a rank or worker directory name."""


def listed_index(parts: list[str], pos: int, count: int) -> int | None:
    """Returns the index below `count` that `parts[pos]` names as `list_virtual_dir` spells it.

    Other spellings of the same number (`0`, `-0`, `0_0`) name no path.
    """
    if pos >= len(parts):
        return None
    names = {f"{index:03d}": index for index in range(count)}
    return names.get(parts[pos])
