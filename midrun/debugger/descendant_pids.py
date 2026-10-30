"""Finds the live descendant processes of a process."""

from pathlib import Path


def descendant_pids(pid: int) -> list[int]:
    """Returns the pids of the live descendants of `pid`, each parent before its children.

    Reads `/proc/<pid>/task/<tid>/children` (Linux with `CONFIG_PROC_CHILDREN`), so
    the children of every thread count. A process that exits during the walk is
    skipped together with its descendants; a missing `pid` has none.
    """
    found: list[int] = []
    pending = [pid]
    while pending:
        parent = pending.pop(0)
        for children_file in sorted(Path(f"/proc/{parent}/task").glob("*/children")):
            try:
                children = [int(child) for child in children_file.read_text().split()]
            except (FileNotFoundError, ProcessLookupError):
                continue
            new = [child for child in children if child not in found]
            found.extend(new)
            pending.extend(new)
    return found
