"""Gives the copied directories the ownership and mode of their sources."""

from pathlib import Path

from midrun.hooks.manager.apply_ownership_and_mode import apply_ownership_and_mode


def finalize_directory_modes(
    dir_records: list[tuple[Path, int, int, int]],
) -> list[Path]:
    """Applies each recorded directory's preserved ownership/mode, deepest first.

    Returns:
        The directories whose ownership could not be preserved.
    """
    return [
        dst_path
        for dst_path, uid, gid, mode in dir_records
        if not apply_ownership_and_mode(dst_path, uid, gid, mode)
    ]
