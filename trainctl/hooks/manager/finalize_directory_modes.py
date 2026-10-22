"""Gives the copied directories the ownership and mode of their sources."""

from pathlib import Path

from trainctl.hooks.manager.apply_ownership_and_mode import apply_ownership_and_mode


def finalize_directory_modes(dir_records: list[tuple[Path, int, int, int]]) -> None:
    """Applies each recorded directory's preserved ownership/mode, deepest first."""
    for dst_path, uid, gid, mode in dir_records:
        apply_ownership_and_mode(dst_path, uid, gid, mode)
