"""Builds a new hooks tree in a staging directory and publishes it atomically."""

import errno
import os
import shutil
import socket
import stat as stat_module
import tempfile
from pathlib import Path
from typing import Any

from trainctl.hooks.catalog import CATALOGUE_VERSION
from trainctl.hooks.manager.copy_source_tree import copy_source_tree
from trainctl.hooks.manager.finalize_directory_modes import finalize_directory_modes
from trainctl.hooks.manager.hooks_bootstrap_error import HooksBootstrapError
from trainctl.hooks.manager.hooks_tree_layout import (
    DEFAULT_DIR_MODE,
    MARKER_NAME,
    PROVENANCE_NAME,
    SCHEMA_VERSION,
)
from trainctl.hooks.manager.seed_missing_slots import seed_missing_slots
from trainctl.hooks.manager.write_readme_if_absent import write_readme_if_absent
from trainctl.hooks.write_json_file import write_json_file


def publish_fresh_tree(
    base_dir: Path,
    live_dir: Path,
    resolved_source: Path | None,
    session_id: str,
    log: Any,
) -> bool:
    """Builds a new hooks tree in a staging directory, then renames it to `live_dir`.

    Copies the baseline (if any), seeds the missing slots and the README,
    writes the provenance record, and writes the marker after the rename. The
    staging directory is removed if the publish does not complete.

    Returns:
        `True` if this call published the tree; `False` if a non-empty `live_dir`
        appeared first (e.g. a concurrent initializer won the race), in which case
        nothing is written.

    Raises:
        HooksBootstrapError: The copy or the rename failed.
        PermissionError: The ownership of a copied entry cannot be preserved.
    """
    marker_path = base_dir / MARKER_NAME
    staging_dir = Path(tempfile.mkdtemp(dir=base_dir, prefix=".hooks-staging-"))
    staging_dir.chmod(DEFAULT_DIR_MODE)
    published = False
    try:
        (staging_dir / "light").mkdir(mode=DEFAULT_DIR_MODE, exist_ok=True)
        (staging_dir / "heavy").mkdir(mode=DEFAULT_DIR_MODE, exist_ok=True)

        found_slots: set[tuple[str, str]] = set()
        unknown_paths: list[str] = []
        dir_records: list[tuple[Path, int, int, int]] = []

        if resolved_source is not None:
            copy_source_tree(
                resolved_source,
                staging_dir,
                rel=(),
                found_slots=found_slots,
                unknown_paths=unknown_paths,
                dir_records=dir_records,
            )
            top_st = os.lstat(resolved_source)
            dir_records.append(
                (
                    staging_dir,
                    top_st.st_uid,
                    top_st.st_gid,
                    stat_module.S_IMODE(top_st.st_mode),
                )
            )
        if unknown_paths:
            log.warning(
                "hooks_source_dir has {} file(s) directly under light/ or heavy/ that do not "
                "match a known callback name and will never be dispatched: {}",
                len(unknown_paths),
                ", ".join(sorted(unknown_paths)),
            )

        seed_missing_slots(staging_dir, found_slots)
        write_readme_if_absent(staging_dir)

        record = {
            "schema_version": SCHEMA_VERSION,
            "catalogue_version": CATALOGUE_VERSION,
            "session_id": session_id,
            "source_dir": str(resolved_source) if resolved_source is not None else None,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        }
        write_json_file(staging_dir / PROVENANCE_NAME, record)

        finalize_directory_modes(dir_records)

        try:
            staging_dir.rename(live_dir)
            published = True
        except OSError as exc:
            if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
                return False
            raise HooksBootstrapError(
                f"cannot publish {staging_dir} to {live_dir}: {exc}"
            ) from exc
    finally:
        if not published:
            shutil.rmtree(staging_dir, ignore_errors=True)

    write_json_file(marker_path, record)
    log.info(
        "hooks: bootstrapped live tree at {} (source={})", live_dir, resolved_source
    )
    return True
