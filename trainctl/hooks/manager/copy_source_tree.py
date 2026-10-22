"""Copies a hooks baseline directory into the staging tree."""

import os
import shutil
import stat as stat_module
from pathlib import Path

from trainctl.hooks.catalog import CALLBACK_NAMES, MODALITIES
from trainctl.hooks.manager.apply_ownership_and_mode import apply_ownership_and_mode
from trainctl.hooks.manager.hooks_bootstrap_error import HooksBootstrapError
from trainctl.hooks.manager.hooks_tree_layout import DEFAULT_DIR_MODE
from trainctl.hooks.manager.reject_unsafe_source_entry import reject_unsafe_source_entry


def copy_source_tree(
    src_dir: Path,
    dst_dir: Path,
    *,
    rel: tuple[str, ...],
    found_slots: set[tuple[str, str]],
    unknown_paths: list[str],
    dir_records: list[tuple[Path, int, int, int]],
) -> None:
    """Copies `src_dir` into `dst_dir` recursively, in name order.

    Collects into the keyword arguments: the callback slots that the baseline
    fills, the `light/` and `heavy/` files that match no callback, and each
    copied directory with its source ownership and mode. The caller applies
    the directory ownership and modes after the copy.

    Raises:
        HooksBootstrapError: An entry is unsafe, or a file changed while it was copied.
        PermissionError: The ownership of a copied file cannot be preserved.
    """
    with os.scandir(src_dir) as it:
        entries = sorted(it, key=lambda e: e.name)
    for entry in entries:
        st = os.lstat(entry.path)
        entry_path = Path(entry.path)
        reject_unsafe_source_entry(entry_path, st)
        child_rel = (*rel, entry.name)
        dst_path = dst_dir / entry.name
        if stat_module.S_ISDIR(st.st_mode):
            dst_path.mkdir(mode=DEFAULT_DIR_MODE, exist_ok=True)
            copy_source_tree(
                entry_path,
                dst_path,
                rel=child_rel,
                found_slots=found_slots,
                unknown_paths=unknown_paths,
                dir_records=dir_records,
            )
            dir_records.append(
                (dst_path, st.st_uid, st.st_gid, stat_module.S_IMODE(st.st_mode))
            )
        else:
            shutil.copyfile(entry.path, dst_path)
            post = os.lstat(entry.path)
            if (st.st_size, st.st_mtime_ns) != (post.st_size, post.st_mtime_ns):
                raise HooksBootstrapError(
                    f"hooks_source_dir file {entry.path} changed while being copied; "
                    "do not edit the baseline during initialization"
                )
            apply_ownership_and_mode(
                dst_path, st.st_uid, st.st_gid, stat_module.S_IMODE(st.st_mode)
            )
            if len(child_rel) == 2 and child_rel[0] in MODALITIES:
                stem = child_rel[1][:-3] if child_rel[1].endswith(".sh") else ""
                if stem in CALLBACK_NAMES:
                    found_slots.add((child_rel[0], stem))
                else:
                    unknown_paths.append("/".join(child_rel))
