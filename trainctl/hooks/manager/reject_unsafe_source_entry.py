"""Refuses a baseline entry that the hooks bootstrap must not copy."""

import os
import stat as stat_module
from pathlib import Path

from trainctl.hooks.manager.hooks_bootstrap_error import HooksBootstrapError


def reject_unsafe_source_entry(path: Path, st: os.stat_result) -> None:
    """Raises if `path` is a symlink, a special file, or has a setuid/setgid/sticky bit.

    Args:
        path: The baseline entry, for the error message.
        st: The `lstat` result of `path`.

    Raises:
        HooksBootstrapError: The entry is unsafe to copy.
    """
    if stat_module.S_ISLNK(st.st_mode):
        raise HooksBootstrapError(
            f"hooks_source_dir contains a symlink, which is not supported: {path}"
        )
    if not (stat_module.S_ISREG(st.st_mode) or stat_module.S_ISDIR(st.st_mode)):
        raise HooksBootstrapError(
            f"hooks_source_dir contains a non-regular file, which is not supported: {path}"
        )
    if st.st_mode & (stat_module.S_ISUID | stat_module.S_ISGID | stat_module.S_ISVTX):
        raise HooksBootstrapError(
            f"hooks_source_dir contains privileged mode bits, which is not supported: {path}"
        )
