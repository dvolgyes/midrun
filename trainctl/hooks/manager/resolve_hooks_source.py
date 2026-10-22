"""Resolves a hooks baseline directory and checks that it is safe to copy."""

import os
from pathlib import Path

from trainctl.hooks.manager.hooks_bootstrap_error import HooksBootstrapError
from trainctl.hooks.manager.reject_unsafe_source_entry import reject_unsafe_source_entry


def resolve_hooks_source(source_dir: Path, base_dir: Path) -> Path:
    """Returns the resolved `source_dir` after it passes the safety checks.

    Raises:
        HooksBootstrapError: `source_dir` is unsafe, is not a directory, or
            equals, contains or is inside `base_dir`.
        FileNotFoundError: `source_dir` does not exist.
    """
    resolved_source = source_dir.resolve(strict=True)
    resolved_base = base_dir.resolve()
    reject_unsafe_source_entry(resolved_source, os.lstat(resolved_source))
    if not resolved_source.is_dir():
        raise HooksBootstrapError(f"hooks_source_dir is not a directory: {source_dir}")
    if (
        resolved_source == resolved_base
        or resolved_source.is_relative_to(resolved_base)
        or resolved_base.is_relative_to(resolved_source)
    ):
        raise HooksBootstrapError(
            f"hooks_source_dir {source_dir} must not equal, contain, or be contained by the run "
            f"directory {base_dir}"
        )
    return resolved_source
