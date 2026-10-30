"""Creates the private directory of one hook invocation's payload."""

from pathlib import Path


def create_invocation_dir(tmp_root: Path, invocation_id: str) -> Path:
    """Creates `tmp_root/invocation_id`, with any missing ancestors.

    `tmp_root` (a session's own directory) and its parent (the run's shared
    `.hooks-tmp/`) are created with mode `0o700` if absent; the new invocation
    directory always is.

    Returns:
        The new invocation directory.

    Raises:
        OSError: A directory cannot be created, for example on a full disk.
    """
    tmp_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp_root.mkdir(mode=0o700, exist_ok=True)
    invocation_dir = tmp_root / invocation_id
    invocation_dir.mkdir(mode=0o700)
    return invocation_dir
