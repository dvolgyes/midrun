"""List published checkpoint files and directories for `GET /checkpoints`."""

from pathlib import Path
from typing import Any


def checkpoint_entries(checkpoints_dir: Path) -> list[dict[str, Any]]:
    """Lists `.ckpt` files and directories in `checkpoints_dir`, sorted by name.

    A checkpoint appears only after the save succeeds on all ranks. If an
    operator removes a checkpoint during the listing, it is left out.

    Args:
        checkpoints_dir: The artifact store's `checkpoints/` directory; a missing
            directory has no entries.

    Returns:
        One entry per checkpoint: its name, modification time, `size_bytes`, and
        `url` under `/artifact-files`. A file URL downloads that file. A directory
        URL is a base for known shard filenames, not an archive download.
        `size_bytes` is the filesystem-reported size of the entry, not the sum
        of a directory's contents.
    """
    if not checkpoints_dir.is_dir():
        return []
    entries = []
    for path in sorted(checkpoints_dir.glob("*.ckpt")):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        entries.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
                "url": f"/artifact-files/checkpoints/{path.name}",
            }
        )
    return entries
