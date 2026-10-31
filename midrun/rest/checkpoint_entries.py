"""The entries of `GET /checkpoints`: each saved checkpoint with its download URL."""

from pathlib import Path
from typing import Any


def checkpoint_entries(checkpoints_dir: Path) -> list[dict[str, Any]]:
    """Lists the `.ckpt` files in `checkpoints_dir`, sorted by name.

    A checkpoint that is being saved shows its partial size. A file that a failed
    save, or an operator, removes during the listing is left out.

    Args:
        checkpoints_dir: The artifact store's `checkpoints/` directory; a missing
            directory has no entries.

    Returns:
        One entry per file: its name, size, modification time and the URL under
        `/artifact-files` that downloads it.
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
