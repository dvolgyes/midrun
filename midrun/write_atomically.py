"""Replaces a file with new content in one atomic step."""

import secrets
from pathlib import Path


def write_atomically(path: Path, data: bytes) -> None:
    """Writes `data` to `path` so that a reader never sees a partial file.

    The data goes to a new temporary file in the same directory, which then
    replaces `path`. A concurrent reader sees no file, the previous file, or the
    complete new file. The temporary name starts with a dot and ends in `.tmp`;
    it is removed when the write fails, but stays when the process dies during
    the write.
    """
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    replaced = False
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
        temporary.replace(path)
        replaced = True
    finally:
        if not replaced:
            temporary.unlink(missing_ok=True)
