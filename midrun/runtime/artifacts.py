"""Disk artifact layout for checkpoints, snapshots, and debug captures.

Artifacts are written by the training thread (via command handlers) as real files
under `root`, a sibling of the FUSE mount that the mount does not project (its
`meta/artifacts-path` names `root`); REST serves them at `/artifact-files/`. This
module owns only the on-disk layout and manifest format.
"""

import ctypes
import errno
import os
import re
import shutil
import stat
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from midrun.encode_json import encode_json
from midrun.runtime.commands.next_id import next_id

# Allowed checkpoint labels and capture filenames: one plain path component.
ARTIFACT_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"

_ARTIFACT_NAME = re.compile(ARTIFACT_NAME_PATTERN)
_RESERVED_CAPTURE_FILENAMES = frozenset({"manifest.json"})
# Ids already taken by earlier runs or other ranks that one new artifact skips at most.
_MAX_ID_ATTEMPTS = 100_000
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _publish_checkpoint(staged: Path, path: Path) -> None:
    """Publishes with a file hard link or a no-replace directory rename."""
    if not stat.S_ISDIR(staged.lstat().st_mode):
        os.link(staged, path, follow_symlinks=False)
        return
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS, "checkpoint directory publication requires renameat2", path
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        _AT_FDCWD, os.fsencode(staged), _AT_FDCWD, os.fsencode(path), _RENAME_NOREPLACE
    ):
        error = ctypes.get_errno()
        if error in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            raise OSError(
                error,
                "checkpoint directory publication requires a no-replace rename",
                path,
            )
        raise OSError(error, os.strerror(error), path)


class ArtifactStore:
    """Materializes checkpoints, snapshots, and debug captures under `root`.

    :ivar root: Base artifact directory for this run.
    """

    root: Path

    def __init__(self, root: Path) -> None:
        self.root = root
        (self.root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.root / "snapshots").mkdir(parents=True, exist_ok=True)
        (self.root / "debug").mkdir(parents=True, exist_ok=True)

    @contextmanager
    def stage_checkpoint(self, label: str) -> Iterator[tuple[Path, Path]]:
        """Yields private and public paths; publishes only on successful exit.

        Runs sharing `root` reserve each name by exclusively creating its staging
        directory. The private save path itself does not exist yet: a strategy
        can create a file or a sharded checkpoint directory at that path.
        """
        if not _ARTIFACT_NAME.fullmatch(label):
            raise ValueError(
                f"invalid checkpoint label {label!r}: must match {ARTIFACT_NAME_PATTERN}"
            )
        checkpoints = self.root / "checkpoints"
        if self.root.stat().st_dev != checkpoints.stat().st_dev:
            raise OSError(
                "checkpoint staging and publication require the same filesystem"
            )
        for _ in range(_MAX_ID_ATTEMPTS):
            path = checkpoints / f"ckpt-{next_id('c')}-{label}.ckpt"
            stage = self.root / f".midrun-stage-{path.stem}"
            try:
                stage.mkdir(mode=0o700)
            except FileExistsError:
                continue
            try:
                if path.exists(follow_symlinks=False):
                    continue
                staged = stage / path.name
                yield staged, path
                _publish_checkpoint(staged, path)
                return
            finally:
                shutil.rmtree(stage)
        raise FileExistsError(
            f"no free checkpoint id under {checkpoints} after {_MAX_ID_ATTEMPTS} attempts"
        )

    def write_snapshot(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Writes a snapshot's manifest and payload; returns the manifest.

        Args:
            kind: Snapshot kind, e.g. `runtime`, `model_summary`, `optimizer_summary`.
            payload: JSON-serializable snapshot content.

        Returns:
            The written manifest (id, kind, created_at, and payload path).
        """
        if not _ARTIFACT_NAME.fullmatch(kind) or kind == "manifest":
            raise ValueError(f"invalid snapshot kind {kind!r}")
        with _stage_artifact(self.root / "snapshots", "snap") as (stage, published):
            manifest = {"id": published.name, "kind": kind, "created_at": time.time()}
            (stage / f"{kind}.json").write_bytes(encode_json(payload))
            (stage / "manifest.json").write_bytes(encode_json(manifest))
        manifest["path"] = str(published)
        return manifest

    def write_debug_capture(
        self,
        kind: str,
        files: Mapping[str, str | bytes],
        metadata: dict[str, Any] | None = None,
        *,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """Writes a debug capture's manifest plus one or more named files; returns the manifest.

        Args:
            kind: Debug capture kind, e.g. `pyspy-stack`, `gdb-snapshot`, `batch-tensor`.
            files: Mapping of bare filename (no path separators) to content. A `str`
                value is written as text; a `bytes` value (e.g. a `torch.save` buffer)
                is written as-is.
            metadata: Optional JSON-serializable context to record alongside the capture.
            artifact_id: Optional preassigned id for a capture advertised before publication.

        Returns:
            The written manifest (id, kind, created_at, metadata, filenames, and path).

        Raises:
            ValueError: a filename in `files` does not match `ARTIFACT_NAME_PATTERN`
                or is `manifest.json` (REST-supplied input boundary: a capture can
                never write outside its own directory or replace its manifest), or
                `artifact_id` is not a valid debug id. Nothing is written.
            FileExistsError: the preassigned id is already in use.
        """
        if artifact_id is not None and (
            not artifact_id.startswith("dbg_")
            or not _ARTIFACT_NAME.fullmatch(artifact_id)
        ):
            raise ValueError(f"invalid debug capture id {artifact_id!r}")
        for filename in files:
            if (
                not _ARTIFACT_NAME.fullmatch(filename)
                or filename in _RESERVED_CAPTURE_FILENAMES
            ):
                raise ValueError(
                    f"invalid capture filename {filename!r}: must match"
                    f" {ARTIFACT_NAME_PATTERN} and not be manifest.json"
                )
        with _stage_artifact(self.root / "debug", "dbg", artifact_id=artifact_id) as (
            stage,
            published,
        ):
            manifest = {
                "id": published.name,
                "kind": kind,
                "created_at": time.time(),
                "metadata": metadata or {},
                "files": sorted(files),
            }
            for filename, content in files.items():
                if isinstance(content, bytes):
                    (stage / filename).write_bytes(content)
                else:
                    (stage / filename).write_text(content)
            (stage / "manifest.json").write_bytes(encode_json(manifest))
        manifest["path"] = str(published)
        return manifest


@contextmanager
def _stage_artifact(
    parent: Path, prefix: str, *, artifact_id: str | None = None
) -> Iterator[tuple[Path, Path]]:
    """Reserves a private directory outside the published categories until all writes succeed."""
    staging_parent = parent.parent
    if staging_parent.stat().st_dev != parent.stat().st_dev:
        raise OSError("artifact staging and publication require the same filesystem")
    for _ in range(1 if artifact_id is not None else _MAX_ID_ATTEMPTS):
        candidate = artifact_id if artifact_id is not None else next_id(prefix)
        destination = parent / candidate
        stage = staging_parent / f".midrun-stage-{candidate}"
        try:
            stage.mkdir(mode=0o700)
        except FileExistsError:
            continue
        try:
            if not destination.exists():
                yield stage, destination
                stage.rename(destination)
                return
        except BaseException:
            shutil.rmtree(stage)
            raise
        stage.rmdir()
    if artifact_id is not None:
        raise FileExistsError(f"artifact id {artifact_id!r} is in use under {parent}")
    raise FileExistsError(
        f"no free {prefix} id under {parent} after {_MAX_ID_ATTEMPTS} attempts"
    )
