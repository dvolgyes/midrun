"""Selects the default directory for the FUSE mount and the artifact store."""

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule

_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]")


def default_base_dir(pl_module: "LightningModule", run_id: str) -> Path:
    """Resolves the default directory for Midrun's run files.

    The directory holds `hooks/`, `procfs/`, `artifacts/`, `midrun.log` and the hooks
    state marker. Lightning makes `trainer.log_dir` per version only for
    `TensorBoardLogger` and `CSVLogger`; Midrun then uses that directory, next to the
    checkpoints and the logger output. For every other logger `trainer.log_dir` is the
    logger's `save_dir`, which all runs share, and without a logger it is
    `default_root_dir`; Midrun then uses `<that directory>/midrun/<id>`. `<id>` is the
    logger's version, reduced to one safe path component, or `run_id` when the logger
    has no usable version. Without an attached Trainer, the directory is a per-uid
    runtime/tmp directory, scoped by run id.

    `trainer.log_dir` and the final choice are both collectives under DDP, so every
    rank must call this at the same point. All ranks get rank zero's directory, even
    though each process has its own `run_id`.
    """
    trainer = getattr(pl_module, "trainer", None)
    if trainer is None:
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        tmp_root = f"/tmp/midrun-{os.getuid()}"  # noqa: S108 -- per-uid tmp root
        return (Path(xdg) / "midrun" if xdg else Path(tmp_root)) / run_id
    log_dir = trainer.log_dir
    logger = trainer.loggers[0] if trainer.loggers else None
    save_dir = getattr(logger, "save_dir", None)
    if logger is not None and log_dir and save_dir and Path(log_dir) != Path(save_dir):
        candidate = Path(log_dir)
    else:
        version = getattr(logger, "version", None)
        # The version is untrusted text: keep one path component without a leading dot.
        safe = (
            ""
            if version is None
            else _UNSAFE_CHARACTERS.sub("_", str(version)).lstrip(".")
        )
        candidate = (
            Path(log_dir or trainer.default_root_dir) / "midrun" / (safe or run_id)
        )
    return Path(trainer.strategy.broadcast(str(candidate)))
