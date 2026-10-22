"""Selects the default directory for the FUSE mount and the artifact store."""

import os
from pathlib import Path
from typing import Any


def default_base_dir(pl_module: Any, run_id: str) -> Path:
    """Resolves the default directory for Trainctl's FUSE mount and artifact store.

    Colocates under the Lightning run's own `trainer.log_dir` (alongside checkpoints
    and logger output) whenever a Trainer is attached and reports one; falls back to a
    per-uid runtime/tmp directory, scoped by run id, when no Trainer is attached yet or
    no log directory is available.
    """
    trainer = getattr(pl_module, "trainer", None)
    log_dir = getattr(trainer, "log_dir", None) if trainer is not None else None
    if log_dir:
        return Path(log_dir)
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(xdg) / "trainctl" if xdg else Path(f"/tmp/trainctl-{os.getuid()}")  # noqa: S108 -- per-uid tmp root
    return root / run_id
