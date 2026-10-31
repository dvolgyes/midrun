"""A rate-limited scan of the hook directories for unknown filenames.

The scan is a coarse-grained diagnostic, separate from the per-use enable check in
`midrun.hooks.resolve_enabled_script`: it costs an `os.scandir` per modality and
is rate-limited to once per logical epoch token so a high-frequency callback (e.g.
`on_train_batch_start`) doesn't pay for it every invocation.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from midrun.hooks.catalog import CALLBACK_NAMES, MODALITIES

if TYPE_CHECKING:
    from loguru import Logger


class EpochScanState:  # pylint: disable=too-few-public-methods
    """Rate-limits the unknown-filename scan to once per distinct epoch token.

    Attributes:
        last_epoch_token: The most recently scanned token, or `None` before the
            first scan. Callers pass `(trainer.state.fn, trainer.current_epoch)`.
    """

    def __init__(self) -> None:
        self.last_epoch_token: tuple[Any, ...] | None = None
        self._warned_paths: set[str] = set()

    def scan_if_due(
        self, live_dir: Path, epoch_token: tuple[Any, ...], log: "Logger"
    ) -> None:
        """Scans `light/` and `heavy/` once per distinct `epoch_token`.

        Warns once (ever, per relative path) about a file directly under a modality
        directory whose name does not match a known callback -- the same diagnostic
        Increment I1 gives at bootstrap time, extended to files an operator adds
        after bootstrap. Never raises: an unreadable modality directory is skipped.
        """
        if epoch_token == self.last_epoch_token:
            return
        self.last_epoch_token = epoch_token
        for modality in MODALITIES:
            self._scan_modality(live_dir / modality, modality, log)

    def _scan_modality(self, modality_dir: Path, modality: str, log: "Logger") -> None:
        try:
            entries = os.scandir(modality_dir)
        except OSError:
            return
        with entries:
            for entry in entries:
                self._check_entry(entry, modality, log)

    def _check_entry(
        self, entry: os.DirEntry[str], modality: str, log: "Logger"
    ) -> None:
        if not entry.is_file(follow_symlinks=False):
            return
        stem = entry.name[: -len(".sh")] if entry.name.endswith(".sh") else ""
        if stem in CALLBACK_NAMES:
            return
        relative = f"{modality}/{entry.name}"
        if relative in self._warned_paths:
            return
        self._warned_paths.add(relative)
        log.warning(
            "hooks: {} does not match a known callback name and will never be "
            "dispatched",
            relative,
        )
