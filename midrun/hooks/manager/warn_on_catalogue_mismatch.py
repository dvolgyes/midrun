"""Warns when a reused hooks tree was made for another hook catalogue."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from midrun.hooks.catalog import CATALOGUE_VERSION

if TYPE_CHECKING:
    from loguru import Logger


def warn_on_catalogue_mismatch(
    record: dict[str, Any], record_path: Path, live_dir: Path, log: "Logger"
) -> None:
    """Logs one warning when `record` holds a catalogue version other than this Midrun's.

    Args:
        record: The marker or provenance record of the reused tree.
        record_path: The file that holds `record`, named in the warning.
        live_dir: The reused live hooks tree.
        log: A bound logger (e.g. `LoggingSession.bind()`).
    """
    if record.get("catalogue_version") != CATALOGUE_VERSION:
        log.warning(
            "hooks: the live tree {} has catalogue version {} (recorded in {}), this "
            "Midrun has catalogue version {}; the tree is reused unchanged, so a callback "
            "without a script slot in it is never dispatched",
            live_dir,
            record.get("catalogue_version"),
            record_path,
            CATALOGUE_VERSION,
        )
