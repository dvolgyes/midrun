"""Publishes or resumes the live hooks tree of a run."""

from pathlib import Path
from typing import TYPE_CHECKING

from midrun.hooks.manager.hooks_bootstrap_error import HooksBootstrapError
from midrun.hooks.manager.hooks_tree import HooksTree
from midrun.hooks.manager.hooks_tree_layout import MARKER_NAME, PROVENANCE_NAME
from midrun.hooks.manager.publish_fresh_tree import publish_fresh_tree
from midrun.hooks.manager.read_hooks_record import read_hooks_record
from midrun.hooks.manager.resolve_hooks_source import resolve_hooks_source
from midrun.hooks.manager.warn_on_catalogue_mismatch import warn_on_catalogue_mismatch
from midrun.hooks.write_json_file import write_json_file

if TYPE_CHECKING:
    from loguru import Logger


def bootstrap_hooks_tree(
    base_dir: Path,
    source_dir: Path | None,
    *,
    session_id: str,
    log: "Logger",
) -> HooksTree:
    """Ensures `base_dir/hooks` exists, seeded/copied as needed, and returns it.

    Idempotent: once `base_dir` holds a compatible marker (`midrun-hooks-state.json`), a later
    call returns the existing live tree unchanged, regardless of `source_dir` -- resuming a run
    never re-copies, re-seeds, or restores a deleted script. A resume warns when the
    configured `source_dir` differs from the marker's, and when the catalogue version
    differs from the marker's or, without a marker, from the tree's provenance record.

    Args:
        base_dir: The run directory (typically the Trainer's log directory), made absolute
            here; this function may create it.
        source_dir: Optional baseline directory (`light/`, `heavy/`, and supporting files),
            copied into the live tree once. Never executed in place and never modified.
        session_id: This run's collision-resistant identity, recorded in the marker/provenance
            when this call publishes a fresh tree.
        log: A bound logger (e.g. `LoggingSession.bind()`) used for warnings/info; never a bare
            module-level logger, so callers control attribution.

    Returns:
        The absolute live `hooks/` directory and the session id that the marker records:
        `session_id` for a fresh tree, the earlier session's id for a resumed one.

    Raises:
        HooksBootstrapError: `source_dir` is unsafe relative to `base_dir` (equal, containing, or
            contained by it), contains a symlink/special file/privileged mode bit, changed while
            being copied, or an existing `hooks/` tree is not a compatible resume (unknown-origin
            conflict). A concurrent initializer that wins the publish race is resumed, not an
            error.
    """
    # Scripts run with the live tree as their working directory, so a relative
    # path (e.g. Lightning's default `runs/...` log_dir) would not resolve there.
    base_dir = base_dir.resolve()
    live_dir = base_dir / "hooks"
    marker_path = base_dir / MARKER_NAME

    if live_dir.is_symlink():
        raise HooksBootstrapError(
            f"{live_dir} is a symlink, which is not supported as the live hooks tree"
        )

    if marker_path.exists():
        record = read_hooks_record(marker_path)
        if record is None:
            raise HooksBootstrapError(
                f"{marker_path} exists but is not a readable, compatible Midrun hooks marker; "
                "refusing to reuse or overwrite it. To start the hooks tree again, delete it and "
                f"the hooks directory {live_dir}, then restart the run."
            )
        # The marker records the resolved path, so compare like with like.
        if source_dir is not None and record.get("source_dir") != str(
            source_dir.resolve()
        ):
            log.warning(
                "hooks_source_dir {} differs from the baseline recorded in {} for this run ({}); "
                "the existing live tree is reused unchanged",
                source_dir,
                marker_path,
                record.get("source_dir"),
            )
        warn_on_catalogue_mismatch(record, marker_path, live_dir, log)
        if not live_dir.exists():
            log.warning(
                "hooks: the live tree {} is missing, but the marker {} records this run, "
                "so no hook runs; a resume never seeds the tree again. To start the hooks "
                "tree again, delete the marker, then restart the run.",
                live_dir,
                marker_path,
            )
        return HooksTree(live_dir, record["session_id"])

    resolved_source = None
    if source_dir is not None:
        resolved_source = resolve_hooks_source(source_dir, base_dir)

    if live_dir.exists():
        return _adopt_live_tree(live_dir, marker_path, log)

    base_dir.mkdir(parents=True, exist_ok=True)
    if publish_fresh_tree(base_dir, live_dir, resolved_source, session_id, log):
        return HooksTree(live_dir, session_id)
    # A concurrent initializer (e.g. another DDP rank sharing this log_dir) won the race.
    return _adopt_live_tree(live_dir, marker_path, log)


def _adopt_live_tree(live_dir: Path, marker_path: Path, log: "Logger") -> HooksTree:
    """Resumes an existing live tree from its provenance record, restoring the marker.

    Warns, as a resume from the marker does, when the record holds another
    catalogue version.

    Raises:
        HooksBootstrapError: `live_dir` has no compatible provenance record.
    """
    provenance_path = live_dir / PROVENANCE_NAME
    record = read_hooks_record(provenance_path)
    if record is not None:
        warn_on_catalogue_mismatch(record, provenance_path, live_dir, log)
        write_json_file(marker_path, record)
        return HooksTree(live_dir, record["session_id"])
    raise HooksBootstrapError(
        f"{live_dir} already exists but is not a Midrun-owned hooks tree (no compatible "
        f"{marker_path.name} or {PROVENANCE_NAME} found); refusing to reuse or overwrite it. "
        "Use a fresh run directory, or remove the conflicting tree yourself."
    )
