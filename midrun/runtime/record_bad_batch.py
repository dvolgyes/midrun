"""Writes the snapshot that records a rank's non-finite dataloader batch."""

from dataclasses import asdict

from loguru import logger

from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.finite_guard import FiniteCheckReport


def record_bad_batch(
    store: ArtifactStore | None,
    report: FiniteCheckReport,
    *,
    rank: int,
    epoch: int,
    global_step: int,
    batch_idx: int,
) -> None:
    """Writes an `anomaly_input_nonfinite` snapshot of `report`, or logs why it cannot.

    It never raises: in a distributed run the other ranks wait for this rank in the
    hold that follows, so a failed record must not keep this rank out of it.
    """
    if store is None:
        logger.error(
            "midrun: rank {} could not record its non-finite batch: no artifact store",
            rank,
        )
        return
    payload = {
        "reason": "nonfinite_dataloader_tensor",
        "rank": rank,
        "epoch": epoch,
        "global_step": global_step,
        "batch_idx": batch_idx,
        "tensors": [asdict(bad) for bad in report.bad_tensors],
    }
    try:
        manifest = store.write_snapshot("anomaly_input_nonfinite", payload)
    # pylint: disable-next=broad-exception-caught
    except Exception as exc:  # noqa: BLE001 -- the hold that follows must still run
        logger.error(
            "midrun: rank {} could not record its non-finite batch: {}", rank, exc
        )
        return
    logger.warning(
        "midrun: rank {} batch {} (epoch {}, global step {}) has non-finite values in {}; "
        "snapshot {} records them",
        rank,
        batch_idx,
        epoch,
        global_step,
        ", ".join(
            f"{bad.path} ({bad.nan_count} NaN, {bad.posinf_count} +inf, "
            f"{bad.neginf_count} -inf)"
            for bad in report.bad_tensors
        ),
        manifest["id"],
    )
