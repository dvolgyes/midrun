"""Build the identity fields shared by both hook payload formats."""

import os
import time
from typing import Any, Literal

from midrun.hooks.payload.occurrence_context import OccurrenceContext


def invocation_header(
    context: OccurrenceContext,
    occurrence_id: str,
    invocation_id: str,
    *,
    modality: Literal["light", "heavy"],
    version: int,
) -> dict[str, Any]:
    """Start a payload with its schema version, process identity and training position."""
    return {
        "schema_version": version,
        "session_id": context.session_id,
        "occurrence_id": occurrence_id,
        "invocation_id": invocation_id,
        "hook": context.hook,
        "modality": modality,
        "timestamp": time.time(),
        "pid": os.getpid(),
        "rank": context.rank,
        "world_size": context.world_size,
        "stage": context.stage,
        "epoch": context.epoch,
        "global_step": context.global_step,
        "batch_idx": context.batch_idx,
        "dataloader_idx": context.dataloader_idx,
    }
