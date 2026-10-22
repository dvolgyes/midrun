"""Assembles the manifest.json document of a heavy hook."""

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

from trainctl.hooks.payload.describe_arguments import describe_arguments
from trainctl.hooks.payload.occurrence_context import OccurrenceContext
from trainctl.hooks.payload.tensor_record import TensorRecord

_HEAVY_SCHEMA_VERSION = 1


def build_heavy_manifest(
    *,
    context: OccurrenceContext,
    occurrence_id: str,
    invocation_id: str,
    arguments: Mapping[str, Any],
    tensor_arg_names: Sequence[str],
    tensor_records: list[TensorRecord],
    max_metadata_items: int,
    max_metadata_depth: int,
    truncated: bool,
    truncated_reason: str | None,
) -> dict[str, Any]:
    """Assembles the heavy hook's `manifest.json` document.

    `arguments` describes only the tensor-carrying arguments' non-tensor structure
    (primitives, containers) plus each tensor's shape/dtype/device -- the same
    bounded, non-evaluating description the light path uses. `tensor_records`
    separately names the actual exported `.npy` files; a script correlates the two
    by `argument_path`.
    """
    restricted = {
        name: arguments[name] for name in tensor_arg_names if name in arguments
    }
    described, meta_truncated, meta_reason = describe_arguments(
        restricted, max_items=max_metadata_items, max_depth=max_metadata_depth
    )
    all_truncated = truncated or meta_truncated
    reasons = [r for r in (truncated_reason, meta_reason) if r]
    return {
        "schema_version": _HEAVY_SCHEMA_VERSION,
        "session_id": context.session_id,
        "occurrence_id": occurrence_id,
        "invocation_id": invocation_id,
        "hook": context.hook,
        "modality": "heavy",
        "timestamp": time.time(),
        "pid": os.getpid(),
        "rank": context.rank,
        "world_size": context.world_size,
        "stage": context.stage,
        "epoch": context.epoch,
        "global_step": context.global_step,
        "batch_idx": context.batch_idx,
        "dataloader_idx": context.dataloader_idx,
        "arguments": described,
        "tensor_records": [
            {
                "argument_path": record.argument_path,
                "file": record.file,
                "shape": record.shape,
                "dtype": record.dtype,
                "original_dtype": record.original_dtype,
                "device": record.device,
            }
            for record in tensor_records
        ],
        "truncated": all_truncated,
        "truncated_reason": "; ".join(reasons) if reasons else None,
    }
