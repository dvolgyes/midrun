"""Assembles the params.json document of a light hook."""

import json
import os
import time
from collections.abc import Mapping
from typing import Any

from trainctl.hooks.payload.describe_arguments import describe_arguments
from trainctl.hooks.payload.occurrence_context import OccurrenceContext

_SCHEMA_VERSION = 1


def build_light_params(
    *,
    context: OccurrenceContext,
    occurrence_id: str,
    invocation_id: str,
    arguments: Mapping[str, Any],
    max_params_bytes: int,
    max_metadata_items: int,
    max_metadata_depth: int,
) -> tuple[dict[str, Any], bool, str | None]:
    """Assembles the light hook's `params.json` document.

    Enforces `max_params_bytes` on the encoded result: if the described arguments
    would push the document over budget, `arguments` is dropped entirely (not
    partially re-truncated) and the truncation reason records why.

    Returns:
        A tuple of `(params, truncated, reason)`, mirroring `describe_arguments`.
    """
    described, truncated, reason = describe_arguments(
        arguments, max_items=max_metadata_items, max_depth=max_metadata_depth
    )
    params = _assemble(
        context, occurrence_id, invocation_id, described, truncated, reason
    )
    encoded = json.dumps(params).encode("utf-8")
    if len(encoded) <= max_params_bytes:
        return params, truncated, reason
    reason = "max_params_bytes" if reason is None else f"{reason}; max_params_bytes"
    params = _assemble(context, occurrence_id, invocation_id, None, True, reason)
    return params, True, reason


def _assemble(
    context: OccurrenceContext,
    occurrence_id: str,
    invocation_id: str,
    arguments: dict[str, Any] | None,
    truncated: bool,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "session_id": context.session_id,
        "occurrence_id": occurrence_id,
        "invocation_id": invocation_id,
        "hook": context.hook,
        "modality": "light",
        "timestamp": time.time(),
        "pid": os.getpid(),
        "rank": context.rank,
        "world_size": context.world_size,
        "stage": context.stage,
        "epoch": context.epoch,
        "global_step": context.global_step,
        "batch_idx": context.batch_idx,
        "dataloader_idx": context.dataloader_idx,
        "arguments": arguments,
        "truncated": truncated,
        "truncated_reason": reason,
    }
