"""Assembles the manifest.json document of a heavy hook."""

from collections.abc import Mapping, Sequence
from typing import Any

from midrun.hooks.payload.describe_arguments import describe_arguments
from midrun.hooks.payload.invocation_header import invocation_header
from midrun.hooks.payload.occurrence_context import OccurrenceContext
from midrun.hooks.payload.tensor_record import TensorRecord

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
    truncation: tuple[bool, str | None],
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
    truncated, truncated_reason = truncation
    manifest = invocation_header(
        context,
        occurrence_id,
        invocation_id,
        modality="heavy",
        version=_HEAVY_SCHEMA_VERSION,
    )
    manifest.update(
        arguments=described,
        tensor_records=[
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
        tensor_capture_partial=truncated,
        truncated=truncated or meta_truncated,
        truncated_reason="; ".join(filter(None, (truncated_reason, meta_reason)))
        or None,
    )
    return manifest
