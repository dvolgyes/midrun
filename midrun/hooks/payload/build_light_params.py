"""Assembles the params.json document of a light hook."""

from collections.abc import Mapping
from typing import Any

from midrun.hooks.encode_json_document import encode_json_document
from midrun.hooks.payload.describe_arguments import describe_arguments
from midrun.hooks.payload.invocation_header import invocation_header
from midrun.hooks.payload.occurrence_context import OccurrenceContext

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

    Enforces `max_params_bytes` on the bytes that `write_json_file` writes: if the
    described arguments would push the document over budget, `arguments` is dropped
    entirely (not partially re-truncated) and the truncation reason records why.

    Returns:
        A tuple of `(params, truncated, reason)`, mirroring `describe_arguments`.
    """
    described, truncated, reason = describe_arguments(
        arguments, max_items=max_metadata_items, max_depth=max_metadata_depth
    )
    params = invocation_header(
        context, occurrence_id, invocation_id, modality="light", version=_SCHEMA_VERSION
    )
    params.update(arguments=described, truncated=truncated, truncated_reason=reason)
    if len(encode_json_document(params)) <= max_params_bytes:
        return params, truncated, reason
    reason = "max_params_bytes" if reason is None else f"{reason}; max_params_bytes"
    params.update(arguments=None, truncated=True, truncated_reason=reason)
    if len(encode_json_document(params)) > max_params_bytes:
        raise ValueError(
            f"light payload envelope exceeds hooks_max_params_bytes ({max_params_bytes})"
        )
    return params, True, reason
