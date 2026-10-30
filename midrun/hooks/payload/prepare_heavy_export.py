"""Validates the tensors of a heavy hook before any copy."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy

from midrun.hooks.payload.collect_tensor_leaves import collect_tensor_leaves
from midrun.hooks.payload.heavy_capture_failure import HeavyCaptureFailure
from midrun.hooks.payload.is_supported_tensor_layout import (
    is_supported_tensor_layout,
)
from midrun.hooks.payload.resolve_export_dtype import resolve_export_dtype
from midrun.hooks.payload.tensor_leaf import TensorLeaf


def prepare_heavy_export(
    arguments: Mapping[str, Any],
    tensor_arg_names: Sequence[str],
    *,
    max_items: int,
    max_depth: int,
    max_export_bytes: int,
) -> tuple[list[TensorLeaf], bool, str | None] | HeavyCaptureFailure:
    """Validates every tensor a heavy hook would export, without copying any of them.

    "All or failed": every found tensor must have a supported layout/dtype and the
    predicted total converted size must fit `max_export_bytes`, checked before any
    host allocation. A callback with no tensor arguments (or none present this
    firing) validates successfully with an empty leaf list -- a valid empty export.
    """
    leaves, truncated, reason = collect_tensor_leaves(
        arguments, tensor_arg_names, max_items=max_items, max_depth=max_depth
    )
    predicted_bytes = 0
    for leaf in leaves:
        if not is_supported_tensor_layout(leaf.tensor):
            return HeavyCaptureFailure(
                f"unsupported tensor layout/device at {leaf.path}: "
                f"type={type(leaf.tensor).__name__}, "
                f"layout={leaf.tensor.layout}, device={leaf.tensor.device}"
            )
        export_dtype = resolve_export_dtype(leaf.tensor)
        if export_dtype is None:
            return HeavyCaptureFailure(
                f"unsupported dtype at {leaf.path}: {leaf.tensor.dtype}"
            )
        predicted_bytes += leaf.tensor.numel() * numpy.dtype(export_dtype).itemsize
    if predicted_bytes > max_export_bytes:
        return HeavyCaptureFailure(
            f"predicted export size {predicted_bytes} bytes exceeds "
            f"hooks_max_export_bytes ({max_export_bytes})"
        )
    return leaves, truncated, reason
