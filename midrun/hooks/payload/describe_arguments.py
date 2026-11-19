"""Describes callback arguments as a bounded, JSON-safe structure."""

import math
from collections.abc import Iterable, Mapping
from typing import Any, cast

import torch

from midrun.hooks.payload.traversal_state import TraversalState

_MAX_STRING_LEN = 200


def describe_arguments(
    arguments: Mapping[str, Any], *, max_items: int, max_depth: int
) -> tuple[dict[str, Any], bool, str | None]:
    """Describes a callback's named arguments as a bounded, JSON-safe structure.

    Args:
        arguments: The callback's own arguments, by name (e.g. `{"batch": ..., "batch_idx": ...}`).
        max_items: Maximum entries kept per traversed `Mapping`/`list`/`tuple`; the
            whole description also stops after 10 x `max_items` values.
        max_depth: Maximum recursion depth below `arguments` itself.

    Returns:
        A tuple of `(description, truncated, reason)`. `reason` is `None` unless
        `truncated` is True, in which case it names every bound that was hit.
    """
    state = TraversalState(max_items=max_items, max_depth=max_depth)
    described = _describe_items(
        arguments.items(),
        state,
        depth=-1,
        object_id=id(arguments),
        is_mapping=True,
        total=len(arguments),
    )
    reason = "; ".join(sorted(state.reasons)) if state.truncated else None
    return cast("dict[str, object]", described), state.truncated, reason


def _describe(value: object, state: TraversalState, *, depth: int) -> object:
    if isinstance(value, torch.Tensor):
        shape = None if value.is_nested else list(value.shape)
        return {
            "kind": "tensor",
            "shape": shape,
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": value.requires_grad,
        }
    if value is None or isinstance(value, bool | int | float):
        if isinstance(value, float) and not math.isfinite(value):
            value = {"kind": "float", "value": repr(float(value))}
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STRING_LEN:
            state.truncated = True
            state.reasons.add("string_length")
            value = value[:_MAX_STRING_LEN] + "...(truncated)"
        return value
    if depth >= state.max_depth:
        state.truncated = True
        state.reasons.add("max_depth")
        return {"kind": "truncated", "reason": "max_depth"}
    object_id = id(value)
    if object_id in state.seen:
        value = {"kind": "cycle"}
    elif isinstance(value, Mapping | list | tuple):
        value = _describe_items(
            value.items() if isinstance(value, Mapping) else enumerate(value),
            state,
            depth=depth,
            object_id=object_id,
            is_mapping=isinstance(value, Mapping),
            total=len(value),
        )
    else:
        value = {"kind": "object", "type": type(value).__qualname__}
    return value


def _describe_items(
    items: Iterable[tuple[object, object]],
    state: TraversalState,
    *,
    depth: int,
    object_id: int,
    is_mapping: bool,
    total: int,
) -> dict[str, object] | list[object]:
    state.seen.add(object_id)
    described = [
        (key, _describe(value, state, depth=depth + 1))
        for key, value in state.bounded_items(items, total)
    ]
    omitted = total - len(described)
    state.seen.discard(object_id)
    if is_mapping:
        result = {str(key): value for key, value in described}
        if omitted:
            result["__truncated__"] = f"{omitted} more item(s) omitted"
        return result
    result_list = [value for _key, value in described]
    if omitted:
        result_list.append(
            {"kind": "truncated", "reason": f"{omitted} more item(s) omitted"}
        )
    return result_list
