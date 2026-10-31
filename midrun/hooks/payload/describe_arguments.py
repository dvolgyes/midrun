"""Describes callback arguments as a bounded, JSON-safe structure."""

import math
from collections.abc import Iterable, Mapping
from typing import Any

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
    described = {
        str(name): _describe(value, state, depth=0) for name, value in arguments.items()
    }
    reason = "; ".join(sorted(state.reasons)) if state.truncated else None
    return described, state.truncated, reason


def _describe(value: object, state: TraversalState, *, depth: int) -> object:
    state.nodes_left -= 1
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": value.requires_grad,
        }
    if value is None or isinstance(value, bool | int | float):
        if isinstance(value, float) and not math.isfinite(value):
            # Standard JSON has no NaN/Infinity literal.
            value = {"kind": "float", "value": repr(float(value))}
        return value
    if isinstance(value, str):
        if len(value) <= _MAX_STRING_LEN:
            return value
        state.truncated = True
        state.reasons.add("string_length")
        return value[:_MAX_STRING_LEN] + "...(truncated)"
    if depth >= state.max_depth:
        state.truncated = True
        state.reasons.add("max_depth")
        return {"kind": "truncated", "reason": "max_depth"}
    object_id = id(value)
    if object_id in state.seen:
        return {"kind": "cycle"}
    if isinstance(value, Mapping):
        return _describe_items(
            value.items(), state, depth=depth, object_id=object_id, is_mapping=True
        )
    if isinstance(value, list | tuple):
        return _describe_items(
            enumerate(value), state, depth=depth, object_id=object_id, is_mapping=False
        )
    return {"kind": "object", "type": type(value).__qualname__}


def _describe_items(
    items: Iterable[tuple[object, object]],
    state: TraversalState,
    *,
    depth: int,
    object_id: int,
    is_mapping: bool,
) -> object:
    state.seen.add(object_id)
    described: list[tuple[Any, Any]] = []
    omitted = 0
    for key, value in items:
        if len(described) >= state.max_items or state.nodes_left <= 0:
            omitted += 1
            continue
        described.append((key, _describe(value, state, depth=depth + 1)))
    state.seen.discard(object_id)
    if omitted:
        state.truncated = True
        state.reasons.add(
            "max_items" if len(described) >= state.max_items else "max_nodes"
        )
    if is_mapping:
        result: dict[str, Any] = {str(key): value for key, value in described}
        if omitted:
            result["__truncated__"] = f"{omitted} more item(s) omitted"
        return result
    result_list: list[Any] = [value for _key, value in described]
    if omitted:
        result_list.append(
            {"kind": "truncated", "reason": f"{omitted} more item(s) omitted"}
        )
    return result_list
