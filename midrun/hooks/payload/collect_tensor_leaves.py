"""Finds the tensors in the tensor-carrying arguments of a callback."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from midrun.hooks.payload.tensor_leaf import TensorLeaf
from midrun.hooks.payload.traversal_state import TraversalState


def collect_tensor_leaves(
    arguments: Mapping[str, Any],
    tensor_arg_names: Sequence[str],
    *,
    max_items: int,
    max_depth: int,
) -> tuple[list[TensorLeaf], bool, str | None]:
    """Finds every tensor nested within this callback's tensor-carrying arguments.

    Only traverses `arguments[name]` for each `name in tensor_arg_names` -- the
    catalogue's static, per-callback list of arguments that may carry tensors.
    Bounded and cycle-safe like `describe_arguments`; hitting a bound stops that
    branch (fewer tensors found), it does not fail the capture.
    """
    state = TraversalState(max_items=max_items, max_depth=max_depth)
    leaves: list[TensorLeaf] = []
    for name in tensor_arg_names:
        if name in arguments:
            _collect_leaves(arguments[name], name, state, leaves, depth=0)
    reason = "; ".join(sorted(state.reasons)) if state.truncated else None
    return leaves, state.truncated, reason


def _collect_leaves(
    value: Any,
    path: str,
    state: TraversalState,
    leaves: list[TensorLeaf],
    *,
    depth: int,
) -> None:
    if isinstance(value, torch.Tensor):
        leaves.append(TensorLeaf(path=path, tensor=value))
        return
    if not isinstance(value, Mapping | list | tuple):
        return
    if depth >= state.max_depth:
        state.truncated = True
        state.reasons.add("max_depth")
        return
    object_id = id(value)
    if object_id in state.seen:
        return
    items = value.items() if isinstance(value, Mapping) else enumerate(value)
    state.seen.add(object_id)
    for index, (key, item) in enumerate(items):
        if index >= state.max_items:
            state.truncated = True
            state.reasons.add("max_items")
            break
        _collect_leaves(item, f"{path}/{key}", state, leaves, depth=depth + 1)
    state.seen.discard(object_id)
