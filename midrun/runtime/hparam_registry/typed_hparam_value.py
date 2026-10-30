"""Checks that a new hparam value has the type of the current value."""

import math

import torch

from midrun.runtime.hparam_registry.describe_hparam import describe_hparam


def typed_hparam_value(current: object, leaf: str, new_value: object) -> object:
    """Returns `new_value` in the type of `current`, the value of the knob `leaf`.

    See `check_and_assign` for the type rules. An int becomes a float for a float
    knob, and a nested list becomes a tensor for a tensor knob.

    Raises:
        ValueError: `new_value` has the wrong type or shape, or is not finite.
    """
    kind = describe_hparam(current).type
    if isinstance(current, torch.Tensor):
        if isinstance(new_value, str):
            raise ValueError(f"{leaf!r} is a tensor hparam; got {new_value!r}")
        tensor = torch.as_tensor(new_value, device=current.device)
        if _changes_element_kind(tensor, current):
            raise ValueError(
                f"{leaf!r} is a {current.dtype} tensor hparam; got {tensor.dtype} "
                f"elements {new_value!r}"
            )
        tensor = tensor.to(current.dtype)
        if tuple(tensor.shape) != tuple(current.shape):
            raise ValueError(
                f"shape mismatch: {leaf!r} has shape {tuple(current.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{leaf!r}: every element must be finite")
        return tensor
    # bool is an int subclass, so a bool is never a valid int or float value.
    valid = {
        "bool": isinstance(new_value, bool),
        "int": isinstance(new_value, int) and not isinstance(new_value, bool),
        "float": isinstance(new_value, int | float) and not isinstance(new_value, bool),
        "str": isinstance(new_value, str),
    }[kind]
    if not valid:
        raise ValueError(
            f"{leaf!r} is a {kind} hparam; got {type(new_value).__name__} {new_value!r}"
        )
    if kind == "float" and isinstance(new_value, int | float):
        new_value = float(new_value)
        if not math.isfinite(new_value):
            raise ValueError(f"{leaf!r} must be finite, got {new_value!r}")
    return new_value


def _changes_element_kind(tensor: torch.Tensor, current: torch.Tensor) -> bool:
    """Returns whether a cast of `tensor` to `current`'s dtype converts its elements.

    A bool and a number are not convertible, and neither is a float to an integer. An
    int is a valid element of a float tensor. An empty tensor has no element type.
    """
    if tensor.numel() == 0:
        return False
    if (tensor.dtype == torch.bool) != (current.dtype == torch.bool):
        return True
    return tensor.is_floating_point() and not (
        current.is_floating_point() or current.is_complex()
    )
