"""Checks that a new hparam value has the type of the current value."""

import math

import torch

from trainctl.runtime.hparam_registry.describe_hparam import describe_hparam


def typed_hparam_value(current: object, leaf: str, new_value: object) -> object:
    """Returns `new_value` in the type of `current`, the value of the knob `leaf`.

    See `check_and_assign` for the type rules. An int becomes a float for a float
    knob, and a nested list becomes a tensor for a tensor knob.

    Raises:
        ValueError: `new_value` has the wrong type or shape, or is not finite.
    """
    kind = describe_hparam(current).type
    if isinstance(current, torch.Tensor):
        if isinstance(new_value, str) or (
            isinstance(new_value, bool) and current.dtype != torch.bool
        ):
            raise ValueError(f"{leaf!r} is a tensor hparam; got {new_value!r}")
        tensor = torch.as_tensor(new_value, dtype=current.dtype, device=current.device)
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
