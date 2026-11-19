"""Checks that a new hparam value has the type of the current value."""

import math

import numpy as np
import torch

from midrun.runtime.hparam_registry.describe_hparam import describe_hparam
from midrun.runtime.hparam_registry.tunable_hparam_spec import TunableHparamSpec
from midrun.runtime.hparam_registry.validate_bounds import validate_bounds


def typed_hparam_value(
    current: object, leaf: str, new_value: object, spec: TunableHparamSpec
) -> object:
    """Returns `new_value` in the type of `current`, the value of the knob `leaf`.

    See `check_and_assign` for the type rules. An int becomes a float for a float
    knob, and a nested list becomes a tensor for a tensor knob.

    Raises:
        ValueError: `new_value` has the wrong type or shape, or is not finite.
    """
    if isinstance(current, torch.Tensor):
        return _tensor_value(current, leaf, new_value, spec)
    kind = describe_hparam(current).type
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


def _tensor_value(
    current: torch.Tensor, leaf: str, new_value: object, spec: TunableHparamSpec
) -> torch.Tensor:
    """Check source elements and the stored representation before assignment."""
    if isinstance(new_value, str):
        raise ValueError(f"{leaf!r} is a tensor hparam; got {new_value!r}")
    if isinstance(new_value, list | tuple) and _has_wrong_boolean_element(
        new_value, current.dtype == torch.bool
    ):
        raise ValueError(
            f"{leaf!r} is a {current.dtype} tensor hparam; got incompatible "
            f"boolean or numeric elements {new_value!r}"
        )
    tensor = _source_tensor(new_value, current.dtype)
    if _changes_element_kind(tensor, current):
        raise ValueError(
            f"{leaf!r} is a {current.dtype} tensor hparam; got {tensor.dtype} "
            f"elements {new_value!r}"
        )
    if tensor.shape != current.shape:
        raise ValueError(
            f"shape mismatch: {leaf!r} has shape {tuple(current.shape)}, "
            f"got {tuple(tensor.shape)}"
        )
    _require_finite(tensor, leaf)
    validate_bounds(spec, new_value if isinstance(new_value, list | tuple) else tensor)
    if tensor.dtype == current.dtype:
        return tensor.to(device=current.device)
    integer_target = not (
        current.is_complex()
        or current.is_floating_point()
        or current.dtype == torch.bool
    )
    if integer_target:
        limits = torch.iinfo(current.dtype)
        validate_bounds(TunableHparamSpec(leaf, limits.min, limits.max), tensor)
    converted = tensor.to(device=current.device, dtype=current.dtype)
    _require_finite(converted, leaf)
    validate_bounds(spec, converted)
    return converted


def _require_finite(tensor: torch.Tensor, leaf: str) -> None:
    """Reject nonfinite real or imaginary components."""
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{leaf!r}: every element must be finite")


def _has_wrong_boolean_element(value: object, boolean_target: bool) -> bool:
    """Check boolean and numeric sequence leaves before dtype inference."""
    if isinstance(value, list | tuple):
        return any(_has_wrong_boolean_element(item, boolean_target) for item in value)
    if isinstance(value, bool | np.bool_):
        return not boolean_target
    return boolean_target and isinstance(value, int | float | complex | np.number)


def _changes_element_kind(tensor: torch.Tensor, current: torch.Tensor) -> bool:
    """Returns whether a cast of `tensor` to `current`'s dtype converts its elements.

    A bool and a number are not convertible, and neither is a float to an integer. An
    int is a valid element of a float tensor. An empty tensor has no element type.
    A complex source requires a complex target even when empty.
    """
    if tensor.is_complex() and not current.is_complex():
        return True
    if tensor.numel() == 0:
        return False
    if (tensor.dtype == torch.bool) != (current.dtype == torch.bool):
        return True
    return tensor.is_floating_point() and not (
        current.is_floating_point() or current.is_complex()
    )


def _source_tensor(value: object, target: torch.dtype) -> torch.Tensor:
    """Infer source elements without narrowing Python floats or wide unsigned integers."""
    if not isinstance(value, list | tuple | int | float):
        return torch.as_tensor(value)
    array = np.asarray(value)
    if target == torch.uint64 and array.dtype.kind in {"f", "O"}:
        return _uint64_source(value)
    return torch.as_tensor(array)


def _uint64_source(value: object) -> torch.Tensor:
    """Preserve exact integers where NumPy would otherwise infer float or object."""
    elements = np.asarray(value, dtype=object)
    if not all(isinstance(item, int) for item in elements.flat):
        return torch.as_tensor(np.asarray(value))
    if any(not 0 <= item < 2**64 for item in elements.flat):
        raise ValueError("value outside the range of torch.uint64")
    return torch.as_tensor(elements.astype(np.uint64))
