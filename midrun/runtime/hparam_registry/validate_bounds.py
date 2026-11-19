"""Checks a tunable-hparam value against its declared bounds."""

import torch

from midrun.runtime.hparam_registry.tunable_hparam_spec import TunableHparamSpec


def validate_bounds(spec: TunableHparamSpec, value: object) -> None:
    """Raises ValueError if `value` violates `spec.min`/`spec.max`.

    A no-op when neither bound is declared (every bare `list[str]`-form entry,
    and any dict-form entry that simply omits `min`/`max`).

    Check tensor and sequence elements at their original precision. In particular,
    a tensor comparison would first round a scalar bound to the tensor's dtype.
    """
    if spec.min is None and spec.max is None:
        return
    if isinstance(value, torch.Tensor):
        _validate_extrema(spec, value)
        return
    if isinstance(value, list | tuple):
        for element in value:
            validate_bounds(spec, element)
        return
    if not isinstance(value, int | float):
        raise TypeError(f"{spec.path!r}: numeric bounds require a numeric value")
    if spec.min is not None and value < spec.min:
        raise ValueError(f"{spec.path!r} must be >= {spec.min}, got {value!r}")
    if spec.max is not None and value > spec.max:
        raise ValueError(f"{spec.path!r} must be <= {spec.max}, got {value!r}")


def _validate_extrema(spec: TunableHparamSpec, tensor: torch.Tensor) -> None:
    """Compare extrema as Python numbers, including unsupported unsigned reductions."""
    if tensor.numel() == 0:
        return
    values = (
        tensor.detach().cpu().numpy()
        if tensor.dtype in (torch.uint16, torch.uint32, torch.uint64)
        else tensor
    )
    if spec.min is not None:
        validate_bounds(spec, values.min().item())
    if spec.max is not None:
        validate_bounds(spec, values.max().item())
