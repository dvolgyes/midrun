"""Checks a tunable-hparam value against its declared bounds."""

from typing import Any

import torch

from trainctl.runtime.hparam_registry.tunable_hparam_spec import TunableHparamSpec


def validate_bounds(spec: TunableHparamSpec, value: Any) -> None:
    """Raises ValueError if `value` violates `spec.min`/`spec.max`.

    A no-op when neither bound is declared (every bare `list[str]`-form entry,
    and any dict-form entry that simply omits `min`/`max`).

    A tensor-shaped `value` (a `torch.Tensor`, or a nested list/tuple -- the JSON
    encoding of one) is checked element-wise via `torch.as_tensor`: every
    element must satisfy both bounds, not just an aggregate. A rank-0 tensor
    (the common case for a single-value tunable tensor) degenerates to the same
    comparison as a plain scalar automatically. A plain scalar `value` is
    compared directly.
    """
    if spec.min is None and spec.max is None:
        return
    if isinstance(value, torch.Tensor | list | tuple):
        tensor = torch.as_tensor(value)
        if spec.min is not None and bool((tensor < spec.min).any()):
            raise ValueError(
                f"{spec.path!r}: all elements must be >= {spec.min}, "
                f"got min {tensor.min().item()!r}"
            )
        if spec.max is not None and bool((tensor > spec.max).any()):
            raise ValueError(
                f"{spec.path!r}: all elements must be <= {spec.max}, "
                f"got max {tensor.max().item()!r}"
            )
        return
    if spec.min is not None and value < spec.min:
        raise ValueError(f"{spec.path!r} must be >= {spec.min}, got {value!r}")
    if spec.max is not None and value > spec.max:
        raise ValueError(f"{spec.path!r} must be <= {spec.max}, got {value!r}")
