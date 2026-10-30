"""The described current value of one tunable hparam."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HparamValue:
    """A tunable hparam's current value, its type, and (for tensors) dtype/device/shape.

    Attributes:
        value: The current value, or `None` for a tensor above
            `MAX_INLINE_TENSOR_ELEMENTS` elements.
        type: One of `"bool"`, `"int"`, `"float"`, `"str"`, `"tensor"`.
        dtype: `str(tensor.dtype)`, tensors only.
        device: `str(tensor.device)`, tensors only.
        shape: `list(tensor.shape)`, tensors only.
        min: The field's declared lower bound, if any -- spec metadata, not
            runtime-introspected.
        max: The field's declared upper bound, if any.
        label: The field's declared display name, if any.
    """

    value: Any
    type: str
    dtype: str | None = None
    device: str | None = None
    shape: list[int] | None = None
    min: int | float | None = None
    max: int | float | None = None
    label: str | None = None
