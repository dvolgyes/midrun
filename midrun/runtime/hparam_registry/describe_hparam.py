"""Describes the current value of a tunable hparam."""

import torch

from midrun.runtime.hparam_registry.hparam_value import HparamValue

MAX_INLINE_TENSOR_ELEMENTS = 4096


def describe_hparam(value: object, *, read_tensor: bool = True) -> HparamValue:
    """Reports a tunable hparam's current value, type, and (for tensors) dtype/device/shape.

    With `read_tensor` False, a tensor's value stays None: the call then does not copy
    the tensor to the host, so it does not wait for the tensor's device.

    Raises:
        ValueError: `value`'s type is not one of bool/int/float/str/`torch.Tensor`.
    """
    if isinstance(value, torch.Tensor):
        inline_value = (
            value.detach().cpu().tolist()
            if read_tensor and value.numel() <= MAX_INLINE_TENSOR_ELEMENTS
            else None
        )
        return HparamValue(
            value=inline_value,
            type="tensor",
            dtype=str(value.dtype),
            device=str(value.device),
            shape=list(value.shape),
        )
    if isinstance(value, bool):  # must precede the int check: bool is an int subclass
        return HparamValue(value=value, type="bool")
    if isinstance(value, int):
        return HparamValue(value=value, type="int")
    if isinstance(value, float):
        return HparamValue(value=value, type="float")
    if isinstance(value, str):
        return HparamValue(value=value, type="str")
    raise ValueError(f"unsupported tunable hparam type: {type(value).__name__}")
