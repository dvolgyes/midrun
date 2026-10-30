"""Checks that a tensor layout and device can be exported."""

import torch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


def is_supported_tensor_layout(tensor: torch.Tensor) -> bool:
    """Returns whether `tensor` is dense, strided, not quantized and has real data.

    A wrapper subclass such as `DTensor` reports a strided layout but has no
    storage of its own, and `.numpy()` refuses it, so it is unsupported.
    """
    if tensor.layout is not torch.strided:
        return False
    if is_traceable_wrapper_subclass(tensor):
        return False
    if tensor.is_quantized:
        return False
    if tensor.device.type == "meta":
        return False
    return not getattr(tensor, "is_nested", False)
