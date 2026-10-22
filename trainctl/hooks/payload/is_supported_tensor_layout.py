"""Checks that a tensor layout and device can be exported."""

import torch


def is_supported_tensor_layout(tensor: torch.Tensor) -> bool:
    """Returns whether `tensor` is dense, strided, not quantized and has real data."""
    if tensor.layout is not torch.strided:
        return False
    if tensor.is_quantized:
        return False
    if tensor.device.type == "meta":
        return False
    return not getattr(tensor, "is_nested", False)
