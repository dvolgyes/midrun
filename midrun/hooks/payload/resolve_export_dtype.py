"""Selects the numpy dtype that a tensor is exported as."""

import torch

_NUMPY_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}


_BFLOAT16_EXPORT_DTYPE = "float32"


def resolve_export_dtype(tensor: torch.Tensor) -> str | None:
    """Returns the numpy dtype name this tensor would be saved as, or `None` if unsupported."""
    if tensor.dtype is torch.bfloat16:
        return _BFLOAT16_EXPORT_DTYPE
    return _NUMPY_DTYPE_NAMES.get(tensor.dtype)
