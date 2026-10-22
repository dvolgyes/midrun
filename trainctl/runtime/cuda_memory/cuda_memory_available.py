"""Tells whether this process has a CUDA device."""

import torch


def cuda_memory_available() -> bool:
    """Returns whether a CUDA device is available in this process."""
    return torch.cuda.is_available()
