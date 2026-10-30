"""The device that torchview must trace a model on."""

import itertools

import torch


def torchview_device(model: torch.nn.Module) -> torch.device:
    """Returns the device of the model's first parameter or buffer, else CPU.

    torchview defaults to cuda whenever it's available and calls model.to(device)
    on the *actual* model (nn.Module.to is in-place) -- pinning it to the model's real
    device means a live capture can't silently relocate the model mid-training. A
    model without parameters or buffers has no state that `to()` could move, so CPU
    is safe for it; torchview then moves only its copy of the input there.
    """
    tensor = next(itertools.chain(model.parameters(), model.buffers()), None)
    return torch.device("cpu") if tensor is None else tensor.device
