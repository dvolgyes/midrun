"""Turns the values of one safe point into the values that it publishes."""

from collections.abc import Sequence

import torch
from torch import Tensor

from midrun.runtime.state.host_scalar import HostScalar
from midrun.runtime.state.publishable_value import publishable_value


def publishable_values(values: Sequence[object]) -> list[float | HostScalar | None]:
    """Returns each value as a publishable scalar, without a wait for a CUDA device.

    The CUDA scalars of one device and dtype are stacked with one kernel and copied
    to pinned host memory on the device's current stream, and an event is recorded
    after the copy; each becomes a `HostScalar` that a reader waits for. No device
    tensor outlives this call. Every other value goes through `publishable_value`.

    Args:
        values: Values from `trainer.callback_metrics`, or learning rates.

    Returns:
        One entry per value, in order: a float, a `HostScalar`, or None when the
        value is not a real scalar.
    """
    published: list[float | HostScalar | None] = []
    groups: dict[tuple[torch.device, torch.dtype], list[tuple[int, Tensor]]] = {}
    for value in values:
        if (
            isinstance(value, Tensor)
            and value.is_cuda
            and value.numel() == 1
            and not value.is_complex()
        ):
            key = (value.device, value.dtype)
            groups.setdefault(key, []).append((len(published), value))
            published.append(None)
        else:
            published.append(publishable_value(value))
    for members in groups.values():
        with torch.no_grad():
            # Most metrics have no dimensions; a reshape for them costs a dispatch.
            stacked = torch.stack(
                [
                    tensor.detach()
                    if tensor.dim() == 0
                    else tensor.detach().reshape(())
                    for _, tensor in members
                ]
            )
            # A non-blocking copy from CUDA allocates pinned host memory.
            host = stacked.to("cpu", non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(stacked.device))
        for index, (position, _) in enumerate(members):
            published[position] = HostScalar(host, index, ready)
    return published
