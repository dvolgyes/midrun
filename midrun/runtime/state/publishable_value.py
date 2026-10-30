"""Turns one logged metric value into the value a safe point publishes."""

from typing import SupportsFloat

from torch import Tensor


def publishable_value(value: object) -> float | Tensor | None:
    """Returns `value` as a publishable scalar without a copy to the host.

    A tensor stays on its device, detached, so the training loop does not wait
    for the device; `host_snapshot` copies it when a reader asks.

    Args:
        value: A value from `trainer.callback_metrics`.

    Returns:
        A detached one-element real tensor, a float, or None when `value` is
        not a real scalar (a multi-element, empty or complex tensor, or a
        value `float` rejects).
    """
    if isinstance(value, Tensor):
        if value.numel() != 1 or value.is_complex():
            return None
        return value.detach()
    if not isinstance(value, str | SupportsFloat):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
