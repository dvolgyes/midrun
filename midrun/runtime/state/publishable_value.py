"""Turns one logged host value into the float that a safe point publishes."""

from typing import SupportsFloat

from torch import Tensor


def publishable_value(value: object) -> float | None:
    """Returns `value` as a float, or None when it is not a real scalar.

    A CUDA tensor goes through `publishable_values` instead, because `float` on it
    makes the training loop wait for the device.

    Args:
        value: A value from `trainer.callback_metrics`, or a learning rate.

    Returns:
        A float, or None when `value` is a multi-element, empty or complex tensor,
        or a value that `float` rejects.
    """
    if isinstance(value, Tensor):
        if value.numel() != 1 or value.is_complex():
            return None
        return float(value)
    if not isinstance(value, str | SupportsFloat):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
