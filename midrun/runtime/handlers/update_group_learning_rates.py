"""Updates the learning rates of selected optimizer parameter groups."""

import math
from typing import Any

import torch


def _stored_rate(lr: torch.Tensor | float, value: float, index: int) -> float:
    """Convert the rate without modifying an optimizer group.

    Raises:
        ValueError: The conversion produces a nonfinite rate or loses a positive rate.
    """
    actual = float(lr.new_tensor(value)) if isinstance(lr, torch.Tensor) else value
    if not math.isfinite(actual) or (value > 0 and actual == 0):
        raise ValueError(
            f"learning rate {value} cannot be represented by param_group {index}"
        )
    return actual


@torch.no_grad()
def update_group_learning_rates(
    groups: list[tuple[int, dict[str, Any]]],
    value: float,
    other_storages: set[torch.UntypedStorage],
) -> dict[str, dict[str, float]]:
    """Prepare and apply the selected rates without modifying unselected storage.

    Raises:
        ValueError: A rate cannot be represented or updated in place.
    """
    changes = {
        str(index): {
            "before": float(group["lr"]),
            "after": _stored_rate(group["lr"], value, index),
        }
        for index, group in groups
    }
    replacements: dict[int, torch.Tensor] = {}
    for index, group in groups:
        lr = group["lr"]
        if not isinstance(lr, torch.Tensor):
            continue
        if other_storages and lr.untyped_storage() in other_storages:
            replacement = lr.clone()
            replacement.fill_(changes[str(index)]["after"])
            replacements[index] = replacement.requires_grad_(lr.requires_grad)
        elif lr.is_inference() and not torch.is_inference_mode_enabled():
            raise ValueError(
                f"cannot update inference tensor in param_group {index} in place"
            )

    for index, group in groups:
        lr = group["lr"]
        after = changes[str(index)]["after"]
        if index in replacements:
            group["lr"] = replacements[index]
        elif isinstance(lr, torch.Tensor):
            lr.fill_(after)
        else:
            group["lr"] = after
    return changes
