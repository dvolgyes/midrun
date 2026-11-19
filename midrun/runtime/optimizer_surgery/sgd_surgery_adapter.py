"""The momentum reset for torch.optim.SGD."""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch


def _apply_reset(
    optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter]
) -> dict[str, Any]:
    """Drops the selected buffers after preparation succeeds."""
    reset_count = 0
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if state is not None and state.pop("momentum_buffer", None) is not None:
            reset_count += 1
    return {
        "semantic": "momentum_buffer_dropped",
        "parameters_reset": reset_count,
    }


class SGDSurgeryAdapter:
    """`torch.optim.SGD`: drops the momentum buffer, letting SGD recreate it as needed."""

    def supports(self, optimizer: torch.optim.Optimizer) -> bool:
        """Accepts optimizers with SGD's momentum buffer layout."""
        return isinstance(optimizer, torch.optim.SGD)

    def prepare_reset_momentum(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: list[torch.nn.Parameter],
        *,
        optimizers: Sequence[torch.optim.Optimizer] = (),
    ) -> Callable[[], dict[str, Any]]:
        """Plans to drop the momentum buffers of `parameters`.

        A conditional gradient after a fused reset could recreate only some buffers.

        Raises:
            ValueError: The reset would drop a buffer from a fused param group with
                more than one parameter.
        """
        del optimizers
        selected = set(parameters)
        for index, group in enumerate(optimizer.param_groups):
            if not group.get("fused") or len(group["params"]) <= 1:
                continue
            if any(
                parameter in selected
                and optimizer.state.get(parameter, {}).get("momentum_buffer")
                is not None
                for parameter in group["params"]
            ):
                raise ValueError(
                    f"param_group {index} is fused and has multiple parameters: "
                    "reset_momentum cannot remove its momentum buffers"
                )

        return partial(_apply_reset, optimizer, parameters)
