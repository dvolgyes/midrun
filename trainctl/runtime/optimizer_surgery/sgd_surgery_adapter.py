"""The momentum reset for torch.optim.SGD."""

from typing import Any

import torch


class SGDSurgeryAdapter:
    """`torch.optim.SGD`: drops the momentum buffer, letting SGD recreate it as needed."""

    def supports(self, optimizer: torch.optim.Optimizer) -> bool:
        return isinstance(optimizer, torch.optim.SGD)

    def reset_momentum(
        self, optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter]
    ) -> dict[str, Any]:
        reset_count = 0
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            if state is not None and state.pop("momentum_buffer", None) is not None:
                reset_count += 1
        return {"semantic": "momentum_buffer_dropped", "parameters_reset": reset_count}
