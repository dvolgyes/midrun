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
        """Drops the momentum buffers of `parameters`.

        Raises:
            ValueError: `parameters` holds only some of the buffered parameters of a
                fused param group, whose step needs a buffer for all or none of them.
        """
        selected = set(parameters)
        for index, group in enumerate(optimizer.param_groups):
            buffered = [
                parameter
                for parameter in group["params"]
                if optimizer.state.get(parameter, {}).get("momentum_buffer") is not None
            ]
            dropped = [parameter for parameter in buffered if parameter in selected]
            if group.get("fused") and 0 < len(dropped) < len(buffered):
                raise ValueError(
                    f"param_group {index} is fused: reset_momentum must select all "
                    "of its parameters or none"
                )
        reset_count = 0
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            if state is not None and state.pop("momentum_buffer", None) is not None:
                reset_count += 1
        return {"semantic": "momentum_buffer_dropped", "parameters_reset": reset_count}
