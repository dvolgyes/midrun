"""The momentum reset for torch.optim.Adam and AdamW."""

from typing import Any

import torch


class AdamSurgeryAdapter:
    """`torch.optim.Adam`/`AdamW`: zeroes the first moment (`exp_avg`) in place only.

    `exp_avg_sq`, `max_exp_avg_sq`, and `step` are left untouched -- reset_momentum has
    one stable meaning: reset first-order momentum while retaining variance/step
    history. A separate `reset_variance`/`reset_optimizer_state` operation would be
    needed to touch those.
    """

    _TYPES = (torch.optim.Adam, torch.optim.AdamW)

    def supports(self, optimizer: torch.optim.Optimizer) -> bool:
        return isinstance(optimizer, self._TYPES)

    def reset_momentum(
        self, optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter]
    ) -> dict[str, Any]:
        reset_count = 0
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            exp_avg = state.get("exp_avg") if state is not None else None
            if exp_avg is not None:
                exp_avg.zero_()
                reset_count += 1
        return {"semantic": "exp_avg_zeroed", "parameters_reset": reset_count}
