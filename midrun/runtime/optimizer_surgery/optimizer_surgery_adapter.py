"""The interface of one optimizer family's momentum reset."""

from typing import Any, Protocol

import torch


class OptimizerSurgeryAdapter(Protocol):
    """One optimizer family's momentum-reset semantics."""

    def supports(self, optimizer: torch.optim.Optimizer) -> bool:
        """Returns whether this adapter knows how to reset `optimizer`'s momentum."""
        ...

    def reset_momentum(
        self, optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter]
    ) -> dict[str, Any]:
        """Resets first-order momentum for `parameters` in `optimizer`'s state."""
        ...
