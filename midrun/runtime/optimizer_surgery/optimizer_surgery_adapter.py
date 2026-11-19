"""The interface of one optimizer family's momentum reset."""

from collections.abc import Sequence
from typing import Any, Protocol

import torch


class OptimizerSurgeryAdapter(Protocol):
    """One optimizer family's momentum-reset semantics."""

    def supports(self, optimizer: torch.optim.Optimizer) -> bool:
        """Returns whether this adapter knows how to reset `optimizer`'s momentum."""
        ...

    def reset_momentum(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: list[torch.nn.Parameter],
        *,
        optimizers: Sequence[torch.optim.Optimizer] = (),
    ) -> dict[str, Any]:
        """Resets selected first moments; `optimizers` includes all owners of live state."""
        ...
