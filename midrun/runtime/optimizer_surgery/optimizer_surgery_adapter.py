"""The interface of one optimizer family's momentum reset."""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import torch


class OptimizerSurgeryAdapter(Protocol):
    """One optimizer family's momentum-reset semantics."""

    def supports(self, optimizer: torch.optim.Optimizer) -> bool:
        """Returns whether this adapter knows how to reset `optimizer`'s momentum."""
        ...

    def prepare_reset_momentum(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: list[torch.nn.Parameter],
        *,
        optimizers: Sequence[torch.optim.Optimizer] = (),
    ) -> Callable[[], dict[str, Any]]:
        """Plans a reset without mutation; `optimizers` includes all live state owners."""
        ...
