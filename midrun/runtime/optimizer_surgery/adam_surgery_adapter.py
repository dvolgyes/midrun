"""The momentum reset for torch.optim.Adam and AdamW."""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch


def _protected_storages(
    target: torch.optim.Optimizer,
    selected: set[torch.nn.Parameter],
    optimizers: Sequence[torch.optim.Optimizer],
) -> set[torch.UntypedStorage]:
    """Finds dense state buffers outside the first moments selected for reset."""
    return {
        value.untyped_storage()
        for owner in optimizers
        for parameter, state in owner.state.items()
        if isinstance(state, dict)
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and not (owner is target and parameter in selected and name == "exp_avg")
    }


def _apply_reset(
    optimizer: torch.optim.Optimizer,
    replacements: dict[torch.Tensor, dict[str, Any]],
    tensors: list[torch.Tensor],
) -> dict[str, Any]:
    """Installs isolated first moments and zeroes unshared tensors in place."""
    optimizer.state.update(replacements)
    for tensor in tensors:
        tensor.zero_()
    return {
        "semantic": "exp_avg_zeroed",
        "parameters_reset": len(replacements) + len(tensors),
    }


class AdamSurgeryAdapter:
    """Resets Adam/AdamW first moments without writing through shared state.

    `exp_avg_sq`, `max_exp_avg_sq`, and `step` are left untouched -- reset_momentum has
    one stable meaning: reset first-order momentum while retaining variance/step
    history. A separate `reset_variance`/`reset_optimizer_state` operation would be
    needed to touch those.
    """

    def supports(self, optimizer: torch.optim.Optimizer) -> bool:
        """Accepts optimizers with Adam's first-moment state layout."""
        return isinstance(optimizer, torch.optim.Adam | torch.optim.AdamW)

    def prepare_reset_momentum(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: list[torch.nn.Parameter],
        *,
        optimizers: Sequence[torch.optim.Optimizer] = (),
    ) -> Callable[[], dict[str, Any]]:
        """Plans in-place zeroing unless it would change another state entry.

        Raises:
            ValueError: An in-place reset would write to an inference tensor outside inference mode.
        """
        protected = _protected_storages(
            optimizer, set(parameters), optimizers or (optimizer,)
        )
        replacements: dict[torch.Tensor, dict[str, Any]] = {}
        tensors: list[torch.Tensor] = []
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            if state is None or (exp_avg := state.get("exp_avg")) is None:
                continue
            if exp_avg.untyped_storage() in protected:
                replacements[parameter] = {
                    **state,
                    "exp_avg": torch.zeros_like(exp_avg),
                }
            elif exp_avg.is_inference() and not torch.is_inference_mode_enabled():
                raise ValueError(
                    "cannot reset an inference first moment in place outside inference mode"
                )
            else:
                tensors.append(exp_avg)

        return partial(_apply_reset, optimizer, replacements, tensors)
