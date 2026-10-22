"""Resolves a REST parameter selector to optimizer parameters."""

from typing import Any

import torch


def resolve_optimizer_parameters(
    pl_module: Any, optimizer: torch.optim.Optimizer, selector: Any
) -> list[torch.nn.Parameter]:
    """Resolves a REST-supplied selector to concrete parameters inside `optimizer`.

    Args:
        pl_module: The composed `LightningModule`, used to map parameters to names.
        optimizer: The target optimizer; only its own `param_groups` are eligible.
        selector: `"all"`, `{"parameter": name}`, `{"parameter_prefix": prefix}`, or
            `{"param_group": index}`.

    Raises:
        ValueError: `selector` has an unknown shape, or matches nothing.
    """
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    if selector == "all":
        return all_params
    if isinstance(selector, dict):
        if "param_group" in selector:
            index = int(selector["param_group"])
            if not 0 <= index < len(optimizer.param_groups):
                raise ValueError(f"optimizer has no param_group {index}")
            return list(optimizer.param_groups[index]["params"])
        names_by_id = {id(p): name for name, p in pl_module.named_parameters()}
        if "parameter" in selector:
            name = selector["parameter"]
            matches = [p for p in all_params if names_by_id.get(id(p)) == name]
            if not matches:
                raise ValueError(f"no optimizer parameter named {name!r}")
            return matches
        if "parameter_prefix" in selector:
            prefix = selector["parameter_prefix"]
            matches = [
                p for p in all_params if names_by_id.get(id(p), "").startswith(prefix)
            ]
            if not matches:
                raise ValueError(f"no optimizer parameter matching prefix {prefix!r}")
            return matches
    raise ValueError(f"invalid momentum-reset selector: {selector!r}")
