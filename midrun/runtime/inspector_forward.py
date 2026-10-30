"""The guard around the forward pass that an inspector runs on the model."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from midrun.lightning_backend import LightningBackend

# The Lightning strategies that shard the parameters, so that a forward pass gathers
# them with collectives. A backend that lacks one of these classes cannot use it.
_SHARDING_STRATEGIES = ("FSDPStrategy", "DeepSpeedStrategy", "ModelParallelStrategy")


def _sharding_reason(
    backend: LightningBackend, trainer: object, model: torch.nn.Module
) -> str | None:
    """Returns why the parameters of `model` are sharded, or `None` if they are not."""
    strategy = getattr(trainer, "strategy", None)
    for name in _SHARDING_STRATEGIES:
        strategy_class = getattr(backend.PL.strategies, name, None)
        if strategy_class is not None and isinstance(strategy, strategy_class):
            return f"the trainer strategy {type(strategy).__name__} shards parameters"
    for param_name, param in model.named_parameters():
        if is_traceable_wrapper_subclass(param):
            return (
                f"parameter {param_name!r} is a {type(param).__name__}, "
                "a sharded or wrapped tensor"
            )
    return None


@contextmanager
def inspector_forward(
    backend: LightningBackend,
    trainer: object,
    model: torch.nn.Module,
    what: str,
    *,
    needs_grad: bool = False,
) -> Iterator[None]:
    """Prepares `model` for an inspector forward pass, and restores it afterwards.

    Only rank 0 runs inspectors. A forward pass on sharded parameters gathers them
    with collectives that the other ranks never join, so the run would hang; the
    guard refuses such a model before any forward pass. Otherwise the body runs in
    eval mode, so dropout is off and batch norm (also SyncBatchNorm, which otherwise
    all-reduces) uses its running statistics, without updating them. The previous
    training flag of each module comes back also when the body raises. Grad is off unless the
    inspector needs the autograd graph.

    Args:
        backend: The Lightning backend whose strategy classes identify sharding.
        trainer: The trainer of `model`; `None` or a trainer without a strategy
            leaves only the parameter check.
        model: The model that the body runs a forward pass on.
        what: The name of the inspector, for the refusal message.
        needs_grad: Whether the body needs autograd to record the graph.

    Raises:
        ValueError: The parameters of `model` are sharded.
    """
    reason = _sharding_reason(backend, trainer, model)
    if reason is not None:
        raise ValueError(
            f"{what} refused: {reason}, and a forward pass on rank 0 alone would "
            "wait on collectives that the other ranks do not join"
        )
    # Per module, as a model can keep a frozen part in eval mode while it trains.
    # `modules()` lists a parent before its children, so a child's own flag wins.
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.enable_grad() if needs_grad else torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.train(training)
