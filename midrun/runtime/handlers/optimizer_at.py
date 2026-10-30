"""Selects one optimizer of a trainer by index."""

from collections.abc import Sequence


def optimizer_at[T](optimizers: Sequence[T], index: int) -> T:
    """Returns `optimizers[index]`.

    Raises:
        ValueError: `index` names no optimizer.
    """
    if index >= len(optimizers):
        raise ValueError(
            f"optimizer {index} does not exist; the run has {len(optimizers)} optimizer(s)"
        )
    return optimizers[index]
