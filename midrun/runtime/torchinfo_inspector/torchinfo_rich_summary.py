"""The torchinfo summary of a model, traced with a live batch."""

from typing import Any


def torchinfo_rich_summary(pl_module: Any, batch: Any) -> dict[str, str]:
    """A forward-traced summary using the live in-flight `batch` as input.

    Args:
        pl_module: The composed `LightningModule`.
        batch: The batch to trace shapes with, e.g. `runtime.current_step.batch`.

    Returns:
        `{"summary.txt": <text>, "summary.json": <json>}`.

    Raises:
        ImportError: torchinfo is not installed.
    """
    import json

    import torchinfo

    stats = torchinfo.summary(pl_module, input_data=batch, verbose=0)
    payload = {
        "total_params": stats.total_params,
        "trainable_params": stats.trainable_params,
        "total_mult_adds": stats.total_mult_adds,
    }
    return {"summary.txt": str(stats), "summary.json": json.dumps(payload, indent=2)}
