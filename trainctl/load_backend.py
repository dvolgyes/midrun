"""Loads a Lightning backend by its package name."""

import importlib

from trainctl.lightning_backend import KNOWN_FAMILIES, LightningBackend


def load_backend(name: str) -> LightningBackend:
    """Lazily imports and describes one Lightning backend family.

    Args:
        name: `lightning.pytorch` or `pytorch_lightning`.

    Returns:
        The loaded backend descriptor.

    Raises:
        ValueError: `name` is not a known backend family.
    """
    if name not in KNOWN_FAMILIES:
        raise ValueError(f"Unknown Lightning backend {name!r}")
    pl_module = importlib.import_module(name)
    return LightningBackend(
        name=name,
        PL=pl_module,
        LightningModule=pl_module.LightningModule,
        Trainer=pl_module.Trainer,
        callback_class=pl_module.Callback,
        version=getattr(pl_module, "__version__", "unknown"),
    )
