"""The descriptor of the Lightning namespace that a MidrunMixin subclass uses."""

from dataclasses import dataclass
from types import ModuleType

LIGHTNING_PYTORCH = "lightning.pytorch"
PYTORCH_LIGHTNING = "pytorch_lightning"
KNOWN_FAMILIES = (LIGHTNING_PYTORCH, PYTORCH_LIGHTNING)


@dataclass(frozen=True)
class LightningBackend:
    """The Lightning namespace a MidrunMixin subclass was composed with.

    Attributes:
        name: Backend family identifier (`lightning.pytorch` or `pytorch_lightning`).
        PL: The imported Lightning package module.
        LightningModule: The backend's `LightningModule` base class.
        Trainer: The backend's `Trainer` class.
        callback_class: The backend's `Callback` base class.
        version: The backend package's reported version string.
    """

    name: str
    PL: ModuleType
    LightningModule: type
    Trainer: type
    callback_class: type
    version: str
