"""Detects the Lightning family that a class was composed with."""

from midrun.lightning_backend import KNOWN_FAMILIES, LightningBackend
from midrun.load_backend import load_backend


def _family_of(module_name: str) -> str | None:
    for family in KNOWN_FAMILIES:
        if module_name == family or module_name.startswith(family + "."):
            return family
    return None


def detect_lightning_backend_from_mro(mro: tuple[type, ...]) -> LightningBackend:
    """Determines the Lightning family represented anywhere in a class MRO.

    Args:
        mro: The `__mro__` of the class being defined.

    Returns:
        The loaded backend for the single Lightning family found in the MRO.

    Raises:
        TypeError: No known Lightning family is present, or more than one is.
    """
    families: set[str] = set()
    for base in mro:
        family = _family_of(base.__module__)
        if family is not None:
            families.add(family)

    if not families:
        raise TypeError(
            "MidrunMixin must be combined with either lightning.pytorch.LightningModule "
            "or pytorch_lightning.LightningModule"
        )
    if len(families) != 1:
        raise TypeError(
            "A Midrun model cannot mix lightning.pytorch and pytorch_lightning classes "
            f"in the same inheritance tree (found: {sorted(families)})"
        )
    return load_backend(families.pop())
