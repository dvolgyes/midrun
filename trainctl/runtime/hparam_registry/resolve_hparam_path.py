"""Resolves a dotted hparam path to its parent object and leaf name."""

from typing import Any

from trainctl.runtime.hparam_registry.get_field import get_field


def resolve_hparam_path(pl_module: Any, trainer: Any, path: str) -> tuple[Any, str]:
    """Walks a dotted path to its parent object and leaf name.

    A `"trainer."` prefix roots the walk at `trainer`; otherwise it roots at
    `pl_module` (e.g. `"hparams.augmentation_strength"` reads
    `pl_module.hparams.augmentation_strength`). Each intermediate segment may be
    an attribute or a dict key (see `get_field`), so a path may walk through a
    nested dict-of-dicts as well as plain attribute chains.
    """
    if path.startswith("trainer."):
        obj: Any = trainer
        parts = path.removeprefix("trainer.").split(".")
    else:
        obj = pl_module
        parts = path.split(".")
    for part in parts[:-1]:
        obj = get_field(obj, part)
    return obj, parts[-1]
