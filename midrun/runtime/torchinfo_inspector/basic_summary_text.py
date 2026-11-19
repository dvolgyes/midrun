"""The static text summary of a model, with or without torchinfo."""

from typing import TYPE_CHECKING

from midrun.runtime.torchinfo_inspector.parameter_totals import (
    parameter_totals,
)

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule


def basic_summary_text(pl_module: "LightningModule") -> str:
    """A static, no-input module/param-tree summary; falls back to `str(pl_module)`.

    Never raises: this backs the always-computed `/model/summary.txt`, so an
    unavailable or failing torchinfo must not change existing behavior.
    """
    try:
        import torchinfo
    except ImportError:
        return str(pl_module)
    try:
        stats = torchinfo.summary(pl_module, verbose=0)
        counts = parameter_totals(pl_module.parameters())
        stats.total_params = counts.total
        stats.trainable_params = counts.trainable
        stats.total_param_bytes = counts.total_bytes
        text = str(stats)
        if counts.partial:
            return f"{text}\nParameter counts exclude uninitialized parameters."
        return text
    except Exception:  # noqa: BLE001 -- best-effort cosmetic summary, never fatal
        return str(pl_module)
