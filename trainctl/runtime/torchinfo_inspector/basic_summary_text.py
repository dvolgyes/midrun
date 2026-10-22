"""The static text summary of a model, with or without torchinfo."""

from typing import Any


def basic_summary_text(pl_module: Any) -> str:
    """A static, no-input module/param-tree summary; falls back to `str(pl_module)`.

    Never raises: this backs the always-computed `/model/summary.txt`, so an
    unavailable or failing torchinfo must not change existing behavior.
    """
    try:
        import torchinfo
    except ImportError:
        return str(pl_module)
    try:
        return str(torchinfo.summary(pl_module, verbose=0))
    except Exception:  # noqa: BLE001 -- best-effort cosmetic summary, never fatal
        return str(pl_module)
