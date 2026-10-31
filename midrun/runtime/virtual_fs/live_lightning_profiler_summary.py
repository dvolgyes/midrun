"""The content of model/lightning-profiler-summary.txt."""

from typing import TYPE_CHECKING

from midrun.runtime.virtual_fs.encode_text import encode_text
from midrun.runtime.virtual_fs.live_capture_timeout import LIVE_CAPTURE_TIMEOUT_S

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule

    from midrun.runtime.runtime import MidrunRuntime

_NOT_ENABLED = "lightning profiler not enabled"


def live_lightning_profiler_summary(rt: "MidrunRuntime") -> bytes:
    """Reads the summary of the Lightning profiler at the next safe point.

    The Lightning profilers record on the training thread and are not safe to read
    from another thread, so the training thread computes the summary. With no
    profiler installed, it returns at once. A `PyTorchProfiler` ends its profile
    when it makes a summary, so the read installs a fresh one in its place.

    Raises:
        TimeoutError: No safe point was reached within `LIVE_CAPTURE_TIMEOUT_S`.
    """
    if rt.lightning_profiler is None:
        return encode_text(_NOT_ENABLED)

    def compute(pl_module: "LightningModule") -> str:
        """Returns the summary, and replaces a `PyTorchProfiler` that it ended."""
        # A command at the same safe point can change the profiler before this runs.
        profiler = rt.lightning_profiler
        if profiler is None:
            return _NOT_ENABLED
        summary: str = profiler.summary()
        pytorch_profiler = rt.backend.PL.profilers.PyTorchProfiler
        if not isinstance(profiler, pytorch_profiler):
            return summary
        fresh = pytorch_profiler()
        # Lightning sets Trainer.profiler outside the class, so its type omits it.
        pl_module.trainer.profiler = fresh  # type: ignore[attr-defined]
        rt.lightning_profiler = fresh
        return (
            f"{summary}\n\nThis read ended the PyTorch profile and reset the profiler: "
            "a fresh PyTorchProfiler now records from this safe point.\n"
        )

    return encode_text(
        rt.live_captures.request(compute, timeout=LIVE_CAPTURE_TIMEOUT_S)
    )
