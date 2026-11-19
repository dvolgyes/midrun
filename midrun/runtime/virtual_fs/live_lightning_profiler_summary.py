"""The content of model/lightning-profiler-summary.txt."""

from typing import TYPE_CHECKING

from midrun.runtime.close_unstarted_recorder import close_unstarted_recorder
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.virtual_fs.encode_text import encode_text
from midrun.runtime.virtual_fs.live_capture_timeout import LIVE_CAPTURE_TIMEOUT_S

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule

_NOT_ENABLED = "lightning profiler not enabled"


def live_lightning_profiler_summary(rt: MidrunRuntime) -> bytes:
    """Reads the summary of the Lightning profiler at the next safe point.

    The Lightning profilers record on the training thread and are not safe to read
    from another thread, so the training thread computes the summary. The next
    Lightning action starts a new recording on the same PyTorchProfiler.

    Raises:
        TimeoutError: The reader did not receive a result within `LIVE_CAPTURE_TIMEOUT_S`.
    """
    if rt.lightning_profiler is None:
        return encode_text(_NOT_ENABLED)

    def compute(pl_module: "LightningModule") -> str:
        """Compute the active profiler summary on the training thread."""
        # A command at the same safe point can change the profiler before this runs.
        profiler = rt.lightning_profiler
        trainer_profiler = pl_module.trainer.profiler  # type: ignore[attr-defined]
        if profiler is None or trainer_profiler is not profiler:
            return _NOT_ENABLED
        if not isinstance(profiler, rt.backend.PL.profilers.PyTorchProfiler):
            summary: str = profiler.summary()
            return summary
        if profiler._schedule is None:
            profiler._profiler_kwargs.pop("schedule", None)
        profiler.profiler = close_unstarted_recorder(profiler.profiler)
        try:
            summary = profiler.summary()
        finally:
            profiler.teardown(stage=pl_module.trainer.state.fn)
            profiler.function_events = None
        return (
            f"{summary}\n\nThis read ended the PyTorch profile; Lightning's next "
            "start recreates its recorder.\n"
        )

    return encode_text(
        rt.live_captures.request(compute, timeout=LIVE_CAPTURE_TIMEOUT_S)
    )
