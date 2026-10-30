"""The content of model/summary-live.txt."""

from typing import TYPE_CHECKING, Any

from midrun.runtime import forward_input
from midrun.runtime.torchinfo_inspector.torchinfo_rich_summary import (
    torchinfo_rich_summary,
)
from midrun.runtime.virtual_fs.encode_text import encode_text
from midrun.runtime.virtual_fs.live_capture_timeout import LIVE_CAPTURE_TIMEOUT_S
from midrun.runtime.virtual_fs.require_in_flight_batch import require_in_flight_batch

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime


def live_model_summary(rt: "MidrunRuntime") -> bytes:
    """Traces a torchinfo summary of the model at the next safe point.

    Blocks the calling thread until the training thread runs the capture.

    Raises:
        TimeoutError: No safe point was reached within `LIVE_CAPTURE_TIMEOUT_S`.
        ValueError: The safe point had no in-flight batch.
    """

    def compute(pl_module: Any) -> str:
        batch = require_in_flight_batch(rt, "model/summary-live.txt")
        target = forward_input.resolve(pl_module, batch)
        return torchinfo_rich_summary(pl_module, target)["summary.txt"]

    return encode_text(
        rt.live_captures.request(compute, timeout=LIVE_CAPTURE_TIMEOUT_S)
    )
