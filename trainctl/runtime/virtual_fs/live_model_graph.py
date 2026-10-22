"""The content of model/graph.dot."""

from typing import TYPE_CHECKING, Any

from trainctl.runtime import forward_input
from trainctl.runtime.graph_inspectors.torchview_dot import torchview_dot
from trainctl.runtime.virtual_fs.encode_text import encode_text
from trainctl.runtime.virtual_fs.live_capture_timeout import LIVE_CAPTURE_TIMEOUT_S
from trainctl.runtime.virtual_fs.require_in_flight_batch import require_in_flight_batch

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def live_model_graph(rt: "TrainctlRuntime") -> bytes:
    """Traces the torchview DOT graph of the model at the next safe point.

    Blocks the calling thread until the training thread runs the capture.

    Raises:
        TimeoutError: No safe point was reached within `LIVE_CAPTURE_TIMEOUT_S`.
        ValueError: The safe point had no in-flight batch.
    """

    def compute(pl_module: Any) -> str:
        batch = require_in_flight_batch(rt, "model/graph.dot")
        target = forward_input.resolve(pl_module, batch)
        return torchview_dot(pl_module, target)

    return encode_text(
        rt.live_captures.request(compute, timeout=LIVE_CAPTURE_TIMEOUT_S)
    )
