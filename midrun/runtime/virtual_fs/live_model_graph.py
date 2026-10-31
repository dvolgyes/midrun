"""The content of model/graph.dot."""

from typing import TYPE_CHECKING

from midrun.runtime.graph_inspectors.torchview_dot import torchview_dot
from midrun.runtime.run_on_in_flight_batch import run_on_in_flight_batch
from midrun.runtime.virtual_fs.encode_text import encode_text
from midrun.runtime.virtual_fs.live_capture_timeout import LIVE_CAPTURE_TIMEOUT_S

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule

    from midrun.runtime.runtime import MidrunRuntime


def live_model_graph(rt: "MidrunRuntime") -> bytes:
    """Traces the torchview DOT graph of the model at the next safe point.

    Blocks the calling thread until the training thread runs the capture.

    Raises:
        TimeoutError: No safe point was reached within `LIVE_CAPTURE_TIMEOUT_S`.
        ValueError: The safe point had no in-flight batch, or the parameters are
            sharded.
    """

    def compute(pl_module: "LightningModule") -> str:
        return run_on_in_flight_batch(
            lambda target: torchview_dot(pl_module, target),
            rt,
            pl_module.trainer,
            pl_module,
            "model/graph.dot",
        )

    return encode_text(
        rt.live_captures.request(compute, timeout=LIVE_CAPTURE_TIMEOUT_S)
    )
