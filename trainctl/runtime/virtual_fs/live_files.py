"""The virtual files whose content a safe-point capture makes."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from trainctl.runtime.virtual_fs.live_model_graph import live_model_graph
from trainctl.runtime.virtual_fs.live_model_summary import live_model_summary

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime

LIVE_FILES: dict[str, Callable[["TrainctlRuntime"], bytes]] = {
    "/model/summary-live.txt": live_model_summary,
    "/model/graph.dot": live_model_graph,
}
