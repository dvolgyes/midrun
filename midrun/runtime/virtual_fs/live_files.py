"""The virtual files whose content a safe-point capture makes."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from midrun.runtime.virtual_fs.live_lightning_profiler_summary import (
    live_lightning_profiler_summary,
)
from midrun.runtime.virtual_fs.live_model_graph import live_model_graph
from midrun.runtime.virtual_fs.live_model_summary import live_model_summary

if TYPE_CHECKING:
    from midrun.runtime.runtime import MidrunRuntime

LIVE_FILES: dict[str, Callable[["MidrunRuntime"], bytes]] = {
    "/model/summary-live.txt": live_model_summary,
    "/model/graph.dot": live_model_graph,
    "/model/lightning-profiler-summary.txt": live_lightning_profiler_summary,
}
