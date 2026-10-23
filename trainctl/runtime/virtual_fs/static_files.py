"""The virtual files whose content comes from cheap runtime state."""

from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING

from trainctl.runtime import inspectors_registry
from trainctl.runtime.cuda_memory.cheap_cuda_memory_stats import cheap_cuda_memory_stats
from trainctl.runtime.virtual_fs.encode_json import encode_json
from trainctl.runtime.virtual_fs.encode_text import encode_text

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


def _bool(value: bool) -> str:
    return "true" if value else "false"


README = """Trainctl live filesystem
========================

This is a read-only live view of a PyTorch Lightning run.

state/
    Current values pushed by TrainctlMixin hooks, including
    state/cuda-memory.json (allocator-history config + cheap counters) and
    state/profiler.json (bounded-profiler phase, idle unless one is armed) and
    state/lightning-profiler.json (which of Lightning's own profilers, if any,
    is currently installed as trainer.profiler -- see model/lightning-profiler-
    summary.txt, and POST /commands kind=set_lightning_profiler to change it).
    These reads do not inspect CUDA tensors or synchronously query the Trainer.

meta/
    Process IDs, selected ports and runtime metadata.

model/
    Static and cheap model info: summary.txt/summary.json (torchinfo when
    installed, else a plain str(model) fallback), inspectors.json
    (torchinfo/torchview/torchviz/torchlens availability), and
    tunable-hparams.json (current value/type/dtype/device of every hparam a
    LightningModule opted into live editing via `trainctl_tunable_hparams` --
    see trainctl.runtime.hparam_registry). summary-live.txt and graph.dot are
    lazy: reading them blocks until the training thread reaches its next
    optimizer step, then returns a fresh torchinfo summary / torchview DOT
    source traced against the in-flight batch -- nothing is persisted, so
    incidental traversal (ls, find, indexers) never litters GET /artifacts, and
    graph.dot skips rendering entirely (no Graphviz `dot` binary needed).
    Rendered SVGs (torchview_capture, torchviz_capture) and TorchLens traces
    (torchlens_capture) stay deliberate, persisted POST /commands captures,
    listed under GET /artifacts -- there is no live "latest" pointer for those.

debug/pytorch/
    Status of the (experimental, best-effort) PyTorch distributed debug HTTP
    server. See debug/pytorch/README.md.

CUDA memory snapshots/history recording are plain REST reads (GET
/debug/cuda-memory/snapshot, GET /debug/cuda-memory/history) since they don't
touch the Trainer/model; torchinfo/torchview/torchviz/torchlens captures and
bounded profiling go through POST /commands since they need the in-flight
batch/loss or training-thread-affine state; live-tunable hparams are edited
through POST /commands (kind=set_hparam) for the same reason.

This entire tree is also reachable read-only over REST at GET /files/<path>
(e.g. GET /files/model/summary.txt), for callers that can't mount FUSE (a
browser, for instance). Checkpoints, snapshots and debug captures produced by
REST are real files on the regular filesystem, one level up from this mount
(see meta/artifacts-path), and are reachable over REST at
/artifact-files/<relative path> (the relative path is given as `files_url` in
GET /artifacts/{id}'s response).

Kernel file-content caching is disabled with FUSE direct I/O. Trainctl may
cache internally where it has explicit consistency semantics.

Use REST for pause/continue/checkpoint/snapshot/control operations.
"""


DEBUG_README = """Trainctl PyTorch distributed debug projection
==============================================

status reports whether torch.distributed.debug's experimental HTTP frontend is
running for this run. Proxying its cheap/expensive/profiler endpoints through
this filesystem is not implemented yet; use its HTTP frontend directly via the
URL in status until then.
"""


STATIC_FILES: dict[str, Callable[["TrainctlRuntime"], bytes]] = {
    "/README.md": lambda rt: encode_text(README),
    "/meta/run-id": lambda rt: encode_text(rt.run_id),
    "/meta/started-at": lambda rt: encode_text(str(rt.started_at)),
    "/meta/versions.json": lambda rt: encode_json(
        {"lightning_backend": rt.backend.name, "lightning_version": rt.backend.version}
    ),
    "/meta/lightning/backend": lambda rt: encode_text(rt.backend.name),
    "/meta/lightning/version": lambda rt: encode_text(rt.backend.version),
    "/meta/rest/enabled": lambda rt: encode_text(_bool(rt.config.rest_enabled)),
    "/meta/rest/host": lambda rt: encode_text(
        rt.config.rest_host if rt.rest_url else ""
    ),
    "/meta/rest/port": lambda rt: encode_text(
        str(rt._rest_port) if rt._rest_port else ""
    ),
    "/meta/rest/url": lambda rt: encode_text(rt.rest_url or ""),
    "/meta/pytorch-debug/enabled": lambda rt: encode_text(
        _bool(rt.torch_debug.frontend_url is not None)
    ),
    "/meta/pytorch-debug/port": lambda rt: encode_text(
        str(rt.torch_debug._selected_port or "")
    ),
    "/meta/pytorch-debug/url": lambda rt: encode_text(
        rt.torch_debug.frontend_url or ""
    ),
    "/meta/artifacts-path": lambda rt: encode_text(
        str(rt.artifacts.root) if rt.artifacts else ""
    ),
    "/meta/optimizer-surgery-enabled": lambda rt: encode_text(
        _bool(rt.config.optimizer_surgery_enabled)
    ),
    "/state/snapshot.json": lambda rt: encode_json(asdict(rt.state.read())),
    "/state/status": lambda rt: encode_text(rt.state.read().status),
    "/state/stage": lambda rt: encode_text(rt.state.read().stage or ""),
    "/state/phase": lambda rt: encode_text(rt.state.read().phase),
    "/state/epoch": lambda rt: encode_text(str(rt.state.read().epoch)),
    "/state/global-step": lambda rt: encode_text(str(rt.state.read().global_step)),
    "/state/batch-idx": lambda rt: encode_text(
        str(rt.state.read().batch_idx) if rt.state.read().batch_idx is not None else ""
    ),
    "/state/generation": lambda rt: encode_text(str(rt.state.read().generation)),
    "/state/metrics/index.json": lambda rt: encode_json(
        {key: asdict(metric) for key, metric in rt.state.read().metrics.items()}
    ),
    "/state/control/hold-state": lambda rt: encode_text(rt.state.read().hold_state),
    "/state/control/pending-command-count": lambda rt: encode_text(
        str(rt.state.read().pending_commands)
    ),
    "/state/breakpoint.json": lambda rt: encode_json(
        asdict(exception) if (exception := rt.state.read().exception) else None
    ),
    "/state/dataloader/finite-check.json": lambda rt: encode_json(
        {
            "enabled": rt.finite_guard.enabled,
            "checked_batches": rt.finite_guard.checked_batches,
            "failures": rt.finite_guard.failures,
        }
    ),
    "/state/pipeline/config.json": lambda rt: encode_json(
        {
            "enabled": rt.config.pipeline_pressure_enabled,
            "window": rt.config.pipeline_pressure_window,
        }
    ),
    "/state/pipeline/latest.json": lambda rt: encode_json(
        asdict(latest) if (latest := rt.pipeline.latest()) else None
    ),
    "/state/pipeline/summary.json": lambda rt: encode_json(rt.pipeline.summary()),
    "/model/class": lambda rt: encode_text(rt._model_class_name or ""),
    "/model/hparams.json": lambda rt: encode_json(rt._model_hparams),
    "/model/summary.txt": lambda rt: encode_text(rt._model_summary or ""),
    "/model/summary.json": lambda rt: encode_json(rt._model_summary_json),
    "/model/inspectors.json": lambda rt: encode_json(
        inspectors_registry.availability_snapshot()
    ),
    "/model/tunable-hparams.json": lambda rt: encode_json(
        {name: asdict(value) for name, value in rt.tunable_hparams.items()}
    ),
    "/state/cuda-memory.json": lambda rt: encode_json(
        {**asdict(rt.cuda_memory_history), **cheap_cuda_memory_stats()}
    ),
    "/state/profiler.json": lambda rt: encode_json(
        rt.profiler_run.phase_state() if rt.profiler_run else {"state": "idle"}
    ),
    "/state/lightning-profiler.json": lambda rt: encode_json(
        {"level": rt.lightning_profiler_level}
    ),
    "/model/lightning-profiler-summary.txt": lambda rt: encode_text(
        rt.lightning_profiler.summary()
        if rt.lightning_profiler
        else "lightning profiler not enabled"
    ),
    "/debug/README.md": lambda rt: encode_text(DEBUG_README),
    "/debug/pytorch/README.md": lambda rt: encode_text(DEBUG_README),
    "/debug/pytorch/status": lambda rt: encode_json(
        {"enabled": rt.torch_debug._started, "url": rt.torch_debug.frontend_url}
    ),
}
