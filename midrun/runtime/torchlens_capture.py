"""Single controlled TorchLens forward-trace capture.

Capture metadata only and unwrap the process-wide PyTorch hooks after the trace.
"""

import tempfile
from pathlib import Path

import torch


def torchlens_capture(model: torch.nn.Module, batch: object) -> dict[str, str | bytes]:
    """Traces one forward pass and returns a summary and a rendered graph.

    Args:
        model: The composed `LightningModule`.
        batch: Positional input for `model.forward()`, e.g.
            `runtime.current_step.batch`. Must already match `forward`'s own
            signature -- a batch shaped for `training_step` (e.g. an `(x, y)` pair)
            may need unpacking by the caller first.

    Returns:
        `{"summary.txt": <overview text>, "graph.svg": <rendered graph bytes>}`.

    Raises:
        ImportError: torchlens is not installed.
    """
    import torchlens as tl

    trace = tl.trace(
        model,
        batch,
        save="none",
        capture=tl.options.CaptureOptions(unwrap_when_done=True),
    )
    summary_text = trace.summary(view="overview")
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "graph.svg"
        trace.draw(path=str(out_path), open_viewer=False)
        graph_bytes = out_path.read_bytes()
    return {"summary.txt": summary_text, "graph.svg": graph_bytes}
