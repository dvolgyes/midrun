"""Dumps the CUDA allocator snapshot."""

import tempfile
from pathlib import Path

import torch


def dump_cuda_memory_snapshot() -> bytes:
    """Dumps the current CUDA allocator snapshot and returns its pickle bytes.

    Raises:
        RuntimeError: CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_memory snapshot requires a CUDA device")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "snapshot.pickle"
        torch.cuda.memory._dump_snapshot(str(path))
        return path.read_bytes()
