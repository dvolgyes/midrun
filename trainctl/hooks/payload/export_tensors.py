"""Writes the validated tensors of a heavy hook as .npy files."""

from pathlib import Path

import numpy
import torch

from trainctl.hooks.payload.resolve_export_dtype import resolve_export_dtype
from trainctl.hooks.payload.tensor_leaf import TensorLeaf
from trainctl.hooks.payload.tensor_record import TensorRecord


def export_tensors(
    leaves: list[TensorLeaf], invocation_dir: Path
) -> list[TensorRecord]:
    """Detaches, moves to CPU, converts if needed, and writes each tensor as `.npy`.

    Only called after `prepare_heavy_export` has validated every leaf, so this never
    encounters an unsupported layout/dtype -- writes are all-or-nothing by
    construction, not by rollback. `numpy.save(..., allow_pickle=False)`: no Python
    object, ever, is pickled into these files.
    """
    records: list[TensorRecord] = []
    for index, leaf in enumerate(leaves, start=1):
        export_dtype = resolve_export_dtype(leaf.tensor)
        if export_dtype is None:
            raise AssertionError(f"unvalidated tensor reached export: {leaf.path}")
        # `.numpy()` refuses lazily conjugated or negated views; materialize them.
        cpu_tensor = leaf.tensor.detach().cpu().resolve_conj().resolve_neg()
        if leaf.tensor.dtype is torch.bfloat16:
            cpu_tensor = cpu_tensor.to(torch.float32)
        array = cpu_tensor.numpy()
        filename = f"tensor-{index:06d}.npy"
        numpy.save(invocation_dir / filename, array, allow_pickle=False)
        records.append(
            TensorRecord(
                argument_path=leaf.path,
                file=filename,
                shape=list(leaf.tensor.shape),
                dtype=export_dtype,
                original_dtype=str(leaf.tensor.dtype)
                if leaf.tensor.dtype is torch.bfloat16
                else None,
                device=str(leaf.tensor.device),
            )
        )
    return records
