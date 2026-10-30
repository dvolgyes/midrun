"""CUDA allocator-history recording ends with the stage that turned it on.

`_record_memory_history` is process-global torch state, so the test disables it
again in a `finally`, regardless of outcome.
"""

import lightning.pytorch as PL
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from midrun.mixin import MidrunMixin
from midrun.runtime.cuda_memory.set_cuda_memory_history import (
    set_cuda_memory_history,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


class _Model(MidrunMixin, PL.LightningModule):
    def __init__(self) -> None:
        super().__init__(
            fuse_enabled=False,
            rest_enabled=False,
            torch_debug_enabled=False,
            hooks_enabled=False,
        )
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        if not self.midrun.cuda_memory_history.enabled:
            # As GET /debug/cuda-memory/history does.
            self.midrun.cuda_memory_history = set_cuda_memory_history(
                True, max_entries=None
            )
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _trace_length() -> int:
    return len(torch.cuda.memory._snapshot()["device_traces"][0])


def test_history_recording_stops_when_the_stage_stops(tmp_path) -> None:
    model = _Model()
    trainer = PL.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(4, 4), torch.randn(4, 1)), batch_size=4
    )
    try:
        trainer.fit(model, loader)

        before = _trace_length()
        torch.ones(1024, device="cuda")

        assert _trace_length() == before
        assert model.midrun.cuda_memory_history.enabled is False
    finally:
        set_cuda_memory_history(False, max_entries=None)


def test_history_recording_has_a_finite_default_bound() -> None:
    try:
        state = set_cuda_memory_history(True, max_entries=None)

        assert state.max_entries == 100_000
    finally:
        set_cuda_memory_history(False, max_entries=None)
