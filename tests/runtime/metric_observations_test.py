"""Sparse Lightning logs keep their original observation across publications."""

from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics import Metric

from midrun import MidrunMixin
from midrun.runtime.state.runtime_snapshot import RuntimeSnapshot


class _ReusedMetric(Metric):
    buffer: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("buffer", torch.zeros(()))

    def update(self, value: torch.Tensor) -> None:
        self.buffer.copy_(value)

    def compute(self) -> torch.Tensor:
        return self.buffer

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.update(value)
        self._forward_cache = self.buffer
        return self.buffer


class _SparseModel(MidrunMixin, pl.LightningModule):
    snapshots: list[RuntimeSnapshot]
    raw: torch.Tensor
    reused: _ReusedMetric
    weight: torch.nn.Parameter

    def __init__(self, root: Path) -> None:
        super().__init__(
            artifact_dir=root,
            rest_enabled=False,
            fuse_enabled=False,
            hooks_enabled=False,
            torch_debug_enabled=False,
        )
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.snapshots = []
        self.reused = _ReusedMetric()
        self.register_buffer("raw", torch.zeros(()))

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        if batch_idx == 0:
            self.log("once", 4.0)
        self.log("each", 4.0)
        self.log("both", batch.float().mean(), on_step=True, on_epoch=True)
        self.reused(batch.mean())
        self.log("custom", self.reused, on_step=True, on_epoch=False)
        self.raw.copy_(batch.mean())
        self.trainer.callback_metrics["raw"] = self.raw
        return self.weight.square()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def on_train_batch_end(
        self, outputs: object, batch: torch.Tensor, batch_idx: int
    ) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.snapshots.append(self.midrun.state.read())


def test_sparse_and_equal_logs_preserve_observation_times(tmp_path: Path) -> None:
    model = _SparseModel(tmp_path)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, DataLoader(torch.tensor([2.0, 6.0, 6.0]), batch_size=1))

    first, second, third = (snapshot.metrics for snapshot in model.snapshots)
    assert first["once"].step == first["each"].step == 0
    assert second["once"] == first["once"]
    assert second["each"].step == 1
    assert second["each"].value == first["each"].value == 4.0
    assert second["each"].updated_at > first["each"].updated_at
    assert [(s["custom"].step, s["custom"].value) for s in (first, second, third)] == [
        (0, 2.0),
        (1, 6.0),
        (2, 6.0),
    ]
    assert [s["raw"].value for s in (first, second, third)] == [2.0, 6.0, 6.0]
    final = model.midrun.state.read().metrics
    assert final["once"] == first["once"]
    assert final["each"] == third["each"]
    assert final["both_step"] == third["both_step"]
    assert final["both"] == final["both_epoch"]
    assert final["both_epoch"].value == float(torch.tensor(14 / 3))
    assert final["both_epoch"].step == 3
