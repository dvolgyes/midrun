"""Behavioral test: a spawn-based strategy fails with a message that names the fix.

Lightning's `ddp_spawn` pickles the model to start the worker processes, and the
Midrun runtime owns threads, locks and servers that cannot be pickled.
"""

from pathlib import Path

import lightning.pytorch as PL
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from midrun.mixin import MidrunMixin


class _Model(MidrunMixin, PL.LightningModule):
    def __init__(self) -> None:
        super().__init__(
            rest_port=18540,
            fuse_enabled=False,
            hooks_enabled=False,
            torch_debug_enabled=False,
        )
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_ddp_spawn_fails_with_a_message_that_names_the_ddp_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The spawn launcher writes MASTER_PORT; monkeypatch restores it afterwards.
    monkeypatch.setenv("MASTER_PORT", "0")
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randn(8, 1)), 4)
    trainer = PL.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(TypeError, match="strategy='ddp'"):
        trainer.fit(_Model(), loader)
