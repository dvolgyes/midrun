"""A training step with logged GPU metrics and a GPU learning rate does not wait for the device."""

from pathlib import Path

import lightning.pytorch as PL
import pytest
import torch
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader

from midrun.mixin import MidrunMixin
from midrun.rest.app import build_app

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

_METRICS = 4


class _Model(MidrunMixin, PL.LightningModule):
    def __init__(self) -> None:
        super().__init__(
            rest_enabled=False,
            fuse_enabled=False,
            torch_debug_enabled=False,
            hooks_enabled=False,
        )
        self.layer = torch.nn.Linear(8, 1)

    def training_step(self, batch, batch_idx):
        del batch, batch_idx
        loss = self.layer(torch.randn(16, 8, device=self.device)).pow(2).mean()
        for index in range(_METRICS):
            self.log(f"m{index}", loss.detach() * 0 + index, on_step=True)
        return loss

    def configure_optimizers(self):
        lr = torch.tensor(0.125, device=self.device)
        return torch.optim.Adam(self.parameters(), lr=lr, capturable=True)


class _SyncsRaise(PL.Callback):
    """Makes each device sync raise from the second batch to the end of the epoch.

    Lightning calls callback hooks before the module's hooks, so the mixin's
    hooks of each of these batches run in the strict mode.
    """

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 1:
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.set_sync_debug_mode("default")


def test_safe_points_do_not_copy_metrics_or_learning_rates_to_the_host(
    tmp_path: Path,
) -> None:
    model = _Model()
    trainer = PL.Trainer(
        default_root_dir=str(tmp_path),
        max_steps=6,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        accelerator="gpu",
        devices=1,
        callbacks=[_SyncsRaise()],
    )
    # Integer batches: Lightning has no batch to copy to the device.
    loader = DataLoader(range(6), batch_size=None)

    try:
        trainer.fit(model, loader)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    client = TestClient(build_app(model.midrun), base_url="http://localhost")
    metrics = client.get("/metrics").json()
    assert {key: metrics[key]["value"] for key in metrics} == {
        f"m{index}": float(index) for index in range(_METRICS)
    }
    assert client.get("/optimizers").json() == {"learning_rates": [0.125]}
