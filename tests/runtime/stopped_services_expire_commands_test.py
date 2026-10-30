"""A command still queued when a stage stops expires; it does not run in the next stage."""

import lightning.pytorch as PL
import torch
from torch.utils.data import DataLoader, TensorDataset

from midrun.mixin import MidrunMixin
from midrun.runtime.commands.parse_when import parse_when


class _Model(MidrunMixin, PL.LightningModule):
    def __init__(self) -> None:
        super().__init__(
            fuse_enabled=False,
            rest_enabled=False,
            torch_debug_enabled=False,
            hooks_enabled=False,
        )
        self.layer = torch.nn.Linear(4, 1)
        self.stop_id: str | None = None

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def test_step(self, batch, batch_idx):
        if self.stop_id is None:
            # A training safe point never comes during `test`.
            command = self.midrun.command_queue.submit(
                "stop", {}, parse_when({"safe_point": "train_batch_end"})
            )
            self.stop_id = command.id

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _trainer(tmp_path) -> PL.Trainer:
    return PL.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        limit_train_batches=3,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )


def _loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.randn(12, 4), torch.randn(12, 1)), batch_size=4
    )


def test_a_stop_queued_during_test_does_not_stop_the_next_fit(tmp_path) -> None:
    model = _Model()
    _trainer(tmp_path).test(model, _loader())

    trainer = _trainer(tmp_path)
    trainer.fit(model, _loader())

    assert trainer.global_step == 3
    command = model.midrun.command_queue.get(model.stop_id)
    assert command.status == "expired"
    assert command.error is not None
