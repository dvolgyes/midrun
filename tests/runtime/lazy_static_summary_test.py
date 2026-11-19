"""Regression tests for static summaries of uninitialized model parameters."""

import json
from pathlib import Path

import torch
from lightning.pytorch import LightningModule, Trainer
from torch.nn.parameter import is_lazy
from torch.utils.data import DataLoader, TensorDataset

from midrun.mixin import MidrunMixin
from midrun.runtime.torchinfo_inspector.basic_summary_json import basic_summary_json
from midrun.runtime.torchinfo_inspector.basic_summary_text import basic_summary_text
from midrun.runtime.virtual_fs.virtual_file_content import virtual_file_content


class _LazyModel(MidrunMixin, LightningModule):
    def __init__(self) -> None:
        super().__init__(
            fuse_enabled=False,
            rest_enabled=False,
            torch_debug_enabled=False,
            hooks_enabled=False,
        )
        first = torch.nn.Linear(2, 2)
        second = torch.nn.Linear(2, 2)
        second.weight = first.weight
        first.bias.requires_grad_(False)
        self.offset = torch.nn.Parameter(torch.zeros(2))
        self.features = torch.nn.Sequential(first, second)
        self.output = torch.nn.LazyLinear(1)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.output(self.features(x) + self.offset)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_static_summary_counts_known_parameters_without_a_forward() -> None:
    model = _LazyModel()

    assert basic_summary_json(model) == {
        "total_params": 10,
        "trainable_params": 8,
        "non_trainable_params": 2,
        "torchinfo_used": True,
        "partial": True,
    }
    text = basic_summary_text(model)
    assert "Total params: 10" in text.splitlines()
    assert "Parameter counts exclude uninitialized parameters." in text.splitlines()
    assert model.forward_calls == 0
    assert is_lazy(model.output.weight)


def test_fit_preserves_the_pre_forward_partial_summary(tmp_path: Path) -> None:
    model = _LazyModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    data = TensorDataset(torch.zeros(4, 2), torch.ones(4, 1))
    trainer.fit(model, DataLoader(data, batch_size=4))

    assert trainer.global_step == 1
    assert model.forward_calls == 1
    assert not is_lazy(model.output.weight)
    content = virtual_file_content(model.midrun, "/model/summary.json")
    assert content is not None
    assert json.loads(content) == {
        "total_params": 10,
        "trainable_params": 8,
        "non_trainable_params": 2,
        "torchinfo_used": True,
        "partial": True,
    }
    assert basic_summary_json(model) == {
        "total_params": 13,
        "trainable_params": 11,
        "non_trainable_params": 2,
        "torchinfo_used": True,
    }
