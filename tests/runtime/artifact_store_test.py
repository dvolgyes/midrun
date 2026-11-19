"""Behavioral tests for MidrunRuntime.artifact_store."""

from pathlib import Path

import lightning.pytorch as PL
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.mixin import MidrunMixin
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.runtime import MidrunRuntime


def _make_runtime() -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    return build_runtime(config, load_backend("lightning.pytorch"))


class _Model(MidrunMixin, PL.LightningModule):
    def __init__(self) -> None:
        super().__init__(
            rest_enabled=False,
            fuse_enabled=False,
            torch_debug_enabled=False,
            hooks_enabled=False,
            run_log_enabled=False,
        )
        self.layer = torch.nn.Linear(2, 1)

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.snapshot_command = self.midrun.command_queue.submit(
            "snapshot", {"kind": "runtime"}, parse_when("now")
        )

    def validation_step(self, batch, batch_idx):
        return self.layer(batch[0]).sum()

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _trainer(root: Path) -> PL.Trainer:
    return PL.Trainer(
        default_root_dir=str(root),
        max_steps=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )


def _tree(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_artifact_store_before_setup_names_the_missing_setup() -> None:
    runtime = _make_runtime()
    with pytest.raises(RuntimeError, match="setup_environment"):
        _ = runtime.artifact_store


def test_artifact_store_returns_the_created_store(tmp_path) -> None:
    runtime = _make_runtime()
    store = ArtifactStore(tmp_path)
    runtime.artifacts = store
    assert runtime.artifact_store is store


def test_failed_stage_store_does_not_use_previous_root_and_recovers(
    tmp_path: Path,
) -> None:
    model = _Model()
    loader = DataLoader(TensorDataset(torch.ones(2, 2)), batch_size=2)
    _trainer(tmp_path / "fit").fit(model, loader)
    assert model.snapshot_command.status == "succeeded"
    old_root = model.midrun.artifact_store.root
    old_tree = _tree(old_root)
    assert old_tree

    validation_dir = tmp_path / "validation"
    new_root = validation_dir / "midrun" / model.midrun.run_id / "artifacts"
    new_root.parent.mkdir(parents=True)
    new_root.touch()
    trainer = _trainer(validation_dir)
    trainer.validate(model, loader, verbose=False)

    assert _tree(old_root) == old_tree
    assert model.snapshot_command.status == "failed"
    assert "artifact store" in model.snapshot_command.error
    assert model.midrun.state.read().artifacts_path is None

    new_root.unlink()
    trainer.validate(model, loader, verbose=False)
    assert model.snapshot_command.status == "succeeded"
    assert model.midrun.state.read().artifacts_path == str(new_root)
    capture = Path(model.snapshot_command.result["path"])
    assert capture.parent == new_root / "snapshots"
    assert (capture / "runtime.json").is_file()
    assert _tree(old_root) == old_tree
