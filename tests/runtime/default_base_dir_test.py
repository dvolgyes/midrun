"""Tests where Midrun puts its run files for each kind of Lightning logger."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightning.pytorch as PL
import pytest
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.logger import Logger
from torch.utils.data import DataLoader, TensorDataset

from midrun.mixin import MidrunMixin
from midrun.runtime.default_base_dir import default_base_dir

_MARKER_NAME = "midrun-hooks-state.json"
_SERVICES_OFF = {
    "fuse_enabled": False,
    "rest_enabled": False,
    "torch_debug_enabled": False,
}


class _SaveDirLogger(Logger):
    """A logger whose `save_dir` is shared by its runs, like W&B or MLflow."""

    def __init__(self, save_dir: Path, version: Any) -> None:
        super().__init__()
        self._save_dir = str(save_dir)
        self._version = version

    @property
    def name(self) -> str:
        return "shared"

    @property
    def version(self) -> Any:
        return self._version

    @property
    def save_dir(self) -> str:
        return self._save_dir

    def log_metrics(self, metrics: Any, step: Any = None) -> None:
        del metrics, step

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        del params, args, kwargs


class _Model(MidrunMixin, PL.LightningModule):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _fit(model: _Model, root: Path, logger: Any) -> PL.Trainer:
    trainer = PL.Trainer(
        default_root_dir=str(root),
        logger=logger,
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(4, 4), torch.randn(4, 1)), batch_size=4
    )
    trainer.fit(model, loader)
    return trainer


def _session_id(base_dir: Path) -> str:
    return json.loads((base_dir / _MARKER_NAME).read_text())["session_id"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="hooks need bash")
def test_runs_that_share_a_logger_save_dir_get_separate_trees(tmp_path) -> None:
    shared = tmp_path / "shared"
    flag = tmp_path / "fired.txt"

    _fit(_Model(**_SERVICES_OFF), tmp_path / "root", _SaveDirLogger(shared, "run-A"))
    run_a = shared / "midrun" / "run-A"
    script = run_a / "hooks" / "light" / "on_fit_start.sh"
    script.write_text(f'#!/usr/bin/env bash\necho fired >> "{flag}"\n')
    script.chmod(0o755)
    _fit(_Model(**_SERVICES_OFF), tmp_path / "root", _SaveDirLogger(shared, "run-B"))
    run_b = shared / "midrun" / "run-B"

    assert _session_id(run_a) != _session_id(run_b)
    assert not flag.exists()
    assert (run_a / "midrun.log").is_file()
    assert (run_b / "midrun.log").is_file()
    assert not (shared / _MARKER_NAME).exists()
    assert not (shared / "hooks").exists()


def test_a_new_runtime_resumes_the_tree_of_the_same_logger_version(tmp_path) -> None:
    shared = tmp_path / "shared"
    _fit(_Model(**_SERVICES_OFF), tmp_path, _SaveDirLogger(shared, "run-A"))
    first_session = _session_id(shared / "midrun" / "run-A")
    _fit(_Model(**_SERVICES_OFF), tmp_path, _SaveDirLogger(shared, "run-A"))

    assert _session_id(shared / "midrun" / "run-A") == first_session


def test_without_a_logger_one_runtime_keeps_one_tree(tmp_path) -> None:
    model = _Model(**_SERVICES_OFF)
    _fit(model, tmp_path, logger=False)
    base_dir = tmp_path / "midrun" / model.midrun.run_id
    first_session = _session_id(base_dir)
    _fit(model, tmp_path, logger=False)
    other = _Model(**_SERVICES_OFF)
    _fit(other, tmp_path, logger=False)

    assert _session_id(base_dir) == first_session
    assert other.midrun.run_id != model.midrun.run_id
    assert (tmp_path / "midrun" / other.midrun.run_id / _MARKER_NAME).is_file()
    assert not (tmp_path / _MARKER_NAME).exists()


def test_a_per_version_logger_keeps_its_log_dir(tmp_path) -> None:
    trainer = _fit(_Model(**_SERVICES_OFF), tmp_path, CSVLogger(save_dir=str(tmp_path)))

    assert Path(trainer.log_dir) == tmp_path / "lightning_logs" / "version_0"
    assert (Path(trainer.log_dir) / _MARKER_NAME).is_file()
    assert not (tmp_path / "midrun").exists()


class _RankStrategy:
    """Stands in for a DDP strategy whose broadcast returns rank zero's object."""

    def __init__(self, rank_zero_value: str | None) -> None:
        self.rank_zero_value = rank_zero_value
        self.sent: list[Any] = []

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        del src
        self.sent.append(obj)
        return obj if self.rank_zero_value is None else self.rank_zero_value


def _module(rank: int, strategy: _RankStrategy, logger: Any, root: Path) -> Any:
    trainer = SimpleNamespace(
        global_rank=rank,
        strategy=strategy,
        loggers=[logger] if logger is not None else [],
        log_dir=str(logger.save_dir) if logger is not None else str(root),
        default_root_dir=str(root),
    )
    return SimpleNamespace(trainer=trainer)


def test_every_rank_uses_the_directory_that_rank_zero_chose(tmp_path) -> None:
    chosen = str(tmp_path / "midrun" / "run-of-rank-zero")
    strategy = _RankStrategy(rank_zero_value=chosen)

    base_dir = default_base_dir(_module(1, strategy, None, tmp_path), "run-of-rank-one")

    assert base_dir == Path(chosen)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("run-A", "run-A"),
        (7, "7"),
        ("a/b", "a_b"),
        ("..", "run-fallback"),
        ("../../etc", "_.._etc"),
        (".hidden", "hidden"),
        ("x\\y\x00z", "x_y_z"),
        ("", "run-fallback"),
        (None, "run-fallback"),
    ],
)
def test_a_logger_version_becomes_one_safe_path_component(
    tmp_path, version: Any, expected: str
) -> None:
    logger = _SaveDirLogger(tmp_path / "shared", version)
    module = _module(0, _RankStrategy(rank_zero_value=None), logger, tmp_path)

    base_dir = default_base_dir(module, "run-fallback")

    assert base_dir == tmp_path / "shared" / "midrun" / expected
