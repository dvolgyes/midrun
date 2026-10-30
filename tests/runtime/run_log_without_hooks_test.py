"""The run log and the Lightning logging bridge work with the shell hooks disabled."""

import logging
from pathlib import Path

import lightning.pytorch as PL
import pytest
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from midrun.logging.loguru_bridge_handler import LoguruBridgeHandler
from midrun.mixin import MidrunMixin
from midrun.runtime.runtime import MidrunRuntime

_FIT_STOPPED = "`Trainer.fit` stopped"


class _TinyModel(MidrunMixin, PL.LightningModule):
    def __init__(self, fail: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layer = torch.nn.Linear(4, 1)
        self.fail = fail

    def training_step(self, batch, batch_idx):
        if self.fail:
            raise RuntimeError("boom")
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _fit(tmp_path: Path, **kwargs) -> PL.Trainer:
    options = {
        "fuse_enabled": False,
        "rest_enabled": False,
        "torch_debug_enabled": False,
        "hooks_enabled": False,
    }
    model = _TinyModel(**(options | kwargs))
    trainer = PL.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(8, 4), torch.randn(8, 1)), batch_size=4
    )
    trainer.fit(model, loader)
    return trainer


def _is_bridged() -> bool:
    std_logger = logging.getLogger("lightning.pytorch")
    return any(isinstance(h, LoguruBridgeHandler) for h in std_logger.handlers)


def test_a_fit_without_hooks_writes_lightning_logging_to_midrun_log(tmp_path) -> None:
    trainer = _fit(tmp_path, run_log_enabled=True, intercept_lightning_logging=True)

    assert _FIT_STOPPED in (Path(trainer.log_dir) / "midrun.log").read_text()
    assert not _is_bridged()


def test_midrun_log_records_midrun_diagnostics_of_unbound_loggers(tmp_path) -> None:
    # Without torch.distributed, setup logs the skip of the debug server unbound.
    trainer = _fit(tmp_path, run_log_enabled=True, torch_debug_enabled=True)

    log = (Path(trainer.log_dir) / "midrun.log").read_text()
    assert "pytorch distributed debug server skipped" in log


def test_a_fit_without_hooks_bridges_lightning_logging_without_a_run_log(
    tmp_path,
) -> None:
    records: list[dict] = []
    sink_id = logger.add(lambda message: records.append(message.record), level=0)
    try:
        trainer = _fit(
            tmp_path, run_log_enabled=False, intercept_lightning_logging=True
        )
    finally:
        logger.remove(sink_id)

    bridged = [r for r in records if "std_logger_name" in r["extra"]]
    assert any(_FIT_STOPPED in r["message"] for r in bridged)
    assert not (Path(trainer.log_dir) / "midrun.log").exists()
    assert not _is_bridged()


def test_a_failed_fit_without_hooks_closes_the_logging_session(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        _fit(
            tmp_path,
            fail=True,
            break_on_backward_exception=False,
            run_log_enabled=True,
            intercept_lightning_logging=True,
        )

    assert (tmp_path / "midrun.log").exists()
    assert not _is_bridged()


def test_midrun_log_records_warnings_of_the_service_shutdown(
    tmp_path, monkeypatch
) -> None:
    def fail() -> None:
        raise OSError("unmount refused")

    monkeypatch.setattr(MidrunRuntime, "_stop_fuse", lambda self: fail())
    trainer = _fit(tmp_path, run_log_enabled=True)

    log = (Path(trainer.log_dir) / "midrun.log").read_text()
    assert "FUSE shutdown failed: unmount refused" in log
