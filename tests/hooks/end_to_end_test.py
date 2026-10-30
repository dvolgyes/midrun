"""Real in-process CPU `Trainer.fit`-based integration tests for the lifecycle
hooks facility (T7): the only test module that exercises `configure_callbacks()`
attachment through an actual Lightning `Trainer`, rather than `HookSession`/
`dispatch_*` directly.
"""

import json
import shlex
import shutil
import sys
from pathlib import Path

import lightning.pytorch as PL
import pytest
import pytorch_lightning as PL2
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from midrun.hooks.manager.hooks_bootstrap_error import HooksBootstrapError
from midrun.mixin import MidrunMixin

_SHELL = shutil.which("bash")
_MARKER_NAME = "midrun-hooks-state.json"


def _make_script(path: Path, body: str, *, executable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755 if executable else 0o644)


def _make_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n, 4)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _read_marker(base_dir: Path) -> dict:
    return json.loads((base_dir / _MARKER_NAME).read_text())


class _TinyModelLP(MidrunMixin, PL.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        x, _y = batch
        self.layer(x)

    def test_step(self, batch, batch_idx):
        x, _y = batch
        self.layer(x)

    def predict_step(self, batch, batch_idx):
        x, _y = batch
        return self.layer(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _TinyModelPL(MidrunMixin, PL2.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        x, _y = batch
        self.layer(x)

    def test_step(self, batch, batch_idx):
        x, _y = batch
        self.layer(x)

    def predict_step(self, batch, batch_idx):
        x, _y = batch
        return self.layer(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _FailingModel(MidrunMixin, PL.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        if batch_idx == 1:
            raise RuntimeError("boom")
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _base_kwargs(**overrides) -> dict:
    kwargs = {
        "fuse_enabled": False,
        "rest_enabled": False,
        "torch_debug_enabled": False,
        "hooks_enabled": True,
        "hooks_timeout_s": 5.0,
    }
    kwargs.update(overrides)
    return kwargs


def _make_trainer(tmp_path: Path, **overrides) -> PL.Trainer:
    kwargs = {
        "default_root_dir": str(tmp_path),
        "max_epochs": 1,
        "limit_train_batches": 2,
        "limit_val_batches": 2,
        "limit_test_batches": 2,
        "limit_predict_batches": 2,
        "num_sanity_val_steps": 0,
        "enable_progress_bar": False,
        "logger": False,
        "enable_checkpointing": False,
    }
    kwargs.update(overrides)
    return PL.Trainer(**kwargs)


@pytest.mark.parametrize(
    ("pl_module", "model_cls"),
    [(PL, _TinyModelLP), (PL2, _TinyModelPL)],
    ids=["lightning.pytorch", "pytorch_lightning"],
)
def test_hooks_fire_during_fit(tmp_path, run_base_dir, pl_module, model_cls) -> None:
    model = model_cls(**_base_kwargs())
    trainer = pl_module.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, _make_loader())

    hooks_dir = run_base_dir(model) / "hooks"
    assert hooks_dir.exists()
    assert len(list((hooks_dir / "light").glob("*.sh"))) == 37
    assert len(list((hooks_dir / "heavy").glob("*.sh"))) == 37
    assert (run_base_dir(model) / _MARKER_NAME).exists()

    captured = tmp_path / "captured.json"
    _make_script(
        hooks_dir / "light" / "on_train_batch_start.sh", f'cp "$1" "{captured}"'
    )
    trainer2 = pl_module.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer2.fit(model, _make_loader())

    assert captured.exists()
    payload = json.loads(captured.read_text())
    assert payload["hook"] == "on_train_batch_start"
    assert payload["modality"] == "light"


def test_sanity_check_hooks_fire(tmp_path, run_base_dir) -> None:
    model = _TinyModelLP(**_base_kwargs())
    bootstrap_trainer = _make_trainer(tmp_path, num_sanity_val_steps=0)
    bootstrap_trainer.fit(model, _make_loader())

    hooks_dir = run_base_dir(model) / "hooks"
    start_marker = tmp_path / "sanity_start.marker"
    end_marker = tmp_path / "sanity_end.marker"
    _make_script(
        hooks_dir / "light" / "on_sanity_check_start.sh", f'touch "{start_marker}"'
    )
    _make_script(
        hooks_dir / "light" / "on_sanity_check_end.sh", f'touch "{end_marker}"'
    )

    trainer = _make_trainer(tmp_path, num_sanity_val_steps=2)
    trainer.fit(model, _make_loader(), val_dataloaders=_make_loader())

    assert start_marker.exists()
    assert end_marker.exists()


def test_validate_standalone_dispatches_validation_hooks(
    tmp_path, run_base_dir
) -> None:
    model = _TinyModelLP(**_base_kwargs())
    trainer = _make_trainer(tmp_path)
    marker = tmp_path / "validated.marker"
    trainer.validate(model, dataloaders=_make_loader())
    hooks_dir = run_base_dir(model) / "hooks"
    _make_script(
        hooks_dir / "light" / "on_validation_epoch_end.sh", f'touch "{marker}"'
    )
    trainer2 = _make_trainer(tmp_path)
    trainer2.validate(model, dataloaders=_make_loader())
    assert marker.exists()


def test_test_standalone_dispatches_test_hooks(tmp_path, run_base_dir) -> None:
    model = _TinyModelLP(**_base_kwargs())
    trainer = _make_trainer(tmp_path)
    trainer.test(model, dataloaders=_make_loader())
    hooks_dir = run_base_dir(model) / "hooks"
    marker = tmp_path / "tested.marker"
    _make_script(hooks_dir / "light" / "on_test_epoch_end.sh", f'touch "{marker}"')
    trainer2 = _make_trainer(tmp_path)
    trainer2.test(model, dataloaders=_make_loader())
    assert marker.exists()


def test_predict_standalone_dispatches_predict_hooks(tmp_path, run_base_dir) -> None:
    model = _TinyModelLP(**_base_kwargs())
    trainer = _make_trainer(tmp_path)
    trainer.predict(model, dataloaders=_make_loader())
    hooks_dir = run_base_dir(model) / "hooks"
    marker = tmp_path / "predicted.marker"
    _make_script(hooks_dir / "light" / "on_predict_epoch_end.sh", f'touch "{marker}"')
    trainer2 = _make_trainer(tmp_path)
    trainer2.predict(model, dataloaders=_make_loader())
    assert marker.exists()


def test_checkpoint_save_and_load_hooks_fire(tmp_path, run_base_dir) -> None:
    # on_save_checkpoint/on_load_checkpoint only dispatch while a stage's hooks
    # session is open (bracketed by that stage's own setup/teardown) -- a
    # standalone Trainer.save_checkpoint() call made after fit() has already
    # returned has no session to dispatch through, by design (there is no run
    # for such a checkpoint to belong to). Both hooks are exercised here through
    # Lightning's own checkpointing machinery, which fires them from *inside* an
    # active fit().
    model = _TinyModelLP(**_base_kwargs())
    trainer = _make_trainer(tmp_path, enable_checkpointing=True)
    trainer.fit(model, _make_loader())

    hooks_dir = run_base_dir(model) / "hooks"
    save_marker = tmp_path / "saved.marker"
    load_marker = tmp_path / "loaded.marker"
    _make_script(
        hooks_dir / "light" / "on_save_checkpoint.sh", f'touch "{save_marker}"'
    )
    _make_script(
        hooks_dir / "light" / "on_load_checkpoint.sh", f'touch "{load_marker}"'
    )

    ckpt_path = tmp_path / "ckpt.ckpt"
    checkpointer = ModelCheckpoint(
        dirpath=str(tmp_path), filename="mid-fit", every_n_train_steps=1, save_top_k=-1
    )
    trainer2 = _make_trainer(
        tmp_path, enable_checkpointing=True, callbacks=[checkpointer]
    )
    trainer2.fit(model, _make_loader())
    assert save_marker.exists()
    shutil.copy(checkpointer.best_model_path, ckpt_path)

    # The same model keeps its run directory, so the load hook script applies.
    trainer3 = _make_trainer(tmp_path)
    trainer3.fit(model, _make_loader(), ckpt_path=str(ckpt_path))
    assert load_marker.exists()


def test_exception_triggers_on_exception_once_and_finalizes(
    tmp_path, run_base_dir
) -> None:
    model = _FailingModel(**_base_kwargs(break_on_backward_exception=False))
    bootstrap_trainer = _make_trainer(tmp_path, limit_train_batches=1)
    bootstrap_trainer.fit(model, _make_loader())

    hooks_dir = run_base_dir(model) / "hooks"
    exception_marker = tmp_path / "exception.marker"
    fit_end_marker = tmp_path / "fit_end.marker"
    _make_script(hooks_dir / "light" / "on_exception.sh", f'touch "{exception_marker}"')
    _make_script(hooks_dir / "light" / "on_fit_end.sh", f'touch "{fit_end_marker}"')

    trainer = _make_trainer(tmp_path, limit_train_batches=4)
    with pytest.raises(RuntimeError, match="boom"):
        trainer.fit(model, _make_loader())

    assert exception_marker.exists()
    assert not fit_end_marker.exists()
    assert list((run_base_dir(model) / ".hooks-tmp").iterdir()) == []
    assert model.midrun._hook_session is None
    assert model.midrun._logging_session is None


def test_sequential_fit_then_test_reuses_hooks_tree_and_adapter_instance(
    tmp_path, run_base_dir
) -> None:
    model = _TinyModelLP(**_base_kwargs())
    trainer = _make_trainer(tmp_path)
    trainer.fit(model, _make_loader())

    base_dir = run_base_dir(model)
    hooks_dir = base_dir / "hooks"
    marker_before = _read_marker(base_dir)
    first_adapter = model._midrun_hooks_adapter
    assert first_adapter is not None

    custom_light = hooks_dir / "light" / "on_test_batch_start.sh"
    fired_log = tmp_path / "fired.log"
    _make_script(custom_light, f'echo run >> "{fired_log}"')

    trainer2 = _make_trainer(tmp_path)
    trainer2.test(model, dataloaders=_make_loader())

    marker_after = _read_marker(base_dir)
    assert marker_after == marker_before
    assert model._midrun_hooks_adapter is first_adapter
    assert custom_light.read_text().startswith("#!/usr/bin/env bash")
    assert fired_log.exists()
    assert len(fired_log.read_text().splitlines()) >= 1


def test_a_resumed_run_dispatches_with_the_marker_session_id(tmp_path) -> None:
    # A new runtime resumes the tree only through a stable logger version.
    def resumable_trainer() -> PL.Trainer:
        return _make_trainer(tmp_path, logger=CSVLogger(tmp_path, version="resumed"))

    model = _TinyModelLP(**_base_kwargs())
    resumable_trainer().fit(model, _make_loader())
    base_dir = tmp_path / "lightning_logs" / "resumed"
    session_ids = tmp_path / "session_ids.log"
    read_id = 'import json, sys; print(json.load(open(sys.argv[1]))["session_id"])'
    _make_script(
        base_dir / "hooks" / "light" / "on_train_start.sh",
        f'{shlex.quote(sys.executable)} -c {shlex.quote(read_id)} "$1" >> "{session_ids}"',
    )

    resumable_trainer().fit(model, _make_loader())
    resumable_trainer().fit(_TinyModelLP(**_base_kwargs()), _make_loader())

    marker_id = _read_marker(base_dir)["session_id"]
    assert session_ids.read_text().splitlines() == [marker_id, marker_id]


def test_user_callback_preserved_alongside_midrun_adapter(tmp_path) -> None:
    started: list[bool] = []

    class _UserCallback(PL.Callback):
        def on_train_start(self, trainer, pl_module) -> None:
            started.append(True)

    model = _TinyModelLP(**_base_kwargs())
    trainer = _make_trainer(tmp_path, callbacks=[_UserCallback()])
    trainer.fit(model, _make_loader())

    assert started == [True]
    assert any(isinstance(cb, _UserCallback) for cb in trainer.callbacks)
    assert any(type(cb).__name__ == "MidrunHooksCallback" for cb in trainer.callbacks)


def test_subclass_configure_callbacks_override_preserves_both(tmp_path) -> None:
    marker: list[bool] = []

    class _OwnCallback(PL.Callback):
        def on_train_start(self, trainer, pl_module) -> None:
            marker.append(True)

    class _ModelWithOwnCallback(_TinyModelLP):
        def configure_callbacks(self):
            return [*super().configure_callbacks(), _OwnCallback()]

    model = _ModelWithOwnCallback(**_base_kwargs())
    trainer = _make_trainer(tmp_path)
    trainer.fit(model, _make_loader())

    assert marker == [True]
    assert any(isinstance(cb, _OwnCallback) for cb in trainer.callbacks)
    assert any(type(cb).__name__ == "MidrunHooksCallback" for cb in trainer.callbacks)


def test_failed_hooks_bootstrap_warns_and_trains_with_hooks_off(
    tmp_path, run_base_dir
) -> None:
    model = _TinyModelLP(**_base_kwargs())
    base_dir = run_base_dir(model)
    base_dir.mkdir(parents=True)
    marker = base_dir / _MARKER_NAME
    marker.write_text("garbage")

    trainer = _make_trainer(tmp_path)
    trainer.fit(model, _make_loader())

    assert trainer.global_step == 2
    assert marker.read_text() == "garbage"
    assert not (base_dir / "hooks").exists()
    run_log = (base_dir / "midrun.log").read_text()
    assert "hooks are off for this stage" in run_log
    assert "delete it and the hooks" in run_log


def test_no_bash_warns_and_trains_with_hooks_off(
    tmp_path, run_base_dir, monkeypatch
) -> None:
    monkeypatch.setattr(
        "midrun.runtime.stage_sessions.shutil.which", lambda _name: None
    )
    model = _TinyModelLP(**_base_kwargs())

    trainer = _make_trainer(tmp_path)
    trainer.fit(model, _make_loader())

    assert trainer.global_step == 2
    assert not (run_base_dir(model) / "hooks").exists()
    assert "requires bash on PATH" in (run_base_dir(model) / "midrun.log").read_text()


def test_failed_hooks_bootstrap_under_strict_mode_raises_and_a_later_fit_runs(
    tmp_path, run_base_dir
) -> None:
    model = _TinyModelLP(**_base_kwargs(inspection_strict=True))
    base_dir = run_base_dir(model)
    base_dir.mkdir(parents=True)
    marker = base_dir / _MARKER_NAME
    marker.write_text("garbage")

    for _ in range(2):
        with pytest.raises(HooksBootstrapError, match="delete it and the hooks"):
            _make_trainer(tmp_path).fit(model, _make_loader())
    assert model.midrun._logging_session is None

    marker.unlink()
    _make_trainer(tmp_path).fit(model, _make_loader())
    assert _read_marker(base_dir)["session_id"]
