"""A command still queued when a stage stops expires; it does not run in the next stage."""

import httpx2
import lightning.pytorch as PL
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from midrun.mixin import MidrunMixin
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.events import SafePoint


class _Model(MidrunMixin, PL.LightningModule):
    def __init__(self, *, rest_enabled=False, **kwargs) -> None:
        super().__init__(
            fuse_enabled=False,
            rest_enabled=rest_enabled,
            torch_debug_enabled=False,
            **kwargs,
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


def test_rest_refuses_commands_after_expiry_before_shutdown(tmp_path, monkeypatch):
    model = _Model(rest_enabled=True, hooks_enabled=False)
    runtime = model.midrun
    stop_rest = runtime.services.stop_rest
    responses = []

    def request_then_stop():
        try:
            with httpx2.Client(base_url=runtime.rest_url, timeout=5) as client:
                responses.extend(
                    client.post(path) for path in ("/control/stop", "/control/pause")
                )
        finally:
            stop_rest()

    with monkeypatch.context() as patch:
        patch.setattr(runtime.services, "stop_rest", request_then_stop)
        _trainer(tmp_path).test(model, _loader())
    assert [r.status_code for r in responses] == [409, 409]
    assert runtime.command_queue.get(model.stop_id).status == "expired"

    first_stop = model.stop_id
    model.stop_id = None
    _trainer(tmp_path).test(model, _loader())
    assert model.stop_id != first_stop
    assert runtime.command_queue.get(model.stop_id).status == "expired"
    trainer = _trainer(tmp_path)
    trainer.fit(model, _loader())
    assert trainer.global_step == 3


class _OtherEnabled(_Model):
    pass


class _Plain(PL.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def test_step(self, batch, batch_idx):
        pass


def test_plain_model_trains_after_enabled_fit_on_same_trainer(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.fit(_Model(), _loader())
    previous_steps = trainer.global_step
    trainer.fit_loop.max_epochs = trainer.current_epoch + 1
    plain = _Plain()
    trainer.fit(plain, _loader())
    assert trainer.global_step == previous_steps + 3
    assert plain.weight.item() == pytest.approx(0.8**3)


@pytest.mark.parametrize("next_model", ["plain", "disabled", "other_enabled"])
def test_reused_trainer_dispatches_only_for_owning_model(
    tmp_path, next_model, monkeypatch
):
    trainer = _trainer(tmp_path)
    first = _Model()
    trainer.fit(first, _loader())
    first_callbacks = first.configure_callbacks()
    second = (
        _Plain()
        if next_model == "plain"
        else _OtherEnabled(midrun_enabled=next_model == "other_enabled")
    )
    calls = []
    for model in [first, second]:
        if isinstance(model, MidrunMixin):
            dispatch = model.midrun.dispatch_hook

            def record(trainer, hook, *, _model=model, _dispatch=dispatch, **kwargs):
                calls.append((_model, hook))
                return _dispatch(trainer, hook, **kwargs)

            monkeypatch.setattr(model.midrun, "dispatch_hook", record)
    trainer.test(second, _loader())
    if next_model == "other_enabled":
        assert calls.count((second, "setup")) == 1
        assert calls.count((second, "on_test_batch_end")) == 1
        assert all(model is second for model, _ in calls)
    else:
        assert not calls
    for callback in first_callbacks:
        assert sum(type(c) is type(callback) for c in trainer.callbacks) == 1
    assert first.midrun._logging_session is None


@pytest.mark.parametrize("status", ["finished", "failed"])
def test_terminal_transition_expires_queued_command_and_publishes_zero(
    tmp_path, status
):
    model = _Model(artifact_dir=tmp_path / "artifacts")
    model.trainer = _trainer(tmp_path)
    runtime = model.midrun
    runtime.setup_environment(model, "fit")
    runtime.on_fit_start(model)
    command = runtime.command_queue.submit(
        "stop", {}, parse_when({"safe_point": "train_batch_end"})
    )
    runtime.process_safe_point(model, SafePoint.TRAIN_EPOCH_END)

    assert runtime.state.read().pending_commands == 1

    if status == "finished":
        runtime.teardown(model, "fit")
    else:
        runtime.stop_after_failure()

    assert command.status == "expired"
    snapshot = runtime.state.read()
    assert snapshot.status == status
    assert snapshot.pending_commands == 0
