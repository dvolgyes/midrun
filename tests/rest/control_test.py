"""Behavioral tests for POST /control/pause, /control/continue and /control/stop.

One continue resumes the run however often pause was sent, and a stop ends a held
run without a continue.
"""

import threading
import time
from collections.abc import Callable

import lightning.pytorch as PL
import torch
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader, TensorDataset

from trainctl.mixin import TrainctlMixin
from trainctl.rest.app import build_app


class _EndlessModel(TrainctlMixin, PL.LightningModule):
    def __init__(self) -> None:
        super().__init__(
            rest_enabled=False,
            fuse_enabled=False,
            torch_debug_enabled=False,
            hooks_enabled=False,
        )
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _fit_while(
    model: _EndlessModel, drive: Callable[[TestClient], None], tmp_path
) -> None:
    """Fits `model` in this thread while `drive` sends REST calls from another one."""
    trainer = PL.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1000,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
    )
    errors: list[BaseException] = []
    fit_done = threading.Event()

    def _drive() -> None:
        _wait_until(lambda: model._trainer is not None and trainer.global_step > 0)
        client = TestClient(build_app(model._trainctl))
        try:
            drive(client)
        except BaseException as exc:  # noqa: BLE001 -- reported by the main thread
            errors.append(exc)
        finally:
            # Ends the run, so a failed check cannot leave fit() blocked.
            trainer.should_stop = True
            runtime = model._trainctl
            while not fit_done.wait(0.05):
                for command in runtime.command_queue.list_all():
                    runtime.command_queue.cancel(command.id)
                runtime.gate.release()

    driver = threading.Thread(target=_drive)
    driver.start()
    loader = DataLoader(
        TensorDataset(torch.randn(64, 4), torch.randn(64, 1)), batch_size=4
    )
    try:
        trainer.fit(model, loader)
    finally:
        fit_done.set()
        driver.join(timeout=10)
    if errors:
        raise errors[0]


def test_one_continue_resumes_after_a_double_pause(tmp_path) -> None:
    model = _EndlessModel()
    gate = model._trainctl.gate

    def drive(client: TestClient) -> None:
        first = client.post("/control/pause").json()
        second = client.post("/control/pause").json()
        assert first["queued"] is True
        assert second == {"hold_id": first["hold_id"], "queued": False}
        assert _wait_until(gate.is_held)
        assert client.post("/control/pause").json() == {
            "hold_id": first["hold_id"],
            "queued": False,
        }

        assert client.post("/control/continue").json() == {"released": True}
        step = model.trainer.global_step
        assert _wait_until(lambda: model.trainer.global_step > step + 5)
        assert not gate.is_held()

    _fit_while(model, drive, tmp_path)


def test_stop_ends_a_held_run(tmp_path) -> None:
    model = _EndlessModel()
    gate = model._trainctl.gate
    stopped = threading.Event()

    def drive(client: TestClient) -> None:
        client.post("/control/pause")
        assert _wait_until(gate.is_held)
        later = client.post(
            "/holds", json={"when": {"safe_point": "train_batch_end"}, "timeout_s": 0}
        ).json()
        client.post("/control/stop")
        assert _wait_until(lambda: not gate.is_held(), timeout=5)
        assert client.get(f"/commands/{later['id']}").json()["status"] == "cancelled"
        stopped.set()

    _fit_while(model, drive, tmp_path)
    assert stopped.is_set()


def test_run_shows_the_batch_index_while_held_before_the_optimizer_step(
    tmp_path,
) -> None:
    model = _EndlessModel()
    gate = model._trainctl.gate
    seen: list[object] = []

    def drive(client: TestClient) -> None:
        client.post(
            "/holds",
            json={"when": {"safe_point": "before_optimizer_step"}, "timeout_s": 5},
        )
        assert _wait_until(gate.is_held)
        seen.append(client.get("/run").json()["batch_idx"])
        client.post("/control/continue")

    _fit_while(model, drive, tmp_path)
    assert len(seen) == 1
    assert isinstance(seen[0], int)
