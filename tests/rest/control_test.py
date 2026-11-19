"""Behavioral tests for POST /control/pause, /control/continue and /control/stop.

One continue resumes the run however often pause was sent, and a stop ends a held
run without a continue.
"""

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import lightning.pytorch as PL
import pytest
import torch
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader, TensorDataset

from midrun.mixin import MidrunMixin
from midrun.rest.app import build_app
from midrun.runtime.events import SafePoint
from midrun.runtime.gate import ActiveHold


class _EndlessModel(MidrunMixin, PL.LightningModule):
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
        client = TestClient(build_app(model.midrun), base_url="http://localhost")
        try:
            drive(client)
        except BaseException as exc:  # noqa: BLE001 -- reported by the main thread
            errors.append(exc)
        finally:
            # Ends the run, so a failed check cannot leave fit() blocked.
            trainer.should_stop = True
            runtime = model.midrun
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
    gate = model.midrun.gate

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

        assert client.post("/control/continue").json() == {
            "released": True,
            "cancelled_holds": [],
        }
        step = model.trainer.global_step
        assert _wait_until(lambda: model.trainer.global_step > step + 5)
        assert not gate.is_held()

    _fit_while(model, drive, tmp_path)


def test_stop_ends_a_held_run(tmp_path) -> None:
    model = _EndlessModel()
    gate = model.midrun.gate
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
    gate = model.midrun.gate
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


class _StepTrainer:
    current_epoch = 0
    global_step = 0
    should_stop = False


def test_a_pause_after_a_stop_in_the_same_step_does_not_hold(tmp_path) -> None:
    runtime = _EndlessModel().midrun
    trainer = _StepTrainer()
    client = TestClient(build_app(runtime), base_url="http://localhost")
    stop = client.post("/control/stop").json()
    hold_id = client.post("/control/pause").json()["hold_id"]

    held = runtime._run_due_commands(
        None, trainer, SafePoint.TRAIN_BATCH_END, allowed=None
    )

    assert client.get(f"/commands/{stop['id']}").json()["result"][
        "cancelled_holds"
    ] == [hold_id]
    assert client.get(f"/commands/{hold_id}").json()["status"] == "cancelled"
    assert not held
    assert not runtime.gate.is_held()


def test_a_continue_before_the_pause_is_due_cancels_the_pause() -> None:
    runtime = _EndlessModel().midrun
    client = TestClient(build_app(runtime), base_url="http://localhost")
    later = client.post(
        "/holds", json={"when": {"safe_point": "train_epoch_end"}, "timeout_s": 0}
    ).json()
    hold_id = client.post("/control/pause").json()["hold_id"]

    resumed = client.post("/control/continue").json()
    held = runtime._run_due_commands(
        None, _StepTrainer(), SafePoint.TRAIN_BATCH_END, allowed=None
    )

    assert not held
    assert client.get(f"/commands/{hold_id}").json()["status"] == "cancelled"
    assert client.get(f"/commands/{later['id']}").json()["status"] == "queued"
    assert resumed == {"released": False, "cancelled_holds": [hold_id]}
    assert client.post("/control/pause").json()["queued"] is True


def test_continue_racing_pause_activation_releases_the_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _EndlessModel().midrun
    gate = runtime.gate
    entered_dispatch = threading.Event()
    allow_activation = threading.Event()
    gate_lock = gate._lock
    enter = gate.enter

    class ObservedLock:
        def __enter__(self) -> None:
            if not gate_lock.acquire(blocking=False):
                allow_activation.set()
                assert gate_lock.acquire(timeout=10)

        def __exit__(self, *_: object) -> None:
            gate_lock.release()

    def delayed_enter(
        command_id: str, safe_point: str, epoch: int, step: int
    ) -> ActiveHold:
        entered_dispatch.set()
        assert allow_activation.wait(timeout=10)
        return enter(command_id, safe_point, epoch, step)

    monkeypatch.setattr(gate, "_lock", ObservedLock())
    monkeypatch.setattr(gate, "enter", delayed_enter)

    with TestClient(build_app(runtime), base_url="http://localhost") as client:
        hold_id = client.post("/control/pause").json()["hold_id"]
        with ThreadPoolExecutor(max_workers=1) as executor:
            dispatched = executor.submit(
                runtime._run_due_commands,
                None,
                _StepTrainer(),
                SafePoint.TRAIN_BATCH_END,
                None,
            )
            try:
                assert entered_dispatch.wait(timeout=10)
                resumed = client.post("/control/continue").json()
            finally:
                allow_activation.set()
            dispatched.result(timeout=10)

        assert resumed == {"released": True, "cancelled_holds": []}
        assert client.get(f"/commands/{hold_id}").json()["status"] == "succeeded"
        assert not gate.is_held()


def test_continue_leaves_a_hold_with_the_same_reason_queued() -> None:
    runtime = _EndlessModel().midrun
    client = TestClient(build_app(runtime), base_url="http://localhost")
    explicit = client.post("/holds", json={"reason": "control/pause", "timeout_s": 0})
    explicit_id = explicit.json()["id"]
    pause = client.post("/control/pause").json()
    pause_id = pause["hold_id"]
    assert pause["queued"]
    assert pause_id != explicit_id

    assert client.post("/control/continue").json() == {
        "released": False,
        "cancelled_holds": [pause_id],
    }
    assert client.get(f"/commands/{explicit_id}").json()["status"] == "queued"
    assert client.get(f"/commands/{pause_id}").json()["status"] == "cancelled"


def test_a_hold_after_an_executed_stop_is_refused(tmp_path) -> None:
    runtime = _EndlessModel().midrun
    client = TestClient(build_app(runtime), base_url="http://localhost")
    client.post("/control/stop")
    runtime._run_due_commands(None, _StepTrainer(), SafePoint.TRAIN_BATCH_END, None)

    responses = [
        client.post("/control/pause"),
        client.post("/holds", json={"timeout_s": 0}),
        client.post("/commands", json={"kind": "hold", "args": {"reason": "x"}}),
    ]

    assert [response.status_code for response in responses] == [409] * 3
    assert [c.kind for c in runtime.command_queue.list_all()] == ["stop"]


def test_a_new_run_accepts_holds_after_a_stopped_run(tmp_path) -> None:
    model = _EndlessModel()
    client = TestClient(build_app(model.midrun), base_url="http://localhost")
    loader = DataLoader(
        TensorDataset(torch.randn(64, 4), torch.randn(64, 1)), batch_size=4
    )

    stop = client.post("/control/stop").json()
    PL.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1000,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
    ).fit(model, loader)
    assert client.get(f"/commands/{stop['id']}").json()["status"] == "succeeded"
    assert client.post("/control/pause").status_code == 409

    def pause_then_continue(live_client: TestClient) -> None:
        response = live_client.post("/control/pause")
        assert response.status_code == 200, response.text
        assert response.json()["queued"] is True
        assert _wait_until(model.midrun.gate.is_held)
        step = model.trainer.global_step
        assert live_client.post("/control/continue").json()["released"] is True
        assert _wait_until(lambda: model.trainer.global_step > step)

    _fit_while(model, pause_then_continue, tmp_path)
    assert client.post("/control/pause").status_code == 409
