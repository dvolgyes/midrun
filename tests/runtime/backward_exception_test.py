"""Behavioral tests for the backward-exception breakpoint and its command handlers."""

import json
import threading
import time
from pathlib import Path

import pytest
import torch

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.commands.command import Command
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.runtime import MidrunRuntime


def _make_runtime(tmp_path: Path, **overrides: object) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False, **overrides
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return runtime


class _FakeState:
    stage = None


class _FakeTrainer:
    def __init__(self) -> None:
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.optimizers: list[object] = []
        self.state = _FakeState()


class _FakePLModule:
    def __init__(self, trainer: _FakeTrainer, parameters: dict | None = None) -> None:
        self.trainer = trainer
        self._parameters = parameters or {}

    def named_parameters(self):
        return self._parameters.items()


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate(), "condition not met before timeout"


def _wait_for_command(runtime: MidrunRuntime, command_id: str) -> Command:
    _wait_until(
        lambda: (
            runtime.command_queue.get(command_id).status not in ("queued", "executing")
        )
    )
    return runtime.command_queue.get(command_id)


def test_handle_backward_exception_noop_when_disabled(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, break_on_backward_exception=False)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    runtime.handle_backward_exception(pl_module, "loss", ValueError("boom"))

    assert runtime._exception_hold is None
    assert runtime.state.read().status != "exception_held"


def test_handle_backward_exception_holds_until_released(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)
    runtime._gather_ranks(pl_module)
    runtime.current_step.begin(
        batch={"x": 1}, batch_idx=0, epoch=0, global_step=0, started_ns=0
    )

    thread = threading.Thread(
        target=runtime.handle_backward_exception,
        args=(pl_module, "loss-value", ValueError("boom")),
    )
    thread.start()
    try:
        _wait_until(lambda: runtime.state.read().status == "exception_held")
        info = runtime._exception_hold
        assert info is not None
        assert info.exception_type == "ValueError"
        assert info.exception_message == "boom"
        assert runtime.state.read().exception is info

        runtime.command_queue.submit("release_exception", {}, parse_when("now"))
        runtime.gate.notify()
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            runtime._exception_hold = None
            runtime.gate.notify()
            thread.join(timeout=5)

    assert runtime._exception_hold is None
    assert runtime.state.read().status == "running"


def test_save_current_batch_and_loss_during_exception_hold(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)
    runtime._gather_ranks(pl_module)
    runtime.current_step.begin(
        batch=torch.tensor([1.0, 2.0]),
        batch_idx=0,
        epoch=0,
        global_step=0,
        started_ns=0,
    )

    thread = threading.Thread(
        target=runtime.handle_backward_exception,
        args=(pl_module, torch.tensor(3.0), ValueError("boom")),
    )
    thread.start()
    try:
        _wait_until(lambda: runtime.state.read().status == "exception_held")

        batch_cmd = runtime.command_queue.submit(
            "save_current_batch", {}, parse_when("now")
        )
        loss_cmd = runtime.command_queue.submit(
            "save_current_loss", {}, parse_when("now")
        )
        runtime.gate.notify()
        batch_cmd = _wait_for_command(runtime, batch_cmd.id)
        loss_cmd = _wait_for_command(runtime, loss_cmd.id)

        assert batch_cmd.status == "succeeded"
        assert loss_cmd.status == "succeeded"
        saved_batch = torch.load(Path(batch_cmd.result["path"]) / "batch.pt")
        assert torch.equal(saved_batch, torch.tensor([1.0, 2.0]))
        saved_loss = torch.load(Path(loss_cmd.result["path"]) / "loss.pt")
        assert torch.equal(saved_loss, torch.tensor(3.0))
    finally:
        runtime.command_queue.submit("release_exception", {}, parse_when("now"))
        runtime.gate.notify()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("kind", "missing"),
    [("save_current_batch", "batch"), ("save_current_loss", "loss")],
)
def test_save_current_batch_and_loss_fail_without_an_in_flight_value(
    tmp_path: Path, kind: str, missing: str
) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    cmd = runtime.command_queue.submit(kind, {}, parse_when("now"))
    runtime._execute(pl_module, trainer, cmd, safe_point=None)

    assert cmd.status == "failed"
    assert f"{kind} requires an in-flight {missing}" in cmd.error
    assert list((tmp_path / "debug").iterdir()) == []


def test_save_current_batch_writes_only_the_viewed_elements(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)
    base = torch.arange(200_000 * 16, dtype=torch.float32).reshape(200_000, 16)
    batch = {"x": base[:32], "pair": (base[32:64, ::2], 7), "names": ["a", "b"]}
    runtime.current_step.begin(
        batch=batch, batch_idx=0, epoch=0, global_step=0, started_ns=0
    )

    cmd = runtime.command_queue.submit("save_current_batch", {}, parse_when("now"))
    runtime._execute(pl_module, trainer, cmd, safe_point=None)

    assert cmd.status == "succeeded"
    path = Path(cmd.result["path"]) / "batch.pt"
    viewed_bytes = (32 * 16 + 32 * 8) * base.element_size()
    assert path.stat().st_size < viewed_bytes + 64 * 1024
    saved = torch.load(path)
    assert saved.keys() == batch.keys()
    assert torch.equal(saved["x"], batch["x"])
    assert isinstance(saved["pair"], tuple)
    assert torch.equal(saved["pair"][0], batch["pair"][0])
    assert saved["pair"][1] == 7
    assert saved["names"] == ["a", "b"]


def test_save_partial_gradients_reports_grad_presence_and_norm(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    with_grad = torch.nn.Parameter(torch.zeros(2))
    with_grad.grad = torch.ones(2)
    without_grad = torch.nn.Parameter(torch.zeros(3))
    pl_module = _FakePLModule(
        trainer, parameters={"with_grad": with_grad, "without_grad": without_grad}
    )
    runtime._gather_ranks(pl_module)

    thread = threading.Thread(
        target=runtime.handle_backward_exception,
        args=(pl_module, "loss", ValueError("boom")),
    )
    thread.start()
    try:
        _wait_until(lambda: runtime.state.read().status == "exception_held")

        cmd = runtime.command_queue.submit(
            "save_partial_gradients", {}, parse_when("now")
        )
        runtime.gate.notify()
        cmd = _wait_for_command(runtime, cmd.id)

        assert cmd.status == "succeeded"
        summary = json.loads((Path(cmd.result["path"]) / "gradients.json").read_text())
        assert summary["with_grad"]["grad_is_none"] is False
        assert summary["with_grad"]["grad_norm"] == pytest.approx(2**0.5)
        assert summary["without_grad"]["grad_is_none"] is True
    finally:
        runtime.command_queue.submit("release_exception", {}, parse_when("now"))
        runtime.gate.notify()
        thread.join(timeout=5)


def test_save_optimizer_summary_during_exception_hold(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    linear = torch.nn.Linear(2, 2)
    trainer.optimizers = [torch.optim.SGD(linear.parameters(), lr=0.01)]
    pl_module = _FakePLModule(trainer)
    runtime._gather_ranks(pl_module)

    thread = threading.Thread(
        target=runtime.handle_backward_exception,
        args=(pl_module, "loss", ValueError("boom")),
    )
    thread.start()
    try:
        _wait_until(lambda: runtime.state.read().status == "exception_held")

        cmd = runtime.command_queue.submit(
            "save_optimizer_summary", {}, parse_when("now")
        )
        runtime.gate.notify()
        cmd = _wait_for_command(runtime, cmd.id)

        assert cmd.status == "succeeded"
        payload = json.loads((Path(cmd.result["path"]) / "optimizer.json").read_text())
        assert payload["0"]["type"] == "SGD"
        assert payload["0"]["param_groups"][0]["lr"] == 0.01
    finally:
        runtime.command_queue.submit("release_exception", {}, parse_when("now"))
        runtime.gate.notify()
        thread.join(timeout=5)


def test_disallowed_command_is_not_executed_during_exception_hold(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)
    runtime._gather_ranks(pl_module)

    thread = threading.Thread(
        target=runtime.handle_backward_exception,
        args=(pl_module, "loss", ValueError("boom")),
    )
    thread.start()
    try:
        _wait_until(lambda: runtime.state.read().status == "exception_held")

        cmd = runtime.command_queue.submit("checkpoint", {}, parse_when("now"))
        runtime.gate.notify()
        time.sleep(0.3)

        assert runtime.command_queue.get(cmd.id).status == "queued"
    finally:
        runtime.command_queue.submit("release_exception", {}, parse_when("now"))
        runtime.gate.notify()
        thread.join(timeout=5)


def test_detect_backward_interception_allows_known_strategy(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()

    class SingleDeviceStrategy:
        pass

    trainer.strategy = SingleDeviceStrategy()
    pl_module = _FakePLModule(trainer)

    runtime._detect_backward_interception(pl_module)

    assert runtime.state.read().backward_interception_supported is True
    assert runtime.state.read().backward_interception_reason is None


def test_detect_backward_interception_flags_unknown_strategy(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()

    class _CustomStrategy:
        pass

    trainer.strategy = _CustomStrategy()
    pl_module = _FakePLModule(trainer)

    runtime._detect_backward_interception(pl_module)

    snapshot = runtime.state.read()
    assert snapshot.backward_interception_supported is False
    assert snapshot.backward_interception_reason is not None
    assert "_CustomStrategy" in snapshot.backward_interception_reason


def test_release_exception_fails_without_an_exception_hold(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    cmd = runtime.command_queue.submit("release_exception", {}, parse_when("now"))
    runtime._execute(pl_module, trainer, cmd, safe_point=None)

    assert cmd.status == "failed"
    assert "no exception hold" in cmd.error


def test_an_interrupted_exception_hold_ends_the_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)
    runtime._gather_ranks(pl_module)
    runtime.current_step.begin(
        batch={"x": 1}, batch_idx=0, epoch=0, global_step=0, started_ns=0
    )

    def _interrupt(timeout: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime.gate, "wait_for_control_event", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        runtime.handle_backward_exception(pl_module, "loss", ValueError("boom"))

    assert runtime._exception_hold is None
    assert runtime.state.read().exception is None
    assert runtime.current_step.exception is None
