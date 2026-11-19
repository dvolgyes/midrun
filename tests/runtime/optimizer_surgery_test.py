"""Behavioral tests for optimizer surgery: LR changes and momentum reset."""

import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from midrun.build_runtime import build_runtime
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.rest.app import build_app
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.command_specs.execute_command import execute_command
from midrun.runtime.commands.command import Command
from midrun.runtime.commands.parse_when import parse_when
from midrun.runtime.optimizer_surgery.adam_surgery_adapter import (
    AdamSurgeryAdapter,
)
from midrun.runtime.optimizer_surgery.find_surgery_adapter import (
    find_surgery_adapter,
)
from midrun.runtime.optimizer_surgery.resolve_optimizer_parameters import (
    resolve_optimizer_parameters,
)
from midrun.runtime.optimizer_surgery.sgd_surgery_adapter import SGDSurgeryAdapter
from midrun.runtime.runtime import MidrunRuntime


def _make_runtime(**overrides: object) -> MidrunRuntime:
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False, **overrides
    )
    return build_runtime(config, load_backend("lightning.pytorch"))


class _FakeState:
    stage = None


class _FakeTrainer:
    def __init__(self, optimizers: list | None = None) -> None:
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.optimizers = optimizers or []
        self.state = _FakeState()
        self.lr_scheduler_configs: list = []


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(2, 2)
        self.b = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


class _FakePLModule:
    def __init__(self, trainer: _FakeTrainer, model: torch.nn.Module) -> None:
        self.trainer = trainer
        self._model = model

    def named_parameters(self):
        return self._model.named_parameters()


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()


def _submit_and_run(
    runtime: MidrunRuntime,
    pl_module: _FakePLModule,
    trainer: _FakeTrainer,
    kind: str,
    args: dict,
) -> Command:
    command = runtime.command_queue.submit(kind, args, parse_when("now"))
    execute_command(runtime, pl_module, trainer, command, safe_point=None)
    return command


# ---- adapter unit tests ----


def test_find_surgery_adapter_returns_none_for_unsupported_optimizer() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adagrad(model.parameters())
    assert find_surgery_adapter(optimizer) is None


def test_sgd_adapter_drops_momentum_buffer() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    _step(model, optimizer)
    assert all("momentum_buffer" in optimizer.state[p] for p in model.parameters())

    apply = SGDSurgeryAdapter().prepare_reset_momentum(
        optimizer, list(model.parameters())
    )
    assert all("momentum_buffer" in optimizer.state[p] for p in model.parameters())
    result = apply()

    assert result["semantic"] == "momentum_buffer_dropped"
    assert all("momentum_buffer" not in optimizer.state[p] for p in model.parameters())


def test_adam_adapter_zeroes_exp_avg_but_not_exp_avg_sq_or_step() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    _step(model, optimizer)
    before_sq = {
        id(p): optimizer.state[p]["exp_avg_sq"].clone() for p in model.parameters()
    }
    before_step = {
        id(p): optimizer.state[p]["step"].clone() for p in model.parameters()
    }

    before_avg = [optimizer.state[p]["exp_avg"].clone() for p in model.parameters()]
    apply = AdamSurgeryAdapter().prepare_reset_momentum(
        optimizer, list(model.parameters())
    )
    for p, before in zip(model.parameters(), before_avg, strict=True):
        assert torch.equal(optimizer.state[p]["exp_avg"], before)
    result = apply()

    assert result["semantic"] == "exp_avg_zeroed"
    for p in model.parameters():
        assert torch.equal(
            optimizer.state[p]["exp_avg"],
            torch.zeros_like(optimizer.state[p]["exp_avg"]),
        )
        assert torch.equal(optimizer.state[p]["exp_avg_sq"], before_sq[id(p)])
        assert torch.equal(optimizer.state[p]["step"], before_step[id(p)])


def test_resolve_optimizer_parameters_all_returns_every_param() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resolved = resolve_optimizer_parameters(model, optimizer, "all")
    assert len(resolved) == len(list(model.parameters()))


def test_resolve_optimizer_parameters_by_name() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resolved = resolve_optimizer_parameters(model, optimizer, {"parameter": "a.weight"})
    expected = dict(model.named_parameters())["a.weight"]
    assert len(resolved) == 1
    assert resolved[0] is expected


def test_resolve_optimizer_parameters_by_prefix() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resolved = resolve_optimizer_parameters(
        model, optimizer, {"parameter_prefix": "a."}
    )
    resolved_ids = {id(p) for p in resolved}
    names = {name for name, p in model.named_parameters() if id(p) in resolved_ids}
    assert names == {"a.weight", "a.bias"}


def test_resolve_optimizer_parameters_by_param_group() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(
        [{"params": model.a.parameters()}, {"params": model.b.parameters()}], lr=0.1
    )
    resolved = resolve_optimizer_parameters(model, optimizer, {"param_group": 1})
    expected_ids = {id(p) for p in model.b.parameters()}
    assert {id(p) for p in resolved} == expected_ids


def test_resolve_optimizer_parameters_rejects_unknown_shape() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="invalid momentum-reset selector"):
        resolve_optimizer_parameters(model, optimizer, {"bogus": 1})


def test_resolve_optimizer_parameters_rejects_unknown_name() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="no optimizer parameter named"):
        resolve_optimizer_parameters(model, optimizer, {"parameter": "does-not-exist"})


# ---- command handler tests ----


def test_set_learning_rate_updates_all_groups_by_default() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime, pl_module, trainer, "set_learning_rate", {"value": 0.5}
    )

    assert command.status == "succeeded"
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_set_learning_rate_targets_single_param_group() -> None:
    runtime = _make_runtime()
    model = _Model()
    optimizer = torch.optim.SGD(
        [{"params": model.a.parameters()}, {"params": model.b.parameters()}], lr=0.1
    )
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime,
        pl_module,
        trainer,
        "set_learning_rate",
        {"value": 0.9, "param_group": 1},
    )

    assert command.status == "succeeded"
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[1]["lr"] == 0.9


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_set_learning_rate_rejects_a_non_finite_or_negative_value(
    value: float,
) -> None:
    runtime = _make_runtime()

    with pytest.raises(ValueError, match="invalid args for 'set_learning_rate'"):
        runtime.command_queue.submit(
            "set_learning_rate", {"value": value}, parse_when("now")
        )

    assert runtime.command_queue.list_all() == []


def test_set_learning_rate_warns_when_scheduler_active() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _FakeTrainer(optimizers=[optimizer])
    trainer.lr_scheduler_configs = [object()]
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime, pl_module, trainer, "set_learning_rate", {"value": 0.5}
    )

    assert command.status == "succeeded"
    assert "warning" in command.result


def test_set_learning_rate_param_group_requires_surgery_enabled() -> None:
    runtime = _make_runtime(optimizer_surgery_enabled=False)
    model = _Model()
    optimizer = torch.optim.SGD(
        [{"params": model.a.parameters()}, {"params": model.b.parameters()}], lr=0.1
    )
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime,
        pl_module,
        trainer,
        "set_learning_rate",
        {"value": 0.5, "param_group": 0},
    )

    assert command.status == "failed"


def test_set_learning_rate_all_groups_still_works_when_surgery_disabled() -> None:
    runtime = _make_runtime(optimizer_surgery_enabled=False)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime, pl_module, trainer, "set_learning_rate", {"value": 0.5}
    )

    assert command.status == "succeeded"


def test_reset_momentum_zeroes_adam_exp_avg_via_command() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    _step(model, optimizer)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(runtime, pl_module, trainer, "reset_momentum", {})

    assert command.status == "succeeded"
    assert command.result["semantic"] == "exp_avg_zeroed"
    for p in model.parameters():
        assert torch.equal(
            optimizer.state[p]["exp_avg"],
            torch.zeros_like(optimizer.state[p]["exp_avg"]),
        )


@pytest.mark.parametrize("family", [torch.optim.Adam, torch.optim.AdamW])
def test_reset_momentum_preserves_state_loaded_into_another_optimizer(
    family: type[torch.optim.Adam],
) -> None:
    model = _Model()
    target = family(model.a.parameters())
    other = family(model.b.parameters())
    _step(model.a, target)
    other.load_state_dict(target.state_dict())
    before = other.state[model.b.bias]["exp_avg"].clone()
    assert before.count_nonzero() > 0
    trainer = _FakeTrainer([target, other])

    command = _submit_and_run(
        _make_runtime(),
        _FakePLModule(trainer, model),
        trainer,
        "reset_momentum",
        {"optimizer": 0, "selector": {"parameter": "a.bias"}},
    )

    assert command.status == "succeeded", command.error
    assert command.result["parameters_reset"] == 1
    assert target.state[model.a.bias]["exp_avg"].count_nonzero() == 0
    assert torch.equal(other.state[model.b.bias]["exp_avg"], before)


def test_reset_momentum_preserves_overlapping_variance_and_sibling_views() -> None:
    model = _Model()
    optimizer = torch.optim.Adam(model.parameters())
    _step(model, optimizer)
    values = torch.tensor([1.0, 2.0, 3.0])
    optimizer.state[model.a.bias]["exp_avg"] = values[:2]
    optimizer.state[model.a.bias]["exp_avg_sq"] = values[1:]
    optimizer.state[model.b.bias]["exp_avg"] = values[1:]
    trainer = _FakeTrainer([optimizer])

    command = _submit_and_run(
        _make_runtime(),
        _FakePLModule(trainer, model),
        trainer,
        "reset_momentum",
        {"selector": {"parameter": "a.bias"}},
    )

    assert command.status == "succeeded", command.error
    assert optimizer.state[model.a.bias]["exp_avg"].tolist() == [0.0, 0.0]
    assert optimizer.state[model.a.bias]["exp_avg_sq"].tolist() == [2.0, 3.0]
    assert optimizer.state[model.b.bias]["exp_avg"].tolist() == [2.0, 3.0]


def test_reset_momentum_rejects_unsupported_optimizer_without_mutating_state() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
    _step(model, optimizer)
    before = {id(p): optimizer.state[p]["sum"].clone() for p in model.parameters()}
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(runtime, pl_module, trainer, "reset_momentum", {})

    assert command.status == "failed"
    for p in model.parameters():
        assert torch.equal(optimizer.state[p]["sum"], before[id(p)])


def test_reset_momentum_rejected_when_surgery_disabled() -> None:
    runtime = _make_runtime(optimizer_surgery_enabled=False)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    _step(model, optimizer)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(runtime, pl_module, trainer, "reset_momentum", {})

    assert command.status == "failed"


def test_reset_momentum_with_selector_only_resets_targeted_group() -> None:
    runtime = _make_runtime()
    model = _Model()
    optimizer = torch.optim.SGD(
        [{"params": model.a.parameters()}, {"params": model.b.parameters()}],
        lr=0.1,
        momentum=0.9,
    )
    _step(model, optimizer)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime,
        pl_module,
        trainer,
        "reset_momentum",
        {"selector": {"param_group": 0}},
    )

    assert command.status == "succeeded"
    for p in model.a.parameters():
        assert "momentum_buffer" not in optimizer.state[p]
    for p in model.b.parameters():
        assert "momentum_buffer" in optimizer.state[p]


def test_reset_momentum_rejects_part_of_a_fused_sgd_group_without_mutating_state() -> (
    None
):
    runtime = _make_runtime()
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, fused=True)
    _step(model, optimizer)
    before = {
        id(p): optimizer.state[p]["momentum_buffer"].clone() for p in model.parameters()
    }
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime,
        pl_module,
        trainer,
        "reset_momentum",
        {"selector": {"parameter_prefix": "a."}},
    )

    assert command.status == "failed"
    assert "fused" in command.error
    for p in model.parameters():
        assert torch.equal(optimizer.state[p]["momentum_buffer"], before[id(p)])
    _step(model, optimizer)


def test_set_learning_rate_updates_a_tensor_lr_in_place_and_reports_floats() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    lr = torch.tensor(0.125, dtype=torch.float16)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime, pl_module, trainer, "set_learning_rate", {"value": 0.1}
    )

    assert command.status == "succeeded"
    assert optimizer.param_groups[0]["lr"] is lr
    assert lr.item() == torch.tensor(0.1, dtype=torch.float16).item()
    response = TestClient(build_app(runtime), base_url="http://localhost").get(
        f"/commands/{command.id}"
    )
    assert response.status_code == 200
    assert response.json()["result"]["groups"]["0"] == {
        "before": 0.125,
        "after": lr.item(),
    }


@pytest.mark.parametrize("inference", [False, True], ids=["grad", "inference"])
def test_set_learning_rate_updates_mixed_adam_groups_or_rejects_before_writing(
    inference: bool,
) -> None:
    runtime = _make_runtime()
    model = _Model()
    with torch.inference_mode(inference):
        lr = torch.tensor(0.25, requires_grad=True)
    optimizer = torch.optim.Adam(
        [
            {"params": model.a.parameters(), "lr": 0.125},
            {"params": model.b.parameters(), "lr": lr},
        ]
    )
    trainer = _FakeTrainer(optimizers=[optimizer])
    command = _submit_and_run(
        runtime,
        _FakePLModule(trainer, model),
        trainer,
        "set_learning_rate",
        {"value": 0.5},
    )

    assert optimizer.param_groups[1]["lr"] is lr
    assert lr.requires_grad
    if inference:
        assert command.status == "failed"
        assert "cannot update inference tensor in param_group 1" in command.error
        assert optimizer.param_groups[0]["lr"] == 0.125
        assert lr.item() == 0.25
    else:
        assert command.status == "succeeded"
        assert optimizer.param_groups[0]["lr"] == 0.5
        assert lr.item() == 0.5
        assert lr.is_leaf
        assert lr.grad_fn is None


@pytest.mark.parametrize("value", [1e5, 1e-8], ids=["overflow", "underflow"])
def test_set_learning_rate_rejects_unrepresentable_rate_atomically(
    value: float,
) -> None:
    runtime = _make_runtime()
    model = _Model()
    wide = torch.tensor(0.125, dtype=torch.float32)
    narrow = torch.tensor(0.25, dtype=torch.float16)
    optimizer = torch.optim.SGD(
        [
            {"params": [model.a.weight], "lr": 0.5},
            {"params": [model.a.bias], "lr": wide},
            {"params": model.b.parameters(), "lr": narrow},
        ]
    )
    other = torch.optim.SGD(model.parameters(), lr=wide.view_as(wide))
    trainer = _FakeTrainer(optimizers=[optimizer, other])
    command = _submit_and_run(
        runtime,
        _FakePLModule(trainer, model),
        trainer,
        "set_learning_rate",
        {"value": value},
    )

    assert command.status == "failed"
    assert optimizer.param_groups[0]["lr"] == 0.5
    assert optimizer.param_groups[1]["lr"] is wide
    assert optimizer.param_groups[2]["lr"] is narrow
    assert (wide.item(), narrow.item()) == (0.125, 0.25)


@pytest.mark.parametrize("use_view", [False, True], ids=["inherited", "view"])
@pytest.mark.parametrize("selection", [1, "*"], ids=["single", "all"])
def test_set_learning_rate_with_aliased_tensor_groups(
    use_view: bool, selection: int | str
) -> None:
    runtime = _make_runtime()
    model = _Model()
    lr = torch.tensor(0.125, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam(
        [{"params": model.a.parameters()}, {"params": model.b.parameters()}], lr=lr
    )
    if use_view:
        optimizer.param_groups[1]["lr"] = lr.view_as(lr)
    trainer = _FakeTrainer(optimizers=[optimizer])
    command = _submit_and_run(
        runtime,
        _FakePLModule(trainer, model),
        trainer,
        "set_learning_rate",
        {"param_group": selection, "value": 0.5},
    )

    assert command.status == "succeeded"
    assert optimizer.param_groups[0]["lr"] is lr
    assert lr.item() == (0.5 if selection == "*" else 0.125)
    target = optimizer.param_groups[1]["lr"]
    assert (target.item(), target.dtype, target.device) == (0.5, lr.dtype, lr.device)
    assert target.requires_grad
    response = TestClient(build_app(runtime), base_url="http://localhost").get(
        f"/commands/{command.id}"
    )
    assert response.status_code == 200
    assert response.json()["result"]["groups"] == {
        str(index): {"before": 0.125, "after": 0.5}
        for index in ((0, 1) if selection == "*" else (1,))
    }


def test_set_learning_rate_preserves_alias_in_another_optimizer() -> None:
    runtime = _make_runtime()
    model = _Model()
    lr = torch.tensor(0.125)
    target = torch.optim.Adam(model.a.parameters(), lr=lr)
    other = torch.optim.Adam(model.b.parameters(), lr=lr)
    trainer = _FakeTrainer(optimizers=[target, other])

    command = _submit_and_run(
        runtime,
        _FakePLModule(trainer, model),
        trainer,
        "set_learning_rate",
        {"value": 0.5},
    )

    assert command.status == "succeeded"
    assert target.param_groups[0]["lr"].item() == 0.5
    assert other.param_groups[0]["lr"] is lr
    assert lr.item() == 0.125


def test_published_learning_rates_of_a_tensor_lr_are_floats_over_rest() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(0.01))
    pl_module = _FakePLModule(_FakeTrainer(optimizers=[optimizer]), model)

    runtime._publish(pl_module, "running")

    client = TestClient(build_app(runtime), base_url="http://localhost")
    optimizers = client.get("/optimizers")
    assert optimizers.status_code == 200
    assert optimizers.json() == {"learning_rates": [pytest.approx(0.01)]}
    assert client.get("/run").status_code == 200


def test_optimizer_summary_of_a_tensor_lr_reports_a_float(tmp_path: Path) -> None:
    runtime = _make_runtime()
    runtime.artifacts = ArtifactStore(tmp_path)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(0.01))
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    capture = _submit_and_run(runtime, pl_module, trainer, "save_optimizer_summary", {})
    snapshot = _submit_and_run(
        runtime, pl_module, trainer, "snapshot", {"kind": "optimizer_summary"}
    )

    assert capture.status == "succeeded", capture.error
    assert snapshot.status == "succeeded", snapshot.error
    for path in (
        Path(capture.result["path"]) / "optimizer.json",
        Path(snapshot.result["path"]) / "optimizer_summary.json",
    ):
        lr = json.loads(path.read_text())["0"]["param_groups"][0]["lr"]
        assert lr == pytest.approx(0.01)


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-standard JSON constant {name}")


@pytest.mark.parametrize("lr", [float("nan"), float("inf")])
def test_optimizer_summary_of_a_non_finite_lr_is_standard_json(
    tmp_path: Path, lr: float
) -> None:
    runtime = _make_runtime()
    runtime.artifacts = ArtifactStore(tmp_path)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.param_groups[0]["lr"] = lr
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    capture = _submit_and_run(runtime, pl_module, trainer, "save_optimizer_summary", {})

    assert capture.status == "succeeded", capture.error
    text = (Path(capture.result["path"]) / "optimizer.json").read_text()
    summary = json.loads(text, parse_constant=_reject_constant)
    assert summary["0"]["param_groups"][0]["lr"] is None
