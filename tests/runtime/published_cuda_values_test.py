"""A safe point publishes CUDA values through a host copy that only a reader waits for."""

import gc
import math
import time
import weakref
from pathlib import Path

import pytest
import torch

from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.events import SafePoint
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.state.host_snapshot import host_snapshot
from midrun.runtime.state.publishable_values import publishable_values

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

_CUDA = torch.device("cuda")


class _FakeState:
    stage = None


class _FakeTrainer:
    def __init__(self, metrics: dict[str, object], lrs: list[object]) -> None:
        self.callback_metrics = metrics
        self.current_epoch = 0
        self.global_step = 3
        self.optimizers = [
            torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr) for lr in lrs
        ]
        self.state = _FakeState()


class _Model(torch.nn.Module):
    pass


def _published(tmp_path: Path, metrics: dict[str, object], lrs: list[object]):
    config = MidrunConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    model = _Model()
    model.trainer = _FakeTrainer(metrics, lrs)
    runtime._gather_ranks(model)
    runtime.on_fit_start(model)
    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)
    return runtime, model


def _tensors_in(root: object) -> list[torch.Tensor]:
    """Returns the tensors that midrun objects, dicts and tuples under `root` hold."""
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, torch.Tensor):
            tensors.append(current)
        elif isinstance(current, dict | list | tuple) or (
            not isinstance(current, type)
            and type(current).__module__.startswith("midrun.")
        ):
            pending.extend(gc.get_referents(current))
    return tensors


def test_mixed_dtypes_and_devices_read_as_float_of_each_value(tmp_path: Path) -> None:
    metrics: dict[str, object] = {
        "f32": torch.tensor(0.1, device=_CUDA),
        "f64": torch.tensor(0.1, dtype=torch.float64, device=_CUDA),
        "bf16": torch.tensor(0.3, dtype=torch.bfloat16, device=_CUDA),
        "f16": torch.tensor(0.3, dtype=torch.float16, device=_CUDA),
        "i64": torch.tensor(2**53 + 1, device=_CUDA),
        "i32": torch.tensor(-7, dtype=torch.int32, device=_CUDA),
        "bool": torch.tensor(True, device=_CUDA),
        "one_element_vector": torch.tensor([5.0], device=_CUDA),
        "nan": torch.tensor(math.nan, device=_CUDA),
        "inf": torch.tensor(math.inf, device=_CUDA),
        "minus_inf": torch.tensor(-math.inf, dtype=torch.bfloat16, device=_CUDA),
        "cpu": torch.tensor(0.7),
        "cpu_bf16": torch.tensor(0.7, dtype=torch.bfloat16),
        "number": 3,
        "vector": torch.ones(3, device=_CUDA),
        "empty": torch.empty(0, device=_CUDA),
        "complex": torch.tensor(1 + 1j, device=_CUDA),
    }
    skipped = {"vector", "empty", "complex"}
    lrs = [torch.tensor(0.01, device=_CUDA), torch.tensor(0.02), 0.03]
    runtime, _ = _published(tmp_path, metrics, lrs)

    read = host_snapshot(runtime.state.read())

    expected = {
        key: float(value) for key, value in metrics.items() if key not in skipped
    }
    values = {key: metric.value for key, metric in read.metrics.items()}
    assert set(values) == set(expected)
    assert all(type(value) is float for value in values.values())
    assert math.isnan(values.pop("nan"))
    expected.pop("nan")
    assert values == expected
    assert read.learning_rates == tuple(float(lr) for lr in lrs)
    assert all(type(lr) is float for lr in read.learning_rates)


def test_the_state_holds_no_graph_and_no_device_memory(tmp_path: Path) -> None:
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    weight = torch.full((), 2.0, device=_CUDA, requires_grad=True)
    loss = weight * 3
    loss_ref = weakref.ref(loss)
    lr = torch.tensor(0.5, device=_CUDA)
    runtime, model = _published(tmp_path, {"loss": loss, "half": loss / 2}, [lr])

    model.trainer.callback_metrics = {}
    model.trainer.optimizers = []
    del weight, loss, lr
    gc.collect()
    torch.cuda.synchronize()
    snapshot = runtime.state.read()
    held = _tensors_in(snapshot)

    assert loss_ref() is None
    assert torch.cuda.memory_allocated() == baseline
    assert held, "the walk found no host copy"
    for tensor in held:
        assert tensor.device.type == "cpu"
        assert tensor.is_pinned()
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
    read = host_snapshot(snapshot)
    assert {key: metric.value for key, metric in read.metrics.items()} == {
        "loss": 6.0,
        "half": 3.0,
    }
    assert read.learning_rates == (0.5,)


def test_a_publish_does_not_wait_for_a_busy_device() -> None:
    # The first use of a kernel loads it, which waits for the device.
    warm = torch.zeros((), device=_CUDA)
    warm += 4
    [float(scalar) for scalar in publishable_values([warm, warm * 2])]
    torch.cuda.synchronize()
    value = torch.zeros((), device=_CUDA)
    # About 0.4 s on an RTX 2060; the publish must return long before the device.
    torch.cuda._sleep(500_000_000)
    value += 4

    started = time.perf_counter()
    published = publishable_values([value, value * 2])
    publish_seconds = time.perf_counter() - started
    del value
    read = [float(scalar) for scalar in published]
    read_seconds = time.perf_counter() - started

    assert read == [4.0, 8.0]
    assert publish_seconds < 0.05
    assert read_seconds > 0.2
