"""Behavioral tests for `graph_inspectors.torchview_dot`.

Unlike `torchview_capture`, this never renders, so it needs only the `torchview`
Python package -- deliberately not skipped by the `shutil.which("dot") is None` guard
that gates `graph_inspectors_test.py`'s render-based tests.
"""

import sys

import pytest
import torch

from midrun.runtime.graph_inspectors.torchview_dot import torchview_dot


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_torchview_dot_returns_source_without_dot_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    source = torchview_dot(_Model(), torch.randn(3, 4))
    assert "digraph" in source


class _ParameterlessModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class _BufferOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def test_torchview_dot_traces_a_model_without_parameters() -> None:
    source = torchview_dot(_ParameterlessModel(), torch.randn(3, 4))
    assert "digraph" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a cuda device")
def test_torchview_dot_keeps_a_buffer_only_model_on_its_device() -> None:
    model = _BufferOnlyModel().cuda()
    source = torchview_dot(model, torch.randn(3, 4, device="cuda"))
    assert "digraph" in source
    assert model.scale.device.type == "cuda"


def test_torchview_dot_raises_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torchview", None)
    with pytest.raises(ImportError):
        torchview_dot(_Model(), torch.randn(3, 4))
