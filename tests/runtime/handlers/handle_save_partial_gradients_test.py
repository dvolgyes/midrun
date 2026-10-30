"""Behavioral tests: gradients.json is standard JSON and holds the true gradient norm."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from midrun.runtime.artifacts import ArtifactStore
from midrun.runtime.handlers.handle_save_partial_gradients import (
    handle_save_partial_gradients,
)


def _save_gradients(tmp_path: Path, grads: dict[str, torch.Tensor]) -> dict:
    module = torch.nn.Module()
    for name, grad in grads.items():
        parameter = torch.nn.Parameter(torch.zeros_like(grad))
        parameter.grad = grad
        module.register_parameter(name, parameter)
    runtime = SimpleNamespace(artifact_store=ArtifactStore(tmp_path))
    manifest = handle_save_partial_gradients(runtime, module, None, None, None)

    def reject(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant {constant}")

    text = (Path(manifest["path"]) / "gradients.json").read_text()
    return json.loads(text, parse_constant=reject)


def test_non_finite_norms_are_tagged_floats_in_standard_json(tmp_path: Path) -> None:
    summary = _save_gradients(
        tmp_path,
        {
            "nan_grad": torch.tensor([float("nan"), 1.0]),
            "inf_grad": torch.tensor([float("inf"), 1.0]),
        },
    )

    assert summary["nan_grad"]["grad_norm"] == {"kind": "float", "value": "nan"}
    assert summary["inf_grad"]["grad_norm"] == {"kind": "float", "value": "inf"}


def test_a_half_precision_norm_beyond_the_half_range_is_finite(tmp_path: Path) -> None:
    # Each element fits float16, but the norm, 1000 * 100 = 1e5, does not.
    summary = _save_gradients(
        tmp_path, {"weight": torch.full((10_000,), 1000.0, dtype=torch.float16)}
    )

    assert summary["weight"]["grad_norm"] == pytest.approx(1e5, rel=1e-4)
