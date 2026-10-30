"""Behavioral tests for midrun.hooks.dispatch_light and midrun.hooks.dispatch_heavy."""

import errno
import json
import shutil
from pathlib import Path

import numpy
import pytest
import torch
from loguru import logger

from midrun.hooks.dispatch_heavy import dispatch_heavy
from midrun.hooks.dispatch_light import dispatch_light
from midrun.hooks.manager.bootstrap_hooks_tree import bootstrap_hooks_tree
from midrun.hooks.payload.occurrence_context import OccurrenceContext

_SHELL = Path(shutil.which("bash"))
_SETPRIV_FOUND = shutil.which("setpriv")
_SETPRIV = None if _SETPRIV_FOUND is None else Path(_SETPRIV_FOUND)


class _FakeLog:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []

    def error(self, message, *args) -> None:
        self.errors.append(message.format(*args))

    def info(self, message, *args) -> None:
        self.infos.append(message.format(*args))

    def warning(self, message, *args) -> None:
        pass


def _context(**overrides) -> OccurrenceContext:
    fields = {
        "session_id": "sess-1",
        "hook": "on_train_batch_start",
        "stage": "fit",
        "epoch": 3,
        "global_step": 42,
        "batch_idx": 7,
        "dataloader_idx": None,
        "rank": 0,
        "world_size": 1,
    }
    fields.update(overrides)
    return OccurrenceContext(**fields)


def _make_light_script(live_dir, hook: str, body: str) -> None:
    script = live_dir / "light" / f"{hook}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"#!/usr/bin/env bash\n{body}\n")
    script.chmod(0o755)


def _make_heavy_script(
    live_dir, hook: str, body: str, *, executable: bool = True
) -> None:
    script = live_dir / "heavy" / f"{hook}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"#!/usr/bin/env bash\n{body}\n")
    script.chmod(0o755 if executable else 0o644)


def _dispatch(live_dir, tmp_root, context, *, arguments=None, log=None):
    return dispatch_light(
        live_dir,
        context,
        arguments or {},
        occurrence_id="occ-1",
        shell=_SHELL,
        setpriv=_SETPRIV,
        timeout_s=5.0,
        max_params_bytes=256 * 1024,
        max_metadata_items=1000,
        max_metadata_depth=8,
        max_output_bytes=64 * 1024,
        tmp_root=tmp_root,
        log=log or _FakeLog(),
    )


def _dispatch_heavy(
    live_dir,
    tmp_root,
    context,
    occurrence_id,
    arguments,
    tensor_arg_names,
    *,
    log=None,
    max_export_bytes=1024**3,
):
    return dispatch_heavy(
        live_dir,
        context,
        occurrence_id,
        arguments,
        tensor_arg_names,
        shell=_SHELL,
        setpriv=_SETPRIV,
        timeout_s=5.0,
        max_metadata_items=1000,
        max_metadata_depth=8,
        max_export_bytes=max_export_bytes,
        max_output_bytes=64 * 1024,
        tmp_root=tmp_root,
        log=log or _FakeLog(),
    )


def test_absent_script_returns_none_and_allocates_nothing(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    live_dir.mkdir()
    tmp_root = tmp_path / "tmp-root"
    result = _dispatch(live_dir, tmp_root, _context())
    assert result is None
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []


def test_non_executable_script_returns_none_and_allocates_nothing(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(live_dir, "on_train_batch_start", "exit 0")
    (live_dir / "light" / "on_train_batch_start.sh").chmod(0o644)
    tmp_root = tmp_path / "tmp-root"
    result = _dispatch(live_dir, tmp_root, _context())
    assert result is None
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []


def test_successful_script_returns_result_and_logs_nothing_at_error(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(live_dir, "on_train_batch_start", "exit 0")
    tmp_root = tmp_path / "tmp-root"
    log = _FakeLog()
    result = _dispatch(live_dir, tmp_root, _context(), log=log)
    assert result is not None
    assert result.phase == "completed"
    assert result.exit_code == 0
    assert log.errors == []


def test_failing_script_returns_result_and_logs_exactly_one_error(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(live_dir, "on_train_batch_start", "exit 3")
    tmp_root = tmp_path / "tmp-root"
    log = _FakeLog()
    context = _context()
    result = _dispatch(live_dir, tmp_root, context, log=log)
    assert result is not None
    assert result.exit_code == 3
    assert len(log.errors) == 1
    message = log.errors[0]
    assert context.hook in message
    assert context.session_id in message
    assert "exit_code=3" in message


def test_failure_record_says_that_the_output_was_truncated(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(
        live_dir,
        "on_train_batch_start",
        "head -c 100000 /dev/zero | tr '\\0' A\necho FINAL-LINE\nexit 1",
    )
    log = _FakeLog()
    _dispatch(live_dir, tmp_path / "tmp-root", _context(), log=log)
    assert len(log.errors) == 1
    assert "stdout_truncated=True" in log.errors[0]
    assert "stderr_truncated=False" in log.errors[0]
    assert "FINAL-LINE" in log.errors[0]


def test_invocation_dir_exists_during_run_and_is_removed_after(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    marker = tmp_path / "marker.json"
    _make_light_script(live_dir, "on_train_batch_start", f'cp "$1" "{marker}"')
    tmp_root = tmp_path / "tmp-root"
    result = _dispatch(live_dir, tmp_root, _context())
    assert result is not None
    assert result.phase == "completed"
    assert result.exit_code == 0

    # The script's own copy proves params.json existed, mid-run, as valid JSON.
    marker_content = json.loads(marker.read_text())
    assert marker_content["hook"] == "on_train_batch_start"

    # The per-invocation temp dir is cleaned up afterward; nothing lingers under tmp_root.
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []


def test_metacharacter_laden_argument_is_inert(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(live_dir, "on_train_batch_start", 'cat "$1"')
    tmp_root = tmp_path / "tmp-root"
    canary = tmp_path / "should-not-run"
    payload = f"$({{ touch {canary}; }}); `id`; ${{IFS}}"
    result = _dispatch(live_dir, tmp_root, _context(), arguments={"note": payload})
    assert result is not None
    assert result.phase == "completed"
    assert not canary.exists()
    dumped = json.loads(result.stdout_tail)
    assert dumped["arguments"]["note"] == payload


def test_context_fields_round_trip_into_params_json(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(live_dir, "on_train_batch_start", 'cat "$1"')
    tmp_root = tmp_path / "tmp-root"
    context = _context(stage="validate", epoch=9, session_id="sess-xyz")
    result = _dispatch(live_dir, tmp_root, context)
    assert result is not None
    dumped = json.loads(result.stdout_tail)
    assert dumped["hook"] == "on_train_batch_start"
    assert dumped["session_id"] == "sess-xyz"
    assert dumped["stage"] == "validate"
    assert dumped["epoch"] == 9


def test_heavy_absent_or_disabled_script_returns_none_and_allocates_nothing(
    tmp_path,
) -> None:
    live_dir = tmp_path / "hooks"
    live_dir.mkdir()
    tmp_root = tmp_path / "tmp-root"
    result = _dispatch_heavy(live_dir, tmp_root, _context(), "occ-1", {}, ())
    assert result is None
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []

    _make_heavy_script(live_dir, "on_train_batch_start", "exit 0", executable=False)
    result = _dispatch_heavy(live_dir, tmp_root, _context(), "occ-1", {}, ())
    assert result is None
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []


def test_heavy_exports_tensor_and_manifest_describes_it(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    marker = tmp_path / "tensor-copy.npy"
    _make_heavy_script(
        live_dir,
        "on_train_batch_start",
        f'cat "$1"\ncp "$(dirname "$1")/tensor-000001.npy" "{marker}"\n',
    )
    tmp_root = tmp_path / "tmp-root"
    tensor = torch.arange(6.0).reshape(2, 3)
    result = _dispatch_heavy(
        live_dir,
        tmp_root,
        _context(),
        "occ-1",
        {"batch": {"x": tensor}},
        ("batch",),
    )
    assert result is not None
    assert result.phase == "completed"
    assert result.exit_code == 0

    manifest = json.loads(result.stdout_tail)
    assert manifest["modality"] == "heavy"
    [record] = manifest["tensor_records"]
    assert record["argument_path"] == "batch/x"
    assert record["shape"] == [2, 3]
    assert record["file"] == "tensor-000001.npy"

    loaded = numpy.load(marker)
    assert numpy.array_equal(loaded, tensor.numpy())

    # Per-invocation temp dir (and its .npy files) cleaned up afterward.
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []


def test_heavy_with_no_tensors_present_still_runs_with_empty_manifest(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_heavy_script(live_dir, "on_train_batch_start", 'cat "$1"')
    tmp_root = tmp_path / "tmp-root"

    result = _dispatch_heavy(live_dir, tmp_root, _context(), "occ-1", {}, ())
    assert result is not None
    assert result.phase == "completed"
    manifest = json.loads(result.stdout_tail)
    assert manifest["tensor_records"] == []

    result = _dispatch_heavy(
        live_dir, tmp_root, _context(), "occ-1", {"batch": {"x": torch.zeros(2)}}, ()
    )
    assert result is not None
    manifest = json.loads(result.stdout_tail)
    assert manifest["tensor_records"] == []


def test_heavy_capture_failure_never_launches_script_and_logs_once(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    canary = tmp_path / "should-not-run"
    _make_heavy_script(live_dir, "on_train_batch_start", f'touch "{canary}"')
    tmp_root = tmp_path / "tmp-root"
    log = _FakeLog()

    sparse = torch.zeros(4, 4).to_sparse()
    result = _dispatch_heavy(
        live_dir,
        tmp_root,
        _context(),
        "occ-1",
        {"batch": {"x": sparse}},
        ("batch",),
        log=log,
    )

    assert result is not None
    assert result.phase == "capture_failed"
    assert result.error is not None and "batch/x" in result.error
    assert not canary.exists()
    assert len(log.errors) == 1
    assert "capture_failed" in log.errors[0]

    # No invocation directory was ever created for a refused capture.
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []


def test_heavy_over_budget_is_a_capture_failure(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    canary = tmp_path / "should-not-run-2"
    _make_heavy_script(live_dir, "on_train_batch_start", f'touch "{canary}"')
    tmp_root = tmp_path / "tmp-root"

    big = torch.zeros(1000, 1000, dtype=torch.float32)
    result = _dispatch_heavy(
        live_dir,
        tmp_root,
        _context(),
        "occ-1",
        {"batch": {"x": big}},
        ("batch",),
        max_export_bytes=100,
    )

    assert result is not None
    assert result.phase == "capture_failed"
    assert not canary.exists()


def test_heavy_failing_script_reports_exit_code_and_logs_once(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_heavy_script(live_dir, "on_train_batch_start", "exit 5")
    tmp_root = tmp_path / "tmp-root"
    log = _FakeLog()

    result = _dispatch_heavy(live_dir, tmp_root, _context(), "occ-1", {}, (), log=log)

    assert result is not None
    assert result.exit_code == 5
    assert len(log.errors) == 1


def test_heavy_successful_script_logs_nothing_at_error(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_heavy_script(live_dir, "on_train_batch_start", "exit 0")
    tmp_root = tmp_path / "tmp-root"
    log = _FakeLog()

    result = _dispatch_heavy(live_dir, tmp_root, _context(), "occ-1", {}, (), log=log)

    assert result is not None
    assert result.phase == "completed"
    assert log.errors == []


def test_heavy_metacharacter_laden_argument_is_inert(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_heavy_script(live_dir, "on_train_batch_start", 'cat "$1"')
    tmp_root = tmp_path / "tmp-root"
    canary = tmp_path / "should-not-run-3"
    payload = f"$({{ touch {canary}; }}); `id`; ${{IFS}}"

    result = _dispatch_heavy(
        live_dir,
        tmp_root,
        _context(),
        "occ-1",
        {"batch": {"x": torch.zeros(2), "note": payload}},
        ("batch",),
    )

    assert result is not None
    assert result.phase == "completed"
    assert not canary.exists()
    manifest = json.loads(result.stdout_tail)
    assert manifest["arguments"]["batch"]["note"] == payload


def test_heavy_exports_conjugate_and_negative_view_tensors_by_value(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    copies = tmp_path / "copies"
    copies.mkdir()
    _make_heavy_script(
        live_dir, "on_train_batch_start", f'cp "$(dirname "$1")"/*.npy "{copies}"'
    )
    tmp_root = tmp_path / "tmp-root"
    complex_values = torch.tensor([1 + 2j, 3 - 4j])
    conjugate = complex_values.conj()
    negative_view = complex_values.conj().imag

    result = _dispatch_heavy(
        live_dir,
        tmp_root,
        _context(),
        "occ-1",
        {"batch": {"c": conjugate, "n": negative_view}},
        ("batch",),
    )

    assert result is not None
    assert result.phase == "completed"
    assert numpy.array_equal(
        numpy.load(copies / "tensor-000001.npy"), numpy.array([1 - 2j, 3 + 4j])
    )
    assert numpy.array_equal(
        numpy.load(copies / "tensor-000002.npy"), numpy.array([-2.0, 4.0])
    )


def _raise_no_space(*_args, **_kwargs):
    raise OSError(errno.ENOSPC, "No space left on device")


@pytest.mark.parametrize(
    "failing_write",
    ["numpy.save", "midrun.hooks.dispatch_heavy.write_json_file"],
)
def test_heavy_payload_write_error_is_a_capture_failure(
    tmp_path, monkeypatch, failing_write
) -> None:
    live_dir = tmp_path / "hooks"
    canary = tmp_path / "should-not-run-4"
    _make_heavy_script(live_dir, "on_train_batch_start", f'touch "{canary}"')
    tmp_root = tmp_path / "tmp-root"
    log = _FakeLog()
    monkeypatch.setattr(failing_write, _raise_no_space)

    result = _dispatch_heavy(
        live_dir,
        tmp_root,
        _context(),
        "occ-1",
        {"batch": {"x": torch.zeros(2)}},
        ("batch",),
        log=log,
    )

    assert result is not None
    assert result.phase == "capture_failed"
    assert result.error is not None and "No space left on device" in result.error
    assert not canary.exists()
    assert len(log.errors) == 1
    assert list(tmp_root.iterdir()) == []


def test_light_payload_write_error_is_a_capture_failure(tmp_path, monkeypatch) -> None:
    live_dir = tmp_path / "hooks"
    canary = tmp_path / "should-not-run-5"
    _make_light_script(live_dir, "on_train_batch_start", f'touch "{canary}"')
    tmp_root = tmp_path / "tmp-root"
    log = _FakeLog()
    monkeypatch.setattr("midrun.hooks.dispatch_light.write_json_file", _raise_no_space)

    result = _dispatch(live_dir, tmp_root, _context(), log=log)

    assert result is not None
    assert result.phase == "capture_failed"
    assert result.error is not None and "No space left on device" in result.error
    assert not canary.exists()
    assert len(log.errors) == 1
    assert list(tmp_root.iterdir()) == []


def test_seeded_heavy_example_counts_the_exported_tensors(tmp_path) -> None:
    if shutil.which("python3") is None:
        pytest.skip("python3 is not on PATH in this environment")
    live_dir = bootstrap_hooks_tree(
        tmp_path / "run", None, session_id="s-1", log=logger
    ).live_dir
    script = live_dir / "heavy" / "on_train_batch_start.sh"
    preamble, example = script.read_text().split(
        "# Example: summarize the manifest without loading any array.\n"
    )
    uncommented = "".join(
        f"{line.removeprefix('# ')}\n" for line in example.splitlines()
    )
    script.write_text(preamble + uncommented)
    script.chmod(0o755)

    result = _dispatch_heavy(
        live_dir,
        tmp_path / "tmp-root",
        _context(),
        "occ-1",
        {"batch": {"x": torch.zeros(2), "y": torch.ones(3)}},
        ("batch",),
    )

    assert result is not None
    assert result.exit_code == 0
    assert result.stdout_tail.endswith("2 tensor(s)\n")


def _reject_constant(name: str):
    raise ValueError(f"non-standard JSON constant {name}")


_NON_FINITE_ARGUMENTS = {
    "loss": float("nan"),
    "metrics": {"high": numpy.float64("inf"), "low": float("-inf")},
}
_NON_FINITE_DESCRIBED = {
    "loss": {"kind": "float", "value": "nan"},
    "metrics": {
        "high": {"kind": "float", "value": "inf"},
        "low": {"kind": "float", "value": "-inf"},
    },
}


def test_light_params_describe_non_finite_floats_as_standard_json(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(live_dir, "on_train_batch_start", 'cat "$1"')

    result = _dispatch(
        live_dir, tmp_path / "tmp-root", _context(), arguments=_NON_FINITE_ARGUMENTS
    )

    assert result is not None
    params = json.loads(result.stdout_tail, parse_constant=_reject_constant)
    assert params["arguments"] == _NON_FINITE_DESCRIBED


def test_heavy_manifest_describes_non_finite_floats_as_standard_json(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_heavy_script(live_dir, "on_train_batch_start", 'cat "$1"')

    result = _dispatch_heavy(
        live_dir,
        tmp_path / "tmp-root",
        _context(),
        "occ-1",
        {"batch": _NON_FINITE_ARGUMENTS},
        ("batch",),
    )

    assert result is not None
    manifest = json.loads(result.stdout_tail, parse_constant=_reject_constant)
    assert manifest["arguments"] == {"batch": _NON_FINITE_DESCRIBED}
