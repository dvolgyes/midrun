"""Behavioral tests: a fit that raises stops Midrun's services and unmounts FUSE.

Lightning skips `LightningModule.teardown` after an uncaught exception and calls
only the callbacks' `on_exception`, so these tests run a real `Trainer.fit` that
raises, with and without the lifecycle-hooks adapter attached.
"""

import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx2
import lightning.pytorch as PL
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from midrun.mixin import MidrunMixin

_REST_PORT = 18530


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


def _mounted(path: Path) -> bool:
    # os.path.ismount() is False for a dead FUSE mount (stat fails with ENOTCONN).
    mountinfo = Path("/proc/self/mountinfo").read_text().splitlines()
    return any(line.split()[4] == str(path) for line in mountinfo)


@pytest.fixture
def mountpoint(tmp_path: Path):
    path = tmp_path / "procfs"
    yield path
    if _mounted(path):
        subprocess.run(["fusermount3", "-uz", str(path)], check=False)  # noqa: S603, S607 -- test-owned mount


def _fit(model: _FailingModel, tmp_path: Path) -> None:
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randn(8, 1)), 4)
    trainer = PL.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, loader)


@pytest.mark.parametrize("hooks_enabled", [False, True])
def test_failed_fit_stops_services(
    tmp_path: Path, mountpoint: Path, hooks_enabled: bool
) -> None:
    model = _FailingModel(
        rest_port=_REST_PORT + int(hooks_enabled) * 10,
        fuse_mountpoint=mountpoint,
        torch_debug_enabled=False,
        hooks_enabled=hooks_enabled,
    )
    seen: dict[str, object] = {}
    original = model.training_step

    def _record_services(batch, batch_idx):
        if batch_idx == 1:
            seen["rest_url"] = model._midrun.rest_url
            seen["mounted"] = _mounted(mountpoint)
        return original(batch, batch_idx)

    model.training_step = _record_services

    with pytest.raises(RuntimeError, match="boom"):
        _fit(model, tmp_path)

    assert seen["rest_url"] is not None and seen["mounted"] is True
    assert model._midrun.rest_url is None
    with pytest.raises(httpx2.ConnectError):
        httpx2.get(f"{seen['rest_url']}/health", timeout=2)
    assert not _mounted(mountpoint)
    assert model._midrun.state.read().status == "failed"


_MOUNT_AND_WAIT = """
import sys, time
from pathlib import Path
from midrun.config import MidrunConfig
from midrun.load_backend import load_backend
from midrun.runtime.runtime import MidrunRuntime

class _Module:
    device = "cpu"

base = Path(sys.argv[1])
config = MidrunConfig(
    rest_enabled=False,
    torch_debug_enabled=False,
    artifact_dir=base / "artifacts",
    fuse_mountpoint=base / "procfs",
)
runtime = MidrunRuntime(config, load_backend("lightning.pytorch"))
runtime.setup_environment(_Module(), "fit")
print("ready", flush=True)
time.sleep(60)
"""


@pytest.mark.parametrize(
    "signum", [signal.SIGKILL, signal.SIGTERM, signal.SIGINT, signal.SIGHUP]
)
def test_killed_process_leaves_no_stale_mount(
    tmp_path: Path, mountpoint: Path, signum: signal.Signals
) -> None:
    # A file, not a pipe: a child that writes a long crash report to an unread
    # stderr pipe blocks before it exits, and readline() below then never returns.
    stderr_path = tmp_path / "stderr.txt"
    with stderr_path.open("w") as stderr_file:
        proc = subprocess.Popen(  # noqa: S603 -- fixed interpreter and script
            [sys.executable, "-c", _MOUNT_AND_WAIT, str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
        )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    if line != "ready":
        proc.kill()
        proc.wait(timeout=10)
        pytest.fail(f"mount process did not start: {stderr_path.read_text()}")
    assert _mounted(mountpoint)
    proc.send_signal(signum)
    proc.wait(timeout=10)

    deadline = time.monotonic() + 5
    while _mounted(mountpoint) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _mounted(mountpoint)
