"""Behavioral tests: Trainctl in a real two-rank Lightning DDP run on CPU (gloo).

Each test runs this file as a script in its own process group, so a deadlocked
run is killed with all its ranks. Lightning's DDP launcher starts rank 1 as a
child process that runs this same file.

Each rank writes `status-rank<N>.json` once its Trainctl services start, and
appends one JSON line per training step to `steps-rank<N>.jsonl`, so a test can
see what each rank observed.

Environment variables of the script:
    TRAINCTL_TEST_DIR: Required. Root directory of this run's files.
    MASTER_PORT: Required. The `torch.distributed` rendezvous port.
    TRAINCTL_TEST_FUSE: Optional, "1" mounts the FUSE tree. Default "0".
    TRAINCTL_TEST_TORCH_DEBUG: Optional, "1" starts the torch debug server. Default "0".
    TRAINCTL_TEST_MAX_STEPS: Optional, default -1 (until stopped or `max_time`).
    TRAINCTL_TEST_PG_TIMEOUT_S: Optional, the process-group timeout. Default 60.
    TRAINCTL_TEST_STEP_SLEEP_S: Optional, a sleep in each training step. Default 0.
    TRAINCTL_TEST_FAIL_BACKWARD_AT_STEP: Optional, the global step whose backward
        raises on every rank. Default -1 (never).
    TRAINCTL_TEST_REST_PORT: Optional, default 18500.
    TRAINCTL_TEST_TORCH_DEBUG_PORT: Optional, default 18600.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import httpx2
import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainctl import TrainctlMixin

_ENV = os.environ
_FUSE_AVAILABLE = Path("/dev/fuse").exists() and shutil.which("fusermount3")


class _FailingBackward(torch.autograd.Function):
    """Identity in forward; raises in backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:
        del ctx
        return x.clone()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor
    ) -> torch.Tensor:
        del ctx, grad
        raise RuntimeError("trainctl test: backward failed")


class _DdpModel(TrainctlMixin, pl.LightningModule):
    """Tiny classifier that records, for each step, what its rank observes.

    Attributes:
        loss_scale: A live-tunable multiplier of the loss.
        net: Small linear classifier.
    """

    trainctl_tunable_hparams = {"loss_scale": {"min": 0.0, "max": 10.0}}

    def __init__(self, test_dir: Path) -> None:
        super().__init__(
            rest_port=int(_ENV.get("TRAINCTL_TEST_REST_PORT", "18500")),
            torch_debug_port=int(_ENV.get("TRAINCTL_TEST_TORCH_DEBUG_PORT", "18600")),
            fuse_enabled=_ENV.get("TRAINCTL_TEST_FUSE", "0") == "1",
            torch_debug_enabled=_ENV.get("TRAINCTL_TEST_TORCH_DEBUG", "0") == "1",
            hooks_enabled=False,
            inspection_strict=True,
        )
        self._test_dir = test_dir
        self._step_sleep = float(_ENV.get("TRAINCTL_TEST_STEP_SLEEP_S", "0"))
        self._fail_backward_at = int(
            _ENV.get("TRAINCTL_TEST_FAIL_BACKWARD_AT_STEP", "-1")
        )
        self.loss_scale = 1.0
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.net(x)
        return logits

    def training_step(  # pylint: disable=arguments-differ
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        rank = self.trainer.global_rank
        record = {
            "step": self.global_step,
            "batch_idx": batch_idx,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "loss_scale": self.loss_scale,
        }
        with (self._test_dir / f"steps-rank{rank}.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        time.sleep(self._step_sleep)
        loss: torch.Tensor = nn.functional.cross_entropy(self(x), y) * self.loss_scale
        if self.global_step == self._fail_backward_at:
            loss = _FailingBackward.apply(loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def setup(self, stage: str) -> None:
        super().setup(stage)
        snapshot = self._trainctl.state.read()
        mountpoint = self._trainctl.fuse_mountpoint
        status = {
            "pid": os.getpid(),
            "run_id": snapshot.run_id,
            "rest_port": snapshot.rest_port,
            "torch_debug_url": snapshot.torch_debug_url,
            "artifacts_path": snapshot.artifacts_path,
            "fuse_mountpoint": None if mountpoint is None else str(mountpoint),
        }
        status_file = self._test_dir / f"status-rank{self.trainer.global_rank}.json"
        status_file.write_text(json.dumps(status))


def main() -> None:
    """Runs the two-rank fit; Lightning starts rank 1 by running this file again."""
    test_dir = Path(_ENV["TRAINCTL_TEST_DIR"])
    generator = torch.Generator().manual_seed(0)
    dataset = TensorDataset(
        torch.randn(256, 8, generator=generator),
        torch.randint(0, 2, (256,), generator=generator),
    )
    timeout = timedelta(seconds=float(_ENV.get("TRAINCTL_TEST_PG_TIMEOUT_S", "60")))
    trainer = pl.Trainer(
        default_root_dir=str(test_dir / "lightning"),
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(timeout=timeout),
        max_epochs=-1,
        max_steps=int(_ENV.get("TRAINCTL_TEST_MAX_STEPS", "-1")),
        max_time="00:00:02:00",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(_DdpModel(test_dir), DataLoader(dataset, batch_size=8))


class _DdpRun:
    """One launched two-rank run and the files its ranks write.

    Attributes:
        test_dir: Root directory of the run's files.
        proc: The rank-0 launcher process; rank 1 is its child.
    """

    def __init__(self, test_dir: Path, master_port: int, env: dict[str, str]) -> None:
        self.test_dir = test_dir
        self._log = (test_dir / "out.log").open("w")
        self.proc = subprocess.Popen(  # noqa: S603 -- own fixture
            [sys.executable, __file__],
            env={
                **os.environ,
                "TRAINCTL_TEST_DIR": str(test_dir),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(master_port),
                **env,
            },
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def log_text(self) -> str:
        return (self.test_dir / "out.log").read_text()

    def status(self, rank: int, timeout: float = 60.0) -> dict:
        status_file = self.test_dir / f"status-rank{rank}.json"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if status_file.exists():
                try:
                    return json.loads(status_file.read_text())
                except json.JSONDecodeError:
                    pass
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"run exited with {self.proc.returncode} before rank {rank}"
                    f" started:\n{self.log_text()}"
                )
            time.sleep(0.1)
        raise TimeoutError(
            f"rank {rank} did not start within {timeout}s:\n{self.log_text()}"
        )

    def wait(self, timeout: float) -> int:
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            raise AssertionError(
                f"run did not exit within {timeout}s:\n{self.log_text()}"
            ) from None

    def kill(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=10)
        self._log.close()


@pytest.fixture
def launch(tmp_path: Path) -> Iterator:
    runs: list[_DdpRun] = []

    def _launch(**env: str) -> _DdpRun:
        test_dir = tmp_path / f"run{len(runs)}"
        test_dir.mkdir()
        run = _DdpRun(test_dir, 29700 + os.getpid() % 200 + len(runs), env)
        runs.append(run)
        return run

    yield _launch
    for run in runs:
        run.kill()


@pytest.mark.skipif(not _FUSE_AVAILABLE, reason="FUSE is not available")
def test_the_fuse_mount_starts_without_a_rank_zero_only_collective(launch) -> None:
    run = launch(
        TRAINCTL_TEST_FUSE="1",
        TRAINCTL_TEST_MAX_STEPS="20",
        TRAINCTL_TEST_PG_TIMEOUT_S="20",
    )
    assert run.status(0)["fuse_mountpoint"] is not None
    assert run.status(1)["fuse_mountpoint"] is None
    assert run.wait(timeout=60) == 0


def test_the_torch_debug_server_runs_on_every_rank_with_one_frontend(launch) -> None:
    run = launch(
        TRAINCTL_TEST_TORCH_DEBUG="1",
        TRAINCTL_TEST_MAX_STEPS="30",
        TRAINCTL_TEST_STEP_SLEEP_S="0.2",
    )
    rank0, rank1 = run.status(0), run.status(1)
    assert rank0["rest_port"] is not None
    assert rank0["torch_debug_url"] is not None
    assert rank1["torch_debug_url"] is None
    assert httpx2.get(rank0["torch_debug_url"], timeout=10).status_code == 200
    assert run.wait(timeout=60) == 0
    assert "listen on all network interfaces" in run.log_text()


def test_a_backward_exception_on_every_rank_ends_the_run(launch) -> None:
    # Only rank 0 has REST, so an exception hold on another rank could never end.
    run = launch(
        TRAINCTL_TEST_FAIL_BACKWARD_AT_STEP="3", TRAINCTL_TEST_PG_TIMEOUT_S="20"
    )
    assert run.wait(timeout=60) != 0
    assert "trainctl test: backward failed" in run.log_text()


if __name__ == "__main__":
    main()
