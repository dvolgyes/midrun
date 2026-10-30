"""Behavioral tests: Midrun in a real two-rank Lightning DDP run on CPU (gloo).

Each test runs this file as a script in its own process group, so a deadlocked
run is killed with all its ranks. Lightning's DDP launcher starts rank 1 as a
child process that runs this same file.

Each rank writes `status-rank<N>.json` once its Midrun services start, and
appends one JSON line per training step to `steps-rank<N>.jsonl`, so a test can
see what each rank observed.

Environment variables of the script:
    MIDRUN_TEST_DIR: Required. Root directory of this run's files.
    MASTER_PORT: Required. The `torch.distributed` rendezvous port.
    MIDRUN_TEST_FUSE: Optional, "1" mounts the FUSE tree. Default "0".
    MIDRUN_TEST_TORCH_DEBUG: Optional, "1" starts the torch debug server. Default "0".
    MIDRUN_TEST_MAX_STEPS: Optional, default -1 (until stopped or `max_time`).
    MIDRUN_TEST_PG_TIMEOUT_S: Optional, the process-group timeout. Default 60.
    MIDRUN_TEST_STEP_SLEEP_S: Optional, a sleep in each training step. Default 0.
    MIDRUN_TEST_FAIL_BACKWARD_AT_STEP: Optional, the global step whose backward
        raises on every rank. Default -1 (never).
    MIDRUN_TEST_NAN_ON_RANK1_AT_STEP: Optional, the global step whose batch has a
        NaN on rank 1 only. It also turns on the dataloader finite check. Default -1.
    MIDRUN_TEST_REST_PORT: Optional, default 18500.
    MIDRUN_TEST_TORCH_DEBUG_PORT: Optional, default 18600.
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

from midrun import MidrunMixin

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
        raise RuntimeError("midrun test: backward failed")


class _DdpModel(MidrunMixin, pl.LightningModule):
    """Tiny classifier that records, for each step, what its rank observes.

    Attributes:
        loss_scale: A live-tunable multiplier of the loss.
        net: Small linear classifier.
    """

    midrun_tunable_hparams = {"loss_scale": {"min": 0.0, "max": 10.0}}

    def __init__(self, test_dir: Path) -> None:
        nan_at = int(_ENV.get("MIDRUN_TEST_NAN_ON_RANK1_AT_STEP", "-1"))
        super().__init__(
            rest_port=int(_ENV.get("MIDRUN_TEST_REST_PORT", "18500")),
            torch_debug_port=int(_ENV.get("MIDRUN_TEST_TORCH_DEBUG_PORT", "18600")),
            fuse_enabled=_ENV.get("MIDRUN_TEST_FUSE", "0") == "1",
            torch_debug_enabled=_ENV.get("MIDRUN_TEST_TORCH_DEBUG", "0") == "1",
            hooks_enabled=False,
            inspection_strict=True,
            dataloader_finite_check_enabled=nan_at >= 0,
        )
        self._nan_at = nan_at
        self._test_dir = test_dir
        self._step_sleep = float(_ENV.get("MIDRUN_TEST_STEP_SLEEP_S", "0"))
        self._fail_backward_at = int(
            _ENV.get("MIDRUN_TEST_FAIL_BACKWARD_AT_STEP", "-1")
        )
        self.loss_scale = 1.0
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.net(x)
        return logits

    def on_train_batch_start(  # pylint: disable=arguments-differ
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        if self.trainer.global_rank == 1 and self.global_step == self._nan_at:
            batch[0][0, 0] = float("nan")
        super().on_train_batch_start(batch, batch_idx)

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
        snapshot = self._midrun.state.read()
        mountpoint = self._midrun.fuse_mountpoint
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
    test_dir = Path(_ENV["MIDRUN_TEST_DIR"])
    generator = torch.Generator().manual_seed(0)
    dataset = TensorDataset(
        torch.randn(256, 8, generator=generator),
        torch.randint(0, 2, (256,), generator=generator),
    )
    timeout = timedelta(seconds=float(_ENV.get("MIDRUN_TEST_PG_TIMEOUT_S", "60")))
    trainer = pl.Trainer(
        default_root_dir=str(test_dir / "lightning"),
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(timeout=timeout),
        max_epochs=-1,
        max_steps=int(_ENV.get("MIDRUN_TEST_MAX_STEPS", "-1")),
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
                "MIDRUN_TEST_DIR": str(test_dir),
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

    def steps(self, rank: int) -> list[dict]:
        lines = (self.test_dir / f"steps-rank{rank}.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    def get(self, path: str) -> dict:
        url = f"http://127.0.0.1:{self.status(0)['rest_port']}{path}"
        response = httpx2.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, body: dict) -> dict:
        url = f"http://127.0.0.1:{self.status(0)['rest_port']}{path}"
        response = httpx2.post(url, json=body, timeout=10)
        response.raise_for_status()
        return response.json()

    def wait_until(self, predicate, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError(f"condition not met within {timeout}s")
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"run exited with {self.proc.returncode}:\n{self.log_text()}"
                )
            time.sleep(0.1)

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
        MIDRUN_TEST_FUSE="1",
        MIDRUN_TEST_MAX_STEPS="20",
        MIDRUN_TEST_PG_TIMEOUT_S="20",
    )
    assert run.status(0)["fuse_mountpoint"] is not None
    assert run.status(1)["fuse_mountpoint"] is None
    assert run.wait(timeout=60) == 0


def test_the_torch_debug_server_runs_on_every_rank_with_one_frontend(launch) -> None:
    run = launch(
        MIDRUN_TEST_TORCH_DEBUG="1",
        MIDRUN_TEST_MAX_STEPS="30",
        MIDRUN_TEST_STEP_SLEEP_S="0.2",
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
    run = launch(MIDRUN_TEST_FAIL_BACKWARD_AT_STEP="3", MIDRUN_TEST_PG_TIMEOUT_S="20")
    assert run.wait(timeout=60) != 0
    assert "midrun test: backward failed" in run.log_text()


def test_the_run_state_lists_every_rank(launch) -> None:
    run = launch(MIDRUN_TEST_MAX_STEPS="40", MIDRUN_TEST_STEP_SLEEP_S="0.1")
    rank0, rank1 = run.status(0), run.status(1)
    url = f"http://127.0.0.1:{rank0['rest_port']}/run"
    snapshot = httpx2.get(url, timeout=10).json()
    assert snapshot["world_size"] == 2
    assert [(r["global_rank"], r["pid"]) for r in snapshot["ranks"]] == [
        (0, rank0["pid"]),
        (1, rank1["pid"]),
    ]
    assert run.wait(timeout=60) == 0


def test_a_learning_rate_and_hparam_change_reach_every_rank(launch) -> None:
    run = launch(MIDRUN_TEST_MAX_STEPS="60", MIDRUN_TEST_STEP_SLEEP_S="0.05")
    run.status(1)
    run.post("/commands", {"kind": "set_learning_rate", "args": {"value": 0.05}})
    run.post(
        "/commands",
        {"kind": "set_hparam", "args": {"name": "loss_scale", "value": 2.0}},
    )
    assert run.wait(timeout=60) == 0
    for rank in (0, 1):
        last = run.steps(rank)[-1]
        assert (rank, last["lr"], last["loss_scale"]) == (rank, 0.05, 2.0)


def test_a_checkpoint_completes_on_every_rank(launch) -> None:
    run = launch(
        MIDRUN_TEST_MAX_STEPS="60",
        MIDRUN_TEST_STEP_SLEEP_S="0.05",
        MIDRUN_TEST_PG_TIMEOUT_S="20",
    )
    run.status(1)
    command_id = run.post("/checkpoints", {"label": "manual"})["id"]
    run.wait_until(lambda: run.get(f"/commands/{command_id}")["status"] != "queued")
    command = run.get(f"/commands/{command_id}")
    assert command["status"] == "succeeded", command
    assert Path(command["result"]["checkpoint_path"]).is_file()
    assert run.wait(timeout=60) == 0


def test_stop_ends_every_rank(launch) -> None:
    run = launch(MIDRUN_TEST_STEP_SLEEP_S="0.05", MIDRUN_TEST_PG_TIMEOUT_S="20")
    run.status(1)
    run.post("/control/stop", {})
    assert run.wait(timeout=30) == 0


def test_stop_while_held_ends_every_rank(launch) -> None:
    run = launch(MIDRUN_TEST_STEP_SLEEP_S="0.05", MIDRUN_TEST_PG_TIMEOUT_S="20")
    run.status(1)
    run.post("/control/pause", {})
    run.wait_until(lambda: run.get("/run")["status"] == "held")
    run.post("/control/stop", {})
    assert run.wait(timeout=30) == 0


def test_a_hold_outlasts_the_process_group_timeout(launch) -> None:
    run = launch(MIDRUN_TEST_STEP_SLEEP_S="0.05", MIDRUN_TEST_PG_TIMEOUT_S="10")
    run.status(1)
    run.post("/control/pause", {})
    run.wait_until(lambda: run.get("/run")["status"] == "held")
    held_steps = [len(run.steps(rank)) for rank in (0, 1)]
    time.sleep(15)
    assert [len(run.steps(rank)) for rank in (0, 1)] == held_steps
    assert run.post("/control/continue", {})["released"]
    run.wait_until(lambda: all(len(run.steps(r)) > held_steps[r] + 5 for r in (0, 1)))
    run.post("/control/stop", {})
    assert run.wait(timeout=30) == 0


def test_a_bad_batch_on_one_rank_holds_every_rank(launch) -> None:
    run = launch(
        MIDRUN_TEST_NAN_ON_RANK1_AT_STEP="5",
        MIDRUN_TEST_STEP_SLEEP_S="0.05",
        MIDRUN_TEST_PG_TIMEOUT_S="20",
    )
    run.status(1)
    run.wait_until(lambda: run.get("/run")["status"] == "held")
    held_steps = [len(run.steps(rank)) for rank in (0, 1)]
    time.sleep(2)
    assert [len(run.steps(rank)) for rank in (0, 1)] == held_steps == [5, 5]
    snapshots = Path(run.status(1)["artifacts_path"]) / "snapshots"
    assert list(snapshots.glob("*/anomaly_input_nonfinite.json"))
    assert run.post("/control/continue", {})["released"]
    run.wait_until(lambda: all(len(run.steps(r)) > 10 for r in (0, 1)))
    run.post("/control/stop", {})
    assert run.wait(timeout=30) == 0


if __name__ == "__main__":
    main()
