"""A single bounded `torch.profiler` run, armed by the `profile` command.

Plain, mutable, single-owner, training-thread-only state -- matching
`midrun.runtime.current_step.CurrentStep`'s style. Orchestrates `torch.profiler`
itself; no profiling logic is reimplemented here.
"""

import tempfile
from pathlib import Path
from typing import Any

import torch

_LEVEL_KWARGS: dict[str, dict[str, Any]] = {
    "basic": {},
    "shapes": {"record_shapes": True},
    "memory": {"record_shapes": True, "profile_memory": True},
    "full": {"record_shapes": True, "profile_memory": True, "with_stack": True},
}


class ProfilerRun:
    """One armed-through-finalized bounded profiling run.

    The run starts at a safe point inside a batch, possibly in a hold. That rest of
    the batch, which ends at the first `step()`, is not a counted step, so the
    first warmup or active step is a full batch.

    Attributes:
        id: The `profile` command's id, used to label the resulting artifact.
        warmup_steps: Full batches to run under the profiler before recording starts.
        active_steps: Full batches to actually record after warmup.
        level: One of `basic`, `shapes`, `memory`, `full` (see `_LEVEL_KWARGS`).
        then_hold: Whether to enter a cooperative hold once this run finalizes.
        steps_done: Number of `step()` calls made so far, the first included.
    """

    def __init__(
        self,
        run_id: str,
        warmup_steps: int,
        active_steps: int,
        level: str,
        then_hold: bool,
    ) -> None:
        if level not in _LEVEL_KWARGS:
            raise ValueError(f"unknown profiler level {level!r}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if active_steps < 1:
            raise ValueError(f"active_steps must be >= 1, got {active_steps}")
        self.id = run_id
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.level = level
        self.then_hold = then_hold
        self.steps_done = 0
        self._prof: Any = None

    def start(self) -> None:
        """Builds and starts the underlying `torch.profiler.profile`."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        sched = torch.profiler.schedule(
            wait=1, warmup=self.warmup_steps, active=self.active_steps, repeat=1
        )
        self._prof = torch.profiler.profile(
            activities=activities, schedule=sched, **_LEVEL_KWARGS[self.level]
        )
        self._prof.start()

    def step(self) -> bool:
        """Advances the profiler at the end of a batch.

        Returns:
            True once `warmup_steps + active_steps` full batches have been stepped.
        """
        self._prof.step()
        self.steps_done += 1
        return self.steps_done > self.warmup_steps + self.active_steps

    def finalize(self) -> dict[str, str | bytes]:
        """Stops the profiler and returns its exported artifact contents.

        Returns:
            `{"trace.json": <chrome trace bytes>, "summary.txt": <key-averages text>}`.
        """
        self._prof.stop()
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.json"
            self._prof.export_chrome_trace(str(trace_path))
            trace_bytes = trace_path.read_bytes()
        summary_text = self._prof.key_averages().table()
        return {"trace.json": trace_bytes, "summary.txt": summary_text}

    def phase_state(self) -> dict[str, Any]:
        """Live progress, for FUSE/status reporting."""
        if self.steps_done == 0:
            phase = "waiting"
        elif self.steps_done <= self.warmup_steps:
            phase = "warmup"
        else:
            phase = "active"
        return {
            "state": phase,
            "level": self.level,
            "steps_done": self.steps_done,
            "warmup_steps": self.warmup_steps,
            "active_steps": self.active_steps,
            "then_hold": self.then_hold,
        }
