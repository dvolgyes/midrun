"""A single bounded `torch.profiler` run, armed by the `profile` command.

Plain, mutable, single-owner, training-thread-only state -- matching
`midrun.runtime.current_step.CurrentStep`'s style. Orchestrates `torch.profiler`
itself; no profiling logic is reimplemented here.
"""

import tempfile
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from midrun.runtime.artifacts import ArtifactStore

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

        A failed step stops the profiler before the error propagates, so that it
        does not stay enabled.

        Returns:
            True once `warmup_steps + active_steps` full batches have been stepped.
        """
        try:
            self._prof.step()
        except Exception:
            self.stop()
            raise
        self.steps_done += 1
        return self.steps_done > self.warmup_steps + self.active_steps

    def stop(self) -> None:
        """Stops the profiler without exporting anything."""
        self._prof.stop()

    def save(self, artifact_store: ArtifactStore) -> None:
        """Stops the profiler and saves what it recorded as a `profiler` debug capture.

        A run cut short before its last active step, e.g. by the end of training, is
        saved with `truncated` and the number of full `recorded_steps` in its
        metadata. A run that recorded nothing yet is only stopped, with a warning.
        """
        if self.steps_done <= self.warmup_steps:
            self.stop()
            logger.warning(
                "midrun: profile {} stopped before it recorded anything "
                "({} of {} wait and warmup steps done); nothing was saved",
                self.id,
                self.steps_done,
                1 + self.warmup_steps,
            )
            return
        metadata: dict[str, Any] = {"level": self.level}
        recorded_steps = self.steps_done - 1 - self.warmup_steps
        if recorded_steps < self.active_steps:
            metadata |= {"truncated": True, "recorded_steps": recorded_steps}
        files, finalize_metadata = self.finalize()
        artifact_store.write_debug_capture(
            "profiler", files, metadata | finalize_metadata
        )

    def finalize(self) -> tuple[dict[str, str | bytes], dict[str, Any]]:
        """Stops the profiler and returns its exported artifact contents.

        A failed summary does not lose the trace: the capture keeps `trace.json`
        and records the error as `summary_error`.

        Returns:
            `{"trace.json": <chrome trace bytes>, "summary.txt": <key-averages text>}`,
            and the metadata to add to the capture's own.
        """
        self._prof.stop()
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.json"
            self._prof.export_chrome_trace(str(trace_path))
            trace_bytes = trace_path.read_bytes()
        try:
            summary_text = self._prof.key_averages().table()
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:  # noqa: BLE001 -- keep the trace; record the error
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("midrun: profile {} summary failed: {}", self.id, error)
            return {"trace.json": trace_bytes}, {"summary_error": error}
        return {"trace.json": trace_bytes, "summary.txt": summary_text}, {}

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
