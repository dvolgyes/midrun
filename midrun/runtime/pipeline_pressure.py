"""Cheap, always-on wall-clock phase timing for the Lightning training loop.

Not a profiler -- a small rolling summary of where time goes at the Lightning-hook
level (input gap, forward+loss, backward, post-backward), built from a handful of
`time.perf_counter_ns()` calls per batch. See `midrun.runtime.current_step` for the
separate concern of retaining a batch/loss/exception across hook boundaries.
"""

import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PipelinePressureSample:
    """One batch's phase timing breakdown, in nanoseconds.

    :ivar epoch: Epoch this batch belonged to.
    :ivar global_step: Global step after this batch completed; it can remain unchanged
        across batches when gradient accumulation is enabled.
    :ivar batch_idx: Batch index within the epoch.
    :ivar input_gap_ns: Time between the previous batch's end and this batch's start --
        some mix of DataLoader wait, batch transfer, and framework overhead; not
        proven to be pure DataLoader time. `None` for the first batch.
    :ivar forward_loss_ns: Time before the first backward and between backward passes,
        including work other than forward/loss. `None` if backward never started.
    :ivar backward_ns: Total time inside backward passes. `None` if the final pass
        has no matching start or end.
    :ivar post_backward_ns: Time from the final backward end to batch end, including
        optimizer work and bookkeeping. `None` without a final backward end.
    :ivar batch_total_ns: Time from batch start to batch end.
    """

    epoch: int
    global_step: int
    batch_idx: int
    input_gap_ns: int | None
    forward_loss_ns: int | None
    backward_ns: int | None
    post_backward_ns: int | None
    batch_total_ns: int | None


@dataclass
class _BatchState:
    """Timestamps and identity for the batch currently being measured.

    :ivar batch_start_ns: Timestamp of the current batch's start.
    :ivar before_backward_ns: Most recent backward start within this batch.
    :ivar after_backward_ns: Most recent backward end within this batch.
    :ivar prior_backward_ns: Total duration of passes preceding the current one.
    :ivar previous_batch_end_ns: End of the last completed batch, across epochs.
    :ivar epoch: Current epoch.
    :ivar batch_idx: Index within the current epoch.
    """

    batch_start_ns: int | None = None
    before_backward_ns: int | None = None
    after_backward_ns: int | None = None
    prior_backward_ns: int = 0
    previous_batch_end_ns: int | None = None
    epoch: int = 0
    batch_idx: int = 0


class PipelinePressureTracker:
    """Records per-batch phase timestamps and keeps a rolling window of samples.

    :ivar window: Configured window length.
    :ivar _samples: Most recent completed batches, oldest first.
    :ivar _batch: Mutable timing state reused across batches.
    """

    window: int
    _samples: deque[PipelinePressureSample]
    _batch: _BatchState

    def __init__(self, window: int) -> None:
        self.window = window
        self._samples = deque(maxlen=window)
        self._batch = _BatchState()

    def batch_start(self, now_ns: int, epoch: int, batch_idx: int) -> None:
        """Records the start of a new batch. Call from `on_train_batch_start`."""
        self._batch.batch_start_ns = now_ns
        self._batch.before_backward_ns = None
        self._batch.after_backward_ns = None
        self._batch.prior_backward_ns = 0
        self._batch.epoch = epoch
        self._batch.batch_idx = batch_idx

    def before_backward(self, now_ns: int) -> None:
        """Records the point just before `backward()`. Call from `on_before_backward`."""
        batch = self._batch
        if batch.before_backward_ns is not None and batch.after_backward_ns is not None:
            batch.prior_backward_ns += (
                batch.after_backward_ns - batch.before_backward_ns
            )
        batch.before_backward_ns = now_ns
        batch.after_backward_ns = None

    def after_backward(self, now_ns: int) -> None:
        """Records the point just after `backward()`. Call from `on_after_backward`."""
        self._batch.after_backward_ns = now_ns

    def batch_end(self, now_ns: int, global_step: int) -> PipelinePressureSample:
        """Finalizes a batch using the global step observed at batch end."""
        batch = self._batch
        input_gap_ns = (
            batch.batch_start_ns - batch.previous_batch_end_ns
            if batch.batch_start_ns is not None
            and batch.previous_batch_end_ns is not None
            else None
        )
        forward_loss_ns = (
            batch.before_backward_ns - batch.batch_start_ns - batch.prior_backward_ns
            if batch.before_backward_ns is not None and batch.batch_start_ns is not None
            else None
        )
        backward_ns = (
            batch.prior_backward_ns + batch.after_backward_ns - batch.before_backward_ns
            if batch.after_backward_ns is not None
            and batch.before_backward_ns is not None
            else None
        )
        post_backward_ns = (
            now_ns - batch.after_backward_ns
            if batch.after_backward_ns is not None
            else None
        )
        batch_total_ns = (
            now_ns - batch.batch_start_ns if batch.batch_start_ns is not None else None
        )
        sample = PipelinePressureSample(
            epoch=batch.epoch,
            batch_idx=batch.batch_idx,
            global_step=global_step,
            input_gap_ns=input_gap_ns,
            forward_loss_ns=forward_loss_ns,
            backward_ns=backward_ns,
            post_backward_ns=post_backward_ns,
            batch_total_ns=batch_total_ns,
        )
        self._samples.append(sample)
        batch.previous_batch_end_ns = now_ns
        return sample

    def latest(self) -> PipelinePressureSample | None:
        """Returns the most recent sample, or `None` before the first batch completes."""
        return self._samples[-1] if self._samples else None

    def summary(self) -> dict[str, Any]:
        """Computes rolling phase statistics over the current window, lazily on read."""
        samples = list(self._samples)
        if not samples:
            return {"window": self.window, "samples": 0}
        totals = [s.batch_total_ns for s in samples if s.batch_total_ns is not None]
        mean_batch = statistics.mean(totals) if totals else 0
        phase_stats: dict[str, Any] = {}
        for phase in (
            "input_gap_ns",
            "forward_loss_ns",
            "backward_ns",
            "post_backward_ns",
        ):
            values = [
                getattr(s, phase) for s in samples if getattr(s, phase) is not None
            ]
            if not values:
                continue
            phase_stats[phase.removesuffix("_ns")] = {
                "p50_ms": _percentile(values, 0.50) / 1e6,
                "p95_ms": _percentile(values, 0.95) / 1e6,
                "max_ms": max(values) / 1e6,
                "mean_to_batch_ratio": statistics.mean(values) / mean_batch
                if mean_batch
                else None,
            }
        largest_phase = (
            max(phase_stats, key=lambda name: phase_stats[name]["p50_ms"])
            if phase_stats
            else None
        )
        return {
            "window": self.window,
            "samples": len(samples),
            "latest_step": samples[-1].global_step,
            "batch_total_ms": (
                {
                    "p50": _percentile(totals, 0.50) / 1e6,
                    "p95": _percentile(totals, 0.95) / 1e6,
                }
                if totals
                else None
            ),
            "phases": phase_stats,
            "largest_phase": largest_phase,
        }


def _percentile(values: list[int], fraction: float) -> float:
    """Nearest-rank percentile over `values`; not interpolated."""
    ordered = sorted(values)
    return float(ordered[math.ceil(fraction * len(ordered)) - 1])
