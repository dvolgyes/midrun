"""Behavioral tests for `LiveCaptureBroker`'s cross-thread hand-off."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from midrun.runtime import live_capture
from midrun.runtime.live_capture import LiveCaptureBroker


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate(), "condition not met before timeout"


def test_request_blocks_until_served_and_returns_result() -> None:
    broker = LiveCaptureBroker()

    def drive() -> None:
        _wait_until(lambda: len(broker._pending) == 1)
        broker.serve(pl_module="marker")

    threading.Thread(target=drive, daemon=True).start()
    result = broker.request(lambda pl_module: f"got {pl_module}", timeout=5.0)

    assert result == "got marker"


def test_request_times_out_when_never_served() -> None:
    broker = LiveCaptureBroker()
    with pytest.raises(TimeoutError):
        broker.request(lambda pl_module: pl_module, timeout=0.05)


def test_request_reraises_the_callables_exception_in_the_caller_thread() -> None:
    broker = LiveCaptureBroker()

    def failing(pl_module: object) -> None:
        del pl_module
        raise ValueError("boom")

    def drive() -> None:
        _wait_until(lambda: len(broker._pending) == 1)
        broker.serve(pl_module=None)

    threading.Thread(target=drive, daemon=True).start()
    with pytest.raises(ValueError, match="boom"):
        broker.request(failing, timeout=5.0)


def test_serve_with_no_pending_requests_is_a_noop() -> None:
    broker = LiveCaptureBroker()
    broker.serve(pl_module=None)


def test_serve_runs_all_pending_requests_independently() -> None:
    broker = LiveCaptureBroker()
    results: list[object] = []

    def drive() -> None:
        _wait_until(lambda: len(broker._pending) == 2)
        broker.serve(pl_module="m")

    threading.Thread(target=drive, daemon=True).start()

    def collect(fn) -> None:
        try:
            results.append(broker.request(fn, timeout=5.0))
        except ValueError as exc:
            results.append(exc)

    t1 = threading.Thread(target=collect, args=(lambda pl: "ok",))
    t2 = threading.Thread(
        target=collect, args=(lambda pl: (_ for _ in ()).throw(ValueError("bad")),)
    )
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert "ok" in results
    assert any(isinstance(r, ValueError) for r in results)


def test_detached_batch_expires_only_the_unclaimed_callback() -> None:
    broker = LiveCaptureBroker()
    first_started = threading.Event()
    release_callback = threading.Event()
    calls = []

    def first(_module: object) -> None:
        first_started.set()
        assert release_callback.wait(5)
        calls.append("first finished")

    def second(_module: object) -> None:
        calls.append("second started")

    with ThreadPoolExecutor(max_workers=3) as pool:
        first_result = pool.submit(broker.request, first, 1.0)
        _wait_until(lambda: len(broker._pending) == 1)
        second_result = pool.submit(broker.request, second, 1.0)
        _wait_until(lambda: len(broker._pending) == 2)
        serving = pool.submit(broker.serve, None)
        try:
            assert first_started.wait(5)
            first_error = first_result.exception(timeout=5)
            second_error = second_result.exception(timeout=5)
        finally:
            release_callback.set()
        serving.result(timeout=5)

    assert calls == ["first finished"]
    assert isinstance(second_error, TimeoutError)
    assert "before the training thread claimed it" in str(second_error)
    assert isinstance(first_error, TimeoutError)
    assert "after the training thread claimed it" in str(first_error)


def test_serve_skips_overdue_capture_before_reader_wakes() -> None:
    broker = LiveCaptureBroker()
    called = threading.Event()
    now = 0.0
    with (
        patch.object(live_capture, "monotonic", lambda: now),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        result = pool.submit(broker.request, lambda _: called.set(), 60)
        _wait_until(lambda: bool(broker._pending))
        now = 60.0
        broker.serve(None)
        error = result.exception(timeout=5)
    assert not called.is_set()
    assert isinstance(error, TimeoutError)
    assert "before the training thread claimed it" in str(error)
