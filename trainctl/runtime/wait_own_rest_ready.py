"""Waits until this run's REST server answers."""

import json
import time


def wait_own_rest_ready(url: str, run_id: str, timeout: float) -> bool:
    """Polls `url` until it reports the expected `run_id`, or `timeout` elapses.

    Checking identity (not just reachability) guards against a narrow residual
    race: some other server -- another Trainctl run, most plausibly -- already
    answering on this exact host:port would otherwise look indistinguishable
    from our own server being ready.
    """
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(  # noqa: S310 -- own localhost server
                url, timeout=1
            ) as response:
                payload = json.loads(response.read())
        except Exception:  # noqa: BLE001 -- readiness probe boundary, not business logic
            time.sleep(0.2)
            continue
        if payload.get("run_id") == run_id:
            return True
        time.sleep(0.2)
    return False
