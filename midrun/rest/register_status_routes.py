"""Register run status and telemetry endpoints."""

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI

from midrun.runtime import inspectors_registry
from midrun.runtime.runtime import MidrunRuntime
from midrun.runtime.state.host_learning_rates import host_learning_rates
from midrun.runtime.state.host_metrics import host_metrics
from midrun.runtime.state.host_snapshot import host_snapshot


def register_status_routes(app: FastAPI, runtime: MidrunRuntime) -> None:
    """Register run status and telemetry endpoints."""

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Reports liveness and identifies the run."""
        return {"status": "ok", "run_id": runtime.run_id}

    @app.get("/run")
    def get_run() -> dict[str, Any]:
        """Returns the published runtime snapshot."""
        return asdict(host_snapshot(runtime.state.read()))

    @app.get("/metrics")
    def get_metrics() -> dict[str, Any]:
        """Returns the latest host-readable metrics."""
        metrics = host_metrics(runtime.state.read().metrics)
        return {key: asdict(metric) for key, metric in metrics.items()}

    @app.get("/optimizers")
    def get_optimizers() -> dict[str, Any]:
        """Lists published optimizer learning rates."""
        rates = host_learning_rates(runtime.state.read().learning_rates)
        return {"learning_rates": rates}

    @app.get("/telemetry/pipeline")
    def telemetry_pipeline() -> dict[str, Any]:
        """Reports the latest pipeline sample and aggregate timings."""
        latest = runtime.pipeline.latest()
        return {
            "latest": asdict(latest) if latest else None,
            "summary": runtime.pipeline.summary(),
        }

    @app.get("/inspectors")
    def get_inspectors() -> dict[str, Any]:
        """Reports available model inspectors."""
        return inspectors_registry.availability_snapshot()
