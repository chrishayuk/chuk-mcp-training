"""Names and numbers shared with the Rust control plane (crates/chuk-train-proto)."""

from __future__ import annotations

from typing import Final

# Environment variables (must match chuk_train_proto::constants::env).
ENV_CP_URL: Final = "CHUK_TRAIN_URL"
ENV_API_TOKEN: Final = "CHUK_TRAIN_API_TOKEN"

DEFAULT_CP_URL: Final = "http://127.0.0.1:8700"

# REST paths (must match the control plane's router).
API_FLEET: Final = "/api/fleet"
API_RUNS: Final = "/api/runs"
API_RUNS_SHELL: Final = "/api/runs/shell"
API_CODE_UNITS: Final = "/api/code_units"
API_ARTIFACT_URL: Final = "/api/artifact_url"
API_PROVIDER_OFFERS: Final = "/api/provider_offers"
API_PROVISION: Final = "/api/provision"
API_COLAB_CELL: Final = "/api/colab_cell"
API_SPEND: Final = "/api/spend"
API_ARCHIVE: Final = "/api/archive"
API_ME: Final = "/api/me"


def api_worker_lease(worker_id: str) -> str:
    return f"/api/workers/{worker_id}/lease"


def api_worker_extend(worker_id: str) -> str:
    return f"/api/workers/{worker_id}/extend"


def api_worker_teardown(worker_id: str) -> str:
    return f"/api/workers/{worker_id}/teardown"


def api_worker_telemetry(worker_id: str) -> str:
    return f"/api/workers/{worker_id}/telemetry"


def api_run(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}"


def api_run_from_experiment(run_id: str) -> str:
    return f"{API_RUNS}/from-experiment/{run_id}"


def api_run_stop(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}/stop"


def api_run_resume(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}/resume"


def api_run_logs(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}/logs"


def api_run_events(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}/events"


def api_run_metrics(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}/metrics"


def api_run_checkpoints(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}/checkpoints"


def api_run_checkpoint_pin(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}/checkpoints/pin"


def api_run_archive(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}/archive"


# Defaults mirrored from chuk-train-proto.
DEFAULT_SHELL_TIMEOUT_S: Final = 600
DEFAULT_TRAIN_TIMEOUT_S: Final = 12 * 3600
DEFAULT_LOG_TAIL_LINES: Final = 100
DEFAULT_RUN_LIST_LIMIT: Final = 50
DEFAULT_METRIC_DOWNSAMPLE: Final = 500
DEFAULT_ARTIFACT_URL_TTL_S: Final = 3600

HTTP_TIMEOUT_S: Final = 30.0
