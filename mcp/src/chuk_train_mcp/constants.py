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


def api_run(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}"


def api_run_logs(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}/logs"


def api_run_events(run_id: str) -> str:
    return f"{API_RUNS}/{run_id}/events"


# Defaults mirrored from chuk-train-proto.
DEFAULT_SHELL_TIMEOUT_S: Final = 600
DEFAULT_LOG_TAIL_LINES: Final = 100
DEFAULT_RUN_LIST_LIMIT: Final = 50

HTTP_TIMEOUT_S: Final = 30.0
