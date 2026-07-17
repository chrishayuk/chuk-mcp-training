"""chuk-train-mcp — the MCP tool surface (M0).

A thin async client over the Rust control plane's REST API, so the MCP server
can run anywhere (the Mac, typically) while the control plane lives on Fly.

    CHUK_TRAIN_URL=https://train.example.com \
    CHUK_TRAIN_API_TOKEN=... \
    chuk-train-mcp            # stdio, for mcp-cli / Claude Desktop / Claude Code
"""

from __future__ import annotations

from typing import Any

from chuk_mcp_server import ChukMCPServer

from . import __version__
from .client import ControlPlaneClient, ControlPlaneError
from .constants import (
    DEFAULT_LOG_TAIL_LINES,
    DEFAULT_RUN_LIST_LIMIT,
    DEFAULT_SHELL_TIMEOUT_S,
    api_run,
    api_run_events,
    api_run_logs,
    API_FLEET,
    API_RUNS,
    API_RUNS_SHELL,
)
from .models import (
    Envelope,
    LogsResponse,
    RunEvent,
    RunRecord,
    RunSummary,
    SubmitRunResponse,
    SubmitShellRequest,
    WorkerInfo,
)

_PARAM_LIMIT = "limit"
_PARAM_LINES = "lines"


def build_server(client: ControlPlaneClient | None = None) -> ChukMCPServer:
    cp = client or ControlPlaneClient()
    mcp = ChukMCPServer(
        name="chuk-mcp-training",
        version=__version__,
        description="MCP-controlled remote training harness (M0: fleet, shell runs, logs)",
    )

    @mcp.tool
    async def fleet() -> dict[str, Any]:
        """List all workers: id, GPU/hardware, connection state, heartbeat age, current run."""
        return await _envelope(cp.get_list(API_FLEET, WorkerInfo))

    @mcp.tool
    async def submit_shell(
        name: str, command: str, timeout_s: int = DEFAULT_SHELL_TIMEOUT_S
    ) -> dict[str, Any]:
        """Submit a shell run to the queue (M0 job kind). Returns the run_id.

        The run is assigned to the first idle worker; follow it with
        run_status / tail_logs.
        """
        request = SubmitShellRequest(name=name, command=command, timeout_s=timeout_s)
        return await _envelope(cp.post_model(API_RUNS_SHELL, request, SubmitRunResponse))

    @mcp.tool
    async def list_runs(limit: int = DEFAULT_RUN_LIST_LIMIT) -> dict[str, Any]:
        """Recent runs with state, worker, and exit code."""
        return await _envelope(cp.get_list(API_RUNS, RunSummary, params={_PARAM_LIMIT: limit}))

    @mcp.tool
    async def run_status(run_id: str) -> dict[str, Any]:
        """Full record for one run: state, spec, worker, exit code, timestamps."""
        return await _envelope(cp.get_model(api_run(run_id), RunRecord))

    @mcp.tool
    async def tail_logs(run_id: str, lines: int = DEFAULT_LOG_TAIL_LINES) -> dict[str, Any]:
        """Last N log lines for a run (live-streamed from the worker)."""
        return await _envelope(
            cp.get_model(api_run_logs(run_id), LogsResponse, params={_PARAM_LINES: lines})
        )

    @mcp.tool
    async def run_events(run_id: str) -> dict[str, Any]:
        """The run's append-only lifecycle event log (provenance record)."""
        return await _envelope(cp.get_list(api_run_events(run_id), RunEvent))

    return mcp


async def _envelope(awaitable: Any) -> dict[str, Any]:
    """Run a client call and wrap the outcome; tools never raise."""
    try:
        result = await awaitable
    except ControlPlaneError as exc:
        return Envelope.failure(exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — envelope pattern: surface, don't crash
        return Envelope.failure("unexpected_error", repr(exc))
    if isinstance(result, list):
        return Envelope.success([item.model_dump(mode="json") for item in result])
    return Envelope.success(result.model_dump(mode="json"))


def main() -> None:
    build_server().run_stdio()


if __name__ == "__main__":
    main()
