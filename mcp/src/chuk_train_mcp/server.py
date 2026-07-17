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
from pydantic import BaseModel

from . import __version__
from .client import ControlPlaneClient, ControlPlaneError
from .constants import (
    DEFAULT_ARTIFACT_URL_TTL_S,
    DEFAULT_LOG_TAIL_LINES,
    DEFAULT_METRIC_DOWNSAMPLE,
    DEFAULT_RUN_LIST_LIMIT,
    DEFAULT_SHELL_TIMEOUT_S,
    DEFAULT_TRAIN_TIMEOUT_S,
    api_run,
    api_run_checkpoint_pin,
    api_run_checkpoints,
    api_run_events,
    api_run_logs,
    api_run_metrics,
    API_ARTIFACT_URL,
    API_CODE_UNITS,
    API_FLEET,
    API_RUNS,
    API_RUNS_SHELL,
)
from .models import (
    BuildCodeUnitRequest,
    CheckpointInfo,
    CodeRef,
    CodeUnitInfo,
    Envelope,
    LogsResponse,
    MetricSeries,
    PinCheckpointRequest,
    RunEvent,
    RunRecord,
    RunSummary,
    SignedUrl,
    SubmitRunRequest,
    SubmitRunResponse,
    SubmitShellRequest,
    TrainSpec,
    WorkerInfo,
)

_PARAM_LIMIT = "limit"
_PARAM_LINES = "lines"
_PARAM_KEYS = "keys"
_PARAM_SINCE_STEP = "since_step"
_PARAM_DOWNSAMPLE = "downsample"
_PARAM_KEY = "key"
_PARAM_TTL_S = "ttl_s"
_METRIC_KEYS_SEP = ","


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

    # -- M1: code units, train runs, metrics, checkpoints, artifacts --------

    @mcp.tool
    async def build_code_unit(
        repo: str, commit: str | None = None, name: str | None = None
    ) -> dict[str, Any]:
        """Build a deployable code unit from a repo/commit (or local path).

        Tars the tree, pins the manifest + lockfile, content-addresses it, and
        registers it. Returns the code ref (name + sha) to pass to submit_run.
        """
        request = BuildCodeUnitRequest(repo=repo, commit=commit, name=name)
        return await _envelope(cp.post_model(API_CODE_UNITS, request, CodeUnitInfo))

    @mcp.tool
    async def submit_run(
        name: str,
        code_name: str,
        code_sha: str,
        entrypoint: str = "train",
        config: str | None = None,
        overrides: dict[str, Any] | None = None,
        seed: int | None = None,
        arch: str | None = None,
        timeout_s: int = DEFAULT_TRAIN_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Queue a train run against a built code unit (spec §5.1 JobSpec).

        code_name/code_sha come from build_code_unit. The run is assigned to an
        idle worker, checkpoints upload with lineage, and it resumes from its
        last checkpoint if the worker is lost mid-run.
        """
        spec = TrainSpec(
            code=CodeRef(name=code_name, sha=code_sha),
            entrypoint=entrypoint,
            config=config,
            overrides=overrides or {},
            seed=seed,
            arch=arch,
            timeout_s=timeout_s,
        )
        request = SubmitRunRequest(name=name, spec=spec)
        return await _envelope(cp.post_model(API_RUNS, request, SubmitRunResponse))

    @mcp.tool
    async def run_metrics(
        run_id: str,
        keys: list[str] | None = None,
        since_step: int = 0,
        downsample: int = DEFAULT_METRIC_DOWNSAMPLE,
    ) -> dict[str, Any]:
        """Metric series for a run: key -> points, ascending by step."""
        params: dict[str, Any] = {_PARAM_SINCE_STEP: since_step, _PARAM_DOWNSAMPLE: downsample}
        if keys:
            params[_PARAM_KEYS] = _METRIC_KEYS_SEP.join(keys)
        return await _envelope(cp.get_model(api_run_metrics(run_id), MetricSeries, params=params))

    @mcp.tool
    async def list_checkpoints(run_id: str) -> dict[str, Any]:
        """Checkpoints uploaded for a run, with lineage-complete metadata."""
        return await _envelope(cp.get_list(api_run_checkpoints(run_id), CheckpointInfo))

    @mcp.tool
    async def pin_checkpoint(run_id: str, step: int, name: str) -> dict[str, Any]:
        """Pin a run's checkpoint by step, exempting it from retention."""
        request = PinCheckpointRequest(step=step, name=name)
        return await _envelope(cp.post_raw(api_run_checkpoint_pin(run_id), request))

    @mcp.tool
    async def artifact_url(key: str, ttl_s: int = DEFAULT_ARTIFACT_URL_TTL_S) -> dict[str, Any]:
        """Time-limited fetch URL for an artifact key (e.g. a checkpoint file).

        This is how lazarus pulls checkpoints to the Mac (spec §10).
        """
        return await _envelope(
            cp.get_model(API_ARTIFACT_URL, SignedUrl, params={_PARAM_KEY: key, _PARAM_TTL_S: ttl_s})
        )

    return mcp


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


async def _envelope(awaitable: Any) -> dict[str, Any]:
    """Run a client call and wrap the outcome; tools never raise."""
    try:
        result = await awaitable
    except ControlPlaneError as exc:
        return Envelope.failure(exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — envelope pattern: surface, don't crash
        return Envelope.failure("unexpected_error", repr(exc))
    if isinstance(result, list):
        return Envelope.success([_dump(item) for item in result])
    return Envelope.success(_dump(result))


def main() -> None:
    build_server().run_stdio()


if __name__ == "__main__":
    main()
