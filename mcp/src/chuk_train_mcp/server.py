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
    DEFAULT_BUDGET_PERIOD,
    DEFAULT_LOG_TAIL_LINES,
    DEFAULT_METRIC_DOWNSAMPLE,
    DEFAULT_RUN_LIST_LIMIT,
    DEFAULT_SHELL_TIMEOUT_S,
    DEFAULT_TRAIN_TIMEOUT_S,
    api_run,
    api_run_archive,
    api_run_checkpoint_pin,
    api_run_checkpoints,
    api_run_events,
    api_run_from_experiment,
    api_run_logs,
    api_run_metrics,
    api_run_resume,
    api_run_stop,
    api_worker_extend,
    api_worker_lease,
    api_worker_teardown,
    api_worker_telemetry,
    API_ME,
    API_ARTIFACT_URL,
    API_CODE_UNITS,
    API_COLAB_CELL,
    API_FLEET,
    API_PROVIDER_OFFERS,
    API_PROVISION,
    API_RUNS,
    API_RUNS_SHELL,
    API_ARCHIVE,
    API_BUDGETS,
    API_SPEND,
)
from .models import (
    Budget,
    BuildCodeUnitRequest,
    CheckpointInfo,
    CodeRef,
    CodeUnitInfo,
    ColabCell,
    Envelope,
    ExtendLeaseRequest,
    Lease,
    LogsResponse,
    MetricSeries,
    Offer,
    PinCheckpointRequest,
    ProvisionRequest,
    ProvisionResult,
    RunEvent,
    RunRecord,
    RunSummary,
    SetBudgetRequest,
    SignedUrl,
    SpendReport,
    SubmitRunRequest,
    SubmitRunResponse,
    SubmitShellRequest,
    TeardownRequest,
    TeardownResult,
    TrainSpec,
    WhoAmI,
    WorkerInfo,
    WorkerTelemetry,
)

_PARAM_LIMIT = "limit"
_PARAM_OFFSET = "offset"
_PARAM_STATE = "state"
_PARAM_EXPERIMENT_REF = "experiment_ref"
_PARAM_LINES = "lines"
_PARAM_KEYS = "keys"
_PARAM_SINCE_STEP = "since_step"
_PARAM_DOWNSAMPLE = "downsample"
_PARAM_KEY = "key"
_PARAM_TTL_S = "ttl_s"
_PARAM_PERIOD = "period"
_PARAM_SCOPE = "scope"
_METRIC_KEYS_SEP = ","


def build_server(client: ControlPlaneClient | None = None) -> ChukMCPServer:
    cp = client or ControlPlaneClient()
    mcp = ChukMCPServer(
        name="chuk-mcp-training",
        version=__version__,
        description="MCP-controlled remote training harness (M0: fleet, shell runs, logs)",
    )

    @mcp.tool
    async def whoami() -> dict[str, Any]:
        """The caller's own resolved identity: role (read/write/admin/sysadmin),
        team, and whether a personal experiments-server key is linked.

        Check this before attempting write- or admin-scoped tools — a 403 from
        them means the presented credential's role is too low, not a bug. If
        `experiments_key_set` is false, mirrored runs report under the shared
        default identity; link a personal key on the dashboard's Team screen
        (never paste an API key through this tool surface).
        """
        return await _envelope(cp.get_model(API_ME, WhoAmI))

    @mcp.tool
    async def fleet() -> dict[str, Any]:
        """List all workers: id, GPU/hardware, connection state, heartbeat age, current run."""
        return await _envelope(
            cp.get_list(API_FLEET, WorkerInfo),
            empty_hint="No workers connected. Provision a leased one (provision), "
            "or generate a Colab bootstrap cell (colab_cell).",
        )

    @mcp.tool
    async def worker_telemetry(worker_id: str) -> dict[str, Any]:
        """The worker's latest host telemetry sample (`sys/*`): GPU/CPU/memory
        utilisation, VRAM, temperature, power — plus recent per-key series.

        Use it to judge whether a worker is actually busy before submitting,
        tearing down, or extending a lease. 404s if the worker has never
        reported telemetry (worker ids come from fleet).
        """
        return await _envelope(cp.get_model(api_worker_telemetry(worker_id), WorkerTelemetry))

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
    async def list_runs(
        limit: int = DEFAULT_RUN_LIST_LIMIT,
        offset: int = 0,
        state: str | None = None,
        experiment_ref: str | None = None,
    ) -> dict[str, Any]:
        """Recent runs with state, worker, and exit code — newest first.

        Filter with `state` (queued|assigned|running|completed|failed|cancelled)
        or `experiment_ref` (an experiments-server RUN-… id: every execution of
        that logical run). Page with limit/offset; a full page means there may
        be more.
        """
        params: dict[str, Any] = {_PARAM_LIMIT: limit, _PARAM_OFFSET: offset}
        if state is not None:
            params[_PARAM_STATE] = state
        if experiment_ref is not None:
            params[_PARAM_EXPERIMENT_REF] = experiment_ref
        filtered = state is not None or experiment_ref is not None or offset > 0
        return await _envelope(
            cp.get_list(API_RUNS, RunSummary, params=params),
            empty_hint=(
                "No runs matched this filter/page — try dropping the state/"
                "experiment_ref filter or resetting offset to 0."
                if filtered
                else "No runs recorded yet. Queue one with submit_shell or submit_run."
            ),
        )

    @mcp.tool
    async def run_status(run_id: str) -> dict[str, Any]:
        """Full record for one run: state, spec, worker, exit code, timestamps."""
        return await _envelope(cp.get_model(api_run(run_id), RunRecord))

    @mcp.tool
    async def stop_run(run_id: str) -> dict[str, Any]:
        """Cancel a run. Signals its worker to stop the process
        (SIGTERM → grace → SIGKILL); the run lands in `cancelled`. A queued run
        is cancelled immediately. Returns the run's current record — a running
        run may still show `running` until the worker confirms the kill.
        Write-scoped."""
        return await _envelope(cp.post_params(api_run_stop(run_id)))

    @mcp.tool
    async def resume_run(run_id: str) -> dict[str, Any]:
        """Re-queue a terminal run (cancelled/failed/completed) to run again. A
        train run resumes from its latest uploaded checkpoint when reassigned; a
        shell run restarts. Write-scoped."""
        return await _envelope(cp.post_params(api_run_resume(run_id)))

    @mcp.tool
    async def tail_logs(run_id: str, lines: int = DEFAULT_LOG_TAIL_LINES) -> dict[str, Any]:
        """Last N log lines for a run (live-streamed from the worker)."""
        return await _envelope(
            cp.get_model(api_run_logs(run_id), LogsResponse, params={_PARAM_LINES: lines})
        )

    @mcp.tool
    async def run_events(run_id: str) -> dict[str, Any]:
        """The run's append-only lifecycle event log (provenance record)."""
        return await _envelope(
            cp.get_list(api_run_events(run_id), RunEvent),
            empty_hint="No events for this run — check the run_id against list_runs.",
        )

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
        experiment_ref: str | None = None,
        confirm_cost: bool = False,
    ) -> dict[str, Any]:
        """Queue a train run against a built code unit (spec §5.1 JobSpec).

        code_name/code_sha come from build_code_unit. The run is assigned to an
        idle worker, checkpoints upload with lineage, and it resumes from its
        last checkpoint if the worker is lost mid-run.

        Pass experiment_ref to attach this execution to an existing
        experiments-server logical run (its RUN-… id): the CP reports into that
        run rather than minting a new one. Omit it for an unattached scratch run.

        Cost pre-flight (spec §8): when the worst-case estimate (timeout_s at
        the most expensive live lease) exceeds the confirm threshold, the
        submission is refused with the estimate — read it, then resubmit with
        confirm_cost=True only if the cost is genuinely intended.
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
        request = SubmitRunRequest(
            name=name, spec=spec, experiment_ref=experiment_ref, confirm_cost=confirm_cost
        )
        return await _envelope(cp.post_model(API_RUNS, request, SubmitRunResponse))

    @mcp.tool
    async def submit_run_from_experiment(
        run_id: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Submit a train run built from an existing experiments-server run.

        `run_id` is a chuk-experiments-server logical run (its `RUN-…` id,
        e.g. from that server's `enqueue_run`). Its own `config`/`workspec`
        is fetched and used to build the TrainSpec directly — you don't
        re-specify code/entrypoint/overrides/etc. here. Equivalent to calling
        submit_run(..., experiment_ref=run_id) with the spec filled in for
        you. Fails if the run has no entrypoint/code reference recorded, is
        not `queued`, or is already attached to another execution.

        `name` overrides the harness run's own display name (defaults to
        `run_id`) — this is separate from the experiments-server run's own
        slug/title, which is untouched.
        """
        params: dict[str, Any] = {}
        if name:
            params["name"] = name
        return await _envelope(
            cp.post_params(api_run_from_experiment(run_id), params or None)
        )

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
        return await _envelope(
            cp.get_list(api_run_checkpoints(run_id), CheckpointInfo),
            empty_hint="No checkpoints for this run. Train runs upload on their "
            "checkpoint-policy cadence; shell runs never checkpoint.",
        )

    @mcp.tool
    async def pin_checkpoint(run_id: str, step: int, name: str) -> dict[str, Any]:
        """Pin a run's checkpoint by step, exempting it from retention."""
        request = PinCheckpointRequest(step=step, name=name)
        return await _envelope(cp.post_raw(api_run_checkpoint_pin(run_id), request))

    @mcp.tool
    async def archive_run(run_id: str, force: bool = False) -> dict[str, Any]:
        """Archive a run's final checkpoint + logs/metrics to Google Drive now
        (spec §11.5). Idempotent; `force` re-archives an already-archived run.
        Admin-scoped (team admins + sysadmins)."""
        return await _envelope(cp.post_params(api_run_archive(run_id), {"force": force}))

    @mcp.tool
    async def archive_runs() -> dict[str, Any]:
        """Sweep: archive every completed run not yet tiered to Drive (the
        backstop, on demand). Admin-scoped."""
        return await _envelope(cp.post_params(API_ARCHIVE))

    @mcp.tool
    async def archive_status() -> dict[str, Any]:
        """Per-run archive state: each recent run's final checkpoint location
        (R2 hot/final or Drive) and when it was archived."""
        return await _envelope(cp.get_raw(API_ARCHIVE))

    @mcp.tool
    async def artifact_url(key: str, ttl_s: int = DEFAULT_ARTIFACT_URL_TTL_S) -> dict[str, Any]:
        """Time-limited fetch URL for an artifact key (e.g. a checkpoint file).

        This is how lazarus pulls checkpoints to the Mac (spec §10).
        """
        return await _envelope(
            cp.get_model(API_ARTIFACT_URL, SignedUrl, params={_PARAM_KEY: key, _PARAM_TTL_S: ttl_s})
        )

    # -- M2: leases + provisioning -----------------------------------------

    @mcp.tool
    async def provider_offers(
        provider: str, gpu: str | None = None, max_price_hr: float | None = None
    ) -> dict[str, Any]:
        """Rentable GPU offers from a provider, optionally filtered."""
        params: dict[str, Any] = {"provider": provider}
        if gpu is not None:
            params["gpu"] = gpu
        if max_price_hr is not None:
            params["max_price_hr"] = max_price_hr
        return await _envelope(
            cp.get_list(API_PROVIDER_OFFERS, Offer, params=params),
            empty_hint="No offers matched — relax the gpu or max_price_hr filter, "
            "or try another provider.",
        )

    @mcp.tool
    async def provision(
        provider: str,
        lease_min: float,
        offer_id: str | None = None,
        gpu: str | None = None,
        max_price_hr: float | None = None,
    ) -> dict[str, Any]:
        """Provision a leased worker (spec §3). The lease is a hard wall: the
        control plane drains at T-drain and destroys the instance at T-0,
        provider-verified, whether or not the agent responds."""
        request = ProvisionRequest(
            provider=provider, lease_min=lease_min, offer_id=offer_id, gpu=gpu,
            max_price_hr=max_price_hr,
        )
        return await _envelope(cp.post_model(API_PROVISION, request, ProvisionResult))

    @mcp.tool
    async def lease_status(worker_id: str) -> dict[str, Any]:
        """The worker's lease: budget, elapsed, extensions, state."""
        return await _envelope(cp.get_model(api_worker_lease(worker_id), Lease))

    @mcp.tool
    async def extend_lease(worker_id: str, minutes: float, reason: str = "") -> dict[str, Any]:
        """Extend a lease's wall — the only path past it (a budget decision)."""
        request = ExtendLeaseRequest(minutes=minutes, reason=reason)
        return await _envelope(cp.post_model(api_worker_extend(worker_id), request, Lease))

    @mcp.tool
    async def teardown(worker_id: str, force: bool = False) -> dict[str, Any]:
        """Tear down a leased worker now: drain (unless force) then destroy,
        provider-verified. Returns whether the instance was confirmed gone."""
        request = TeardownRequest(force=force)
        return await _envelope(cp.post_model(api_worker_teardown(worker_id), request, TeardownResult))

    @mcp.tool
    async def spend_status(period: str = DEFAULT_BUDGET_PERIOD) -> dict[str, Any]:
        """Committed (live leases) vs spent (realised) per provider, from the
        ledger (spec §8), over `period` ("month" = current UTC calendar month,
        "all" = all-time). Where a budget matches the period, each line (and
        the report globally) carries cap + headroom — check headroom before
        provisioning or extending; those refuse on breach.
        """
        return await _envelope(
            cp.get_model(API_SPEND, SpendReport, params={_PARAM_PERIOD: period})
        )

    @mcp.tool
    async def set_budget(
        scope: str, cap: float, period: str = DEFAULT_BUDGET_PERIOD
    ) -> dict[str, Any]:
        """Set (upsert) a spend cap (spec §8). Admin-scoped.

        scope: "global" or "provider:<name>" (e.g. "provider:vast").
        period: "month" (current UTC calendar month) or "all" (all-time).
        provision and extend_lease refuse when projected spend (realised +
        committed live leases + the candidate cost) would exceed any
        applicable cap. Colab caps are in compute units, not dollars.
        """
        request = SetBudgetRequest(scope=scope, cap=cap, period=period)
        return await _envelope(cp.post_model(API_BUDGETS, request, Budget))

    @mcp.tool
    async def list_budgets() -> dict[str, Any]:
        """Every configured spend cap (scope, cap, period)."""
        return await _envelope(
            cp.get_list(API_BUDGETS, Budget),
            empty_hint="No budgets set — spending is uncapped. Set one with "
            "set_budget (admin-scoped).",
        )

    @mcp.tool
    async def delete_budget(scope: str) -> dict[str, Any]:
        """Remove a spend cap by scope (e.g. "provider:vast"). Admin-scoped."""
        return await _envelope(cp.delete_params(API_BUDGETS, {_PARAM_SCOPE: scope}))

    @mcp.tool
    async def colab_cell(lease_min: float | None = None, labels: str = "colab,t4") -> dict[str, Any]:
        """Generate a ready-to-paste Colab bootstrap cell (spec §6).

        The control plane fills in its own URL + join token; paste the returned
        `cell` into one cell of a T4 notebook and run it to join the fleet.
        Optionally pass lease_min to have the worker self-drain at T-drain.
        """
        params: dict[str, Any] = {"labels": labels}
        if lease_min is not None:
            params["lease_min"] = lease_min
        return await _envelope(cp.get_model(API_COLAB_CELL, ColabCell, params=params))

    return mcp


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


async def _envelope(awaitable: Any, empty_hint: str | None = None) -> dict[str, Any]:
    """Run a client call and wrap the outcome; tools never raise.

    List results carry a `count`; an empty list additionally carries
    `empty_hint` as `message`, so the agent can tell "nothing exists" (and what
    to do about it) from "wrong query" or "tool failure".
    """
    try:
        result = await awaitable
    except ControlPlaneError as exc:
        return Envelope.failure(exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — envelope pattern: surface, don't crash
        return Envelope.failure("unexpected_error", repr(exc))
    if isinstance(result, list):
        return Envelope.success(
            [_dump(item) for item in result],
            count=len(result),
            message=empty_hint if not result else None,
        )
    return Envelope.success(_dump(result))


def main() -> None:
    build_server().run_stdio()


if __name__ == "__main__":
    main()
