"""Pydantic mirrors of the Rust wire types (crates/chuk-train-proto/src/wire.rs).

The Rust side is the source of truth; these models validate what crosses the
REST boundary so the MCP tools never hand raw dicts to the model.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .constants import DEFAULT_SHELL_TIMEOUT_S, DEFAULT_TRAIN_TIMEOUT_S


class RunState(StrEnum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerState(StrEnum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


class EventKind(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CHECKPOINT = "checkpoint"
    SLICED = "sliced"
    RESUMED = "resumed"
    GATE_EVALUATED = "gate_evaluated"


class RunKind(StrEnum):
    SHELL = "shell"
    TRAIN = "train"


class ArtifactKind(StrEnum):
    CODE = "code"
    ENV = "env"
    DATASET = "dataset"
    CHECKPOINT = "checkpoint"
    METRICS = "metrics"
    LOGS = "logs"
    DECK = "deck"


class Hardware(BaseModel):
    host: str
    os: str
    gpu: str | None = None
    vram_mb: int | None = None
    driver: str | None = None


class ShellSpec(BaseModel):
    kind: Literal[RunKind.SHELL] = RunKind.SHELL
    command: str
    timeout_s: int = DEFAULT_SHELL_TIMEOUT_S


class CodeRef(BaseModel):
    name: str
    sha: str


class CheckpointPolicy(BaseModel):
    every_steps: int = 500
    keep_last: int = 3
    keep_every: int = 5000


class ArtifactRef(BaseModel):
    name: str
    kind: ArtifactKind
    sha: str | None = None


class TrainSpec(BaseModel):
    kind: Literal[RunKind.TRAIN] = RunKind.TRAIN
    code: CodeRef
    entrypoint: str
    config: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    artifacts_in: list[ArtifactRef] = Field(default_factory=list)
    checkpoint: CheckpointPolicy = Field(default_factory=CheckpointPolicy)
    seed: int | None = None
    arch: str | None = None
    timeout_s: int = DEFAULT_TRAIN_TIMEOUT_S


class UnitRequires(BaseModel):
    cuda: str | None = None
    min_vram_gb: int | None = None


class CodeUnitManifest(BaseModel):
    name: str
    version: str = ""
    entrypoints: dict[str, str] = Field(default_factory=dict)
    python: str | None = None
    requires: UnitRequires = Field(default_factory=UnitRequires)


class CodeUnitInfo(BaseModel):
    code: CodeRef
    manifest: CodeUnitManifest
    uri: str
    created_at: float


class CheckpointMeta(BaseModel):
    step: int = 0
    seed: int | None = None
    arch: str | None = None
    code: CodeRef | None = None
    config_hash: str | None = None
    tokenizer_hash: str | None = None
    parent_checkpoint: str | None = None
    datasets: list[str] = Field(default_factory=list)
    run_id: str | None = None
    slices: list[list[int]] = Field(default_factory=list)


class CheckpointInfo(BaseModel):
    run_id: str
    step: int
    uri: str
    model_hash: str
    pinned: bool
    pin_name: str | None = None
    meta: CheckpointMeta
    created_at: float


class MetricPoint(BaseModel):
    step: int
    value: float


class MetricSeries(BaseModel):
    run_id: str
    series: dict[str, list[MetricPoint]] = Field(default_factory=dict)


class SignedUrl(BaseModel):
    url: str
    expires_at: float


# -- M2: leases, provisioning, spend ---------------------------------------


class LeaseState(StrEnum):
    ACTIVE = "active"
    DRAINING = "draining"
    DESTROYED = "destroyed"


class InstanceStatus(StrEnum):
    RUNNING = "running"
    GONE = "gone"
    UNKNOWN = "unknown"


class LeaseExtension(BaseModel):
    minutes: float
    at: float
    reason: str = ""


class Lease(BaseModel):
    worker_id: str
    provider: str
    instance_id: str
    price_hr: float
    granted_min: float
    drain_window_min: float
    started_at: float
    state: LeaseState
    extensions: list[LeaseExtension] = Field(default_factory=list)


class Offer(BaseModel):
    id: str
    provider: str
    gpu: str
    price_hr: float
    vram_gb: int | None = None
    region: str | None = None


class ProvisionResult(BaseModel):
    worker_id: str
    lease: Lease
    bootstrap: str = ""


class TeardownResult(BaseModel):
    worker_id: str
    destroyed: bool
    status: InstanceStatus


class SpendLine(BaseModel):
    provider: str
    committed: float
    spent: float
    # Present when a provider:<name> budget exists for the report's period.
    cap: float | None = None
    headroom: float | None = None


class SpendReport(BaseModel):
    period: str = ""
    lines: list[SpendLine] = Field(default_factory=list)
    total_committed: float
    total_spent: float
    # Present when a global budget exists for the report's period.
    global_cap: float | None = None
    global_headroom: float | None = None


class Budget(BaseModel):
    """A spend cap (spec §8): scope `global` or `provider:<name>`, per period
    (`month` = current UTC calendar month, `all` = all-time)."""

    scope: str
    cap: float
    period: str
    updated_at: float


class SetBudgetRequest(BaseModel):
    scope: str
    cap: float
    period: str | None = None


class ColabCell(BaseModel):
    cell: str


class TelemetryPoint(BaseModel):
    ts: float
    value: float


class WorkerTelemetry(BaseModel):
    """A worker's latest host sample (chuk-compute M4 `sys/*`): GPU/CPU/memory
    utilisation, VRAM, temperature, power — plus recent per-key series."""

    worker_id: str
    sampled_at: float
    values: dict[str, float] = Field(default_factory=dict)
    series: dict[str, list[TelemetryPoint]] = Field(default_factory=dict)


class WhoAmI(BaseModel):
    """The caller's own resolved identity: role, team, and whether a personal
    experiments-server key is linked."""

    role: str
    team_id: str | None = None
    subject: str | None = None
    experiments_key_set: bool = False


class WorkerInfo(BaseModel):
    id: str
    labels: list[str]
    hardware: Hardware
    state: WorkerState
    current_run: str | None = None
    joined_at: float
    last_seen: float
    heartbeat_age_s: float
    lease: "Lease | None" = None


class RunSummary(BaseModel):
    id: str
    name: str
    kind: RunKind
    state: RunState
    worker_id: str | None = None
    exit_code: int | None = None
    # The experiments-server logical run (RUN-…) this EXEC-… execution belongs
    # to, if any; None for an unattached scratch run.
    experiment_ref: str | None = None
    # The sweep (SWEEP-…) this run is a child of, if any.
    sweep_id: str | None = None
    # Email of the submitting user (from their session or API key's owner);
    # None for pre-tracking runs or the legacy master token.
    created_by: str | None = None
    created_at: float
    updated_at: float


class RunRecord(RunSummary):
    # Discriminated on spec.kind — a run is a train OR a shell run (mirrors
    # the Rust RunSpec enum; spec: ShellSpec alone rejected every train run).
    spec: Annotated[TrainSpec | ShellSpec, Field(discriminator="kind")]


class RunEvent(BaseModel):
    ts: float
    event: EventKind
    detail: dict[str, Any] = Field(default_factory=dict)


class LogsResponse(BaseModel):
    run_id: str
    lines: list[str]


class SweepSpec(BaseModel):
    """One template fanned out over axes (spec §5.2). Axis paths are `seed`
    or `overrides.<key>`; concurrency 0 = unlimited."""

    template: TrainSpec
    axes: dict[str, list[Any]]
    concurrency: int = 0


class SubmitSweepRequest(BaseModel):
    name: str
    spec: SweepSpec
    confirm_cost: bool = False


class SubmitSweepResponse(BaseModel):
    sweep_id: str
    run_ids: list[str]


class SweepChild(BaseModel):
    run_id: str
    state: RunState
    # Axis path -> the value this child got.
    assignment: dict[str, Any] = Field(default_factory=dict)


class SweepAggregatePoint(BaseModel):
    step: int
    n: int
    mean: float
    std: float
    min: float
    max: float


class SweepStatus(BaseModel):
    sweep_id: str
    name: str
    concurrency: int
    children: list[SweepChild] = Field(default_factory=list)
    key: str
    aggregate: list[SweepAggregatePoint] = Field(default_factory=list)


class GateAction(StrEnum):
    RECORD = "record"
    STOP_RUN = "stop_run"


class GateInfo(BaseModel):
    """A registered gate plus its latest verdict (None until first evaluated)."""

    scope: str
    scope_id: str
    name: str
    expr: str
    action: GateAction
    created_at: float
    tripped: bool | None = None
    last_value: float | None = None
    evaluated_at: float | None = None
    detail: str | None = None


class RegisterGateRequest(BaseModel):
    name: str
    expr: str
    action: GateAction = GateAction.RECORD


class SubmitShellRequest(BaseModel):
    name: str
    command: str
    timeout_s: int = DEFAULT_SHELL_TIMEOUT_S


class SubmitRunRequest(BaseModel):
    name: str
    spec: TrainSpec
    # Optional external parent: the experiments-server logical run (RUN-…) this
    # execution realises. When set, the CP's mirror reports into it instead of
    # minting a new run. Omit for an unattached scratch run.
    experiment_ref: str | None = None
    # Spec §8 pre-flight: required when the worst-case estimate exceeds the
    # control plane's confirm threshold; the refusal carries the estimate.
    confirm_cost: bool = False


class BuildCodeUnitRequest(BaseModel):
    repo: str
    commit: str | None = None
    name: str | None = None


class PinCheckpointRequest(BaseModel):
    step: int
    name: str


class ProvisionRequest(BaseModel):
    provider: str
    lease_min: float
    offer_id: str | None = None
    gpu: str | None = None
    max_price_hr: float | None = None


class ExtendLeaseRequest(BaseModel):
    minutes: float
    reason: str = ""


class TeardownRequest(BaseModel):
    force: bool = False


class SubmitRunResponse(BaseModel):
    run_id: str


class Envelope(BaseModel):
    """Error-envelope pattern: tools never raise; they return this."""

    ok: bool
    data: Any | None = None
    # List results carry a count, and empty lists a message saying why the
    # result may be empty — so an agent can tell "nothing exists" from
    # "wrong query" from "tool failure".
    count: int | None = None
    message: str | None = None
    error: str | None = None
    detail: str | None = None

    @classmethod
    def success(
        cls, data: Any, count: int | None = None, message: str | None = None
    ) -> dict[str, Any]:
        return cls(ok=True, data=data, count=count, message=message).model_dump(
            exclude_none=True
        )

    @classmethod
    def failure(cls, error: str, detail: str | None = None) -> dict[str, Any]:
        return cls(ok=False, error=error, detail=detail).model_dump(exclude_none=True)


# WorkerInfo.lease forward-references Lease (defined later); resolve it now.
WorkerInfo.model_rebuild()
