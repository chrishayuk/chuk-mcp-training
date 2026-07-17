"""Pydantic mirrors of the Rust wire types (crates/chuk-train-proto/src/wire.rs).

The Rust side is the source of truth; these models validate what crosses the
REST boundary so the MCP tools never hand raw dicts to the model.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

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


class WorkerInfo(BaseModel):
    id: str
    labels: list[str]
    hardware: Hardware
    state: WorkerState
    current_run: str | None = None
    joined_at: float
    last_seen: float
    heartbeat_age_s: float


class RunSummary(BaseModel):
    id: str
    name: str
    kind: RunKind
    state: RunState
    worker_id: str | None = None
    exit_code: int | None = None
    created_at: float
    updated_at: float


class RunRecord(RunSummary):
    spec: ShellSpec


class RunEvent(BaseModel):
    ts: float
    event: EventKind
    detail: dict[str, Any] = Field(default_factory=dict)


class LogsResponse(BaseModel):
    run_id: str
    lines: list[str]


class SubmitShellRequest(BaseModel):
    name: str
    command: str
    timeout_s: int = DEFAULT_SHELL_TIMEOUT_S


class SubmitRunRequest(BaseModel):
    name: str
    spec: TrainSpec


class BuildCodeUnitRequest(BaseModel):
    repo: str
    commit: str | None = None
    name: str | None = None


class PinCheckpointRequest(BaseModel):
    step: int
    name: str


class SubmitRunResponse(BaseModel):
    run_id: str


class Envelope(BaseModel):
    """Error-envelope pattern: tools never raise; they return this."""

    ok: bool
    data: Any | None = None
    error: str | None = None
    detail: str | None = None

    @classmethod
    def success(cls, data: Any) -> dict[str, Any]:
        return cls(ok=True, data=data).model_dump(exclude_none=True)

    @classmethod
    def failure(cls, error: str, detail: str | None = None) -> dict[str, Any]:
        return cls(ok=False, error=error, detail=detail).model_dump(exclude_none=True)
