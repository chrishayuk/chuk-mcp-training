"""Pydantic mirrors of the Rust wire types (crates/chuk-train-proto/src/wire.rs).

The Rust side is the source of truth; these models validate what crosses the
REST boundary so the MCP tools never hand raw dicts to the model.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .constants import DEFAULT_SHELL_TIMEOUT_S


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


class RunKind(StrEnum):
    SHELL = "shell"


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
