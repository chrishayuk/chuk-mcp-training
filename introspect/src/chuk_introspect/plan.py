"""ProbePlan v3 (spec §3, v0.4) — the fingerprinted capture contract.

I0 consumes ``version`` / ``fingerprint`` / ``model`` / ``pulse`` / ``budget``;
the snapshot-era blocks (corpus, snapshot, whitening, jspace) are validated so
a full plan round-trips today, but nothing reads them until I1+. At I1 this
schema gains a Rust twin in chuk-train-proto plus a JSON-Schema contract test.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

PROBE_PLAN_VERSION = 3


class DeterminismMode(str, Enum):
    STRICT = "strict"
    TOLERANT = "tolerant"


class PulseMetric(str, Enum):
    ACT_NORM = "act_norm"
    GRAD_NORM = "grad_norm"
    DEAD_FRAC = "dead_frac"
    LOGIT_ENTROPY = "logit_entropy"


LAYERS_ALL = "all"


class ModelSpec(BaseModel):
    d_model: int
    n_layers: int
    dtype: str = "auto"


class DeterminismSpec(BaseModel):
    mode: DeterminismMode = DeterminismMode.TOLERANT
    tolerance: dict[str, float] | None = None


class CorpusSplits(BaseModel):
    a: int
    b: int


class CorpusSpec(BaseModel):
    id: str
    sha256: str
    n_prompts: int
    seq_len: int
    splits: CorpusSplits


class PulseSpec(BaseModel):
    every_steps: int = Field(default=1, ge=1)
    metrics: list[PulseMetric] = Field(
        default_factory=lambda: list(PulseMetric)
    )
    layers: Literal["all"] | list[int] = LAYERS_ALL


class ProbeSpec(BaseModel):
    name: str
    labels_from: str


class SnapshotSpec(BaseModel):
    on_checkpoint: bool = True
    corpus_subsample: int
    layers: list[int] | str
    position: str = "last"
    capture: list[str]
    probes: list[ProbeSpec] = Field(default_factory=list)


class WhiteningSpec(BaseModel):
    source: Literal["jspace_rows"] = "jspace_rows"
    min_rows: int = 512
    reference: str = "self"  # "self" | "pinned:<ckpt_id>"
    emit_spectrum: bool = True


class RowPositionsSpec(BaseModel):
    completion_len: int
    exclude_filler: bool = True


class JlensFitSpec(BaseModel):
    dim_batch: int = 8
    skip_first: int = 16
    source_layers: list[int]


class JspaceSpec(BaseModel):
    rows_on_snapshot: bool = True
    row_targets: str = "model_top1"
    row_target_reference: str = "self"  # "self" | "pinned:<ckpt_id>"
    row_positions: RowPositionsSpec
    full_fit_mode: Literal["inline", "eval_job", "auto"] = "auto"
    milestone_every_ckpts: int = 10
    fit: JlensFitSpec


class BudgetSpec(BaseModel):
    pulse_overhead_pct_max: float = 2.0
    snapshot_forward_seconds_max: float = 60.0
    snapshot_rows_seconds_max: float = 120.0
    milestone_seconds_max: float = 3600.0


class ProbePlan(BaseModel):
    version: Literal[3]
    fingerprint: str | None = None
    model: ModelSpec
    determinism: DeterminismSpec = Field(default_factory=DeterminismSpec)
    pulse: PulseSpec | None = None
    # Snapshot-era blocks: optional (a pulse-only plan is valid, spec §1
    # principle 8); consumed from I1 onward.
    corpus: CorpusSpec | None = None
    snapshot: SnapshotSpec | None = None
    whitening: WhiteningSpec | None = None
    jspace: JspaceSpec | None = None
    budget: BudgetSpec = Field(default_factory=BudgetSpec)

    @model_validator(mode="after")
    def _snapshot_needs_corpus(self) -> "ProbePlan":
        if self.snapshot is not None and self.corpus is None:
            raise ValueError("a plan with a snapshot block requires a corpus block")
        if self.jspace is not None and self.snapshot is None:
            raise ValueError("a plan with a jspace block requires a snapshot block")
        return self

    def pulse_layer_indices(self) -> list[int]:
        """Resolve the pulse layer selection against the declared model."""
        if self.pulse is None:
            return []
        if self.pulse.layers == LAYERS_ALL:
            return list(range(self.model.n_layers))
        return list(self.pulse.layers)

    @classmethod
    def load(cls, path: str | Path) -> "ProbePlan":
        return cls.model_validate(json.loads(Path(path).read_text()))
