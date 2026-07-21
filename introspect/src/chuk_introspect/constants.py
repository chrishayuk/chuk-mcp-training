"""Names shared with the Rust proto (must match crates/chuk-train-proto/src/introspect.rs).

Key segment order is normative (spec §5.1), parsed positionally on both sides:

    introspect/<family>[/<qualifier>]/L{i}[@<corpus>]
"""

from __future__ import annotations

from typing import Final

# Script-contract env vars. METRICS matches chuk_train_proto::constants::script_env;
# PROBE_PLAN / PROBE_DIR join script_env at I1 (CP-delivered) — the names are
# already the contract (spec §5.3).
ENV_METRICS: Final = "CHUK_METRICS"
ENV_PROBE_PLAN: Final = "CHUK_PROBE_PLAN"
ENV_PROBE_DIR: Final = "CHUK_PROBE_DIR"

# The step field of a $CHUK_METRICS JSONL record (worker: METRIC_STEP_KEY).
METRIC_STEP_KEY: Final = "step"

INTROSPECT_METRIC_PREFIX: Final = "introspect/"
LAYER_SEGMENT_PREFIX: Final = "L"
CORPUS_SUFFIX_SEPARATOR: Final = "@"

# Pulse families (Tier 0).
FAMILY_ACT_NORM: Final = "act_norm"
FAMILY_GRAD_NORM: Final = "grad_norm"
FAMILY_DEAD_FRAC: Final = "dead_frac"
FAMILY_LOGIT_ENTROPY: Final = "logit_entropy"

# Library health families (model-global).
FAMILY_OVERHEAD_PCT: Final = "overhead_pct"
FAMILY_PROBE_ERROR: Final = "probe_error"


def metric_key(
    family: str,
    qualifier: str | None = None,
    layer: int | None = None,
    corpus: str | None = None,
) -> str:
    """Build a metric key per the normative grammar."""
    key = f"{INTROSPECT_METRIC_PREFIX}{family}"
    if qualifier is not None:
        key += f"/{qualifier}"
    if layer is not None:
        key += f"/{LAYER_SEGMENT_PREFIX}{layer}"
    if corpus is not None:
        key += f"{CORPUS_SUFFIX_SEPARATOR}{corpus}"
    return key


def layer_key(family: str, layer: int) -> str:
    """Per-layer key on the lab-standard corpus, e.g. ``introspect/act_norm/L4``."""
    return metric_key(family, layer=layer)


def global_key(family: str) -> str:
    """Model-global key, e.g. ``introspect/logit_entropy``."""
    return metric_key(family)
